"""
Annotation web server for the /annotate command.

Serves annotation_ui/index.html plus the current image on 127.0.0.1. Save
requests are handed to the CLI thread through a queue, so the CLI can ask
about overwrites in the terminal and perform the actual save.
"""

import asyncio
import mimetypes
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from template_manager import DEFAULT_CROP_SIZE, PROJECT_ROOT, validate_name

HOST = "127.0.0.1"
DEFAULT_PORT = 8080
PORT_TRIES = 10
UI_HTML = PROJECT_ROOT / "annotation_ui" / "index.html"
REPLY_TIMEOUT = 300  # seconds the browser waits for the CLI to decide
NO_STORE = {"Cache-Control": "no-store"}


@dataclass
class Submission:
    """A request from the browser, waiting for the CLI to act on it."""
    kind: str  # "save" or "cancel"
    name: str = ""
    x_pct: float = 0.0
    y_pct: float = 0.0
    crop_size: int = DEFAULT_CROP_SIZE
    _reply: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=1))

    def respond(self, payload: dict) -> None:
        self._reply.put(payload)


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, port))
            return True
        except OSError:
            return False


def _is_fraction(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 1.0


class AnnotationServer:
    def __init__(self, image_path: Path, base_port: int = DEFAULT_PORT, port_tries: int = PORT_TRIES):
        self.image_path = Path(image_path)
        self.base_port = base_port
        self.port_tries = port_tries
        self.port: Optional[int] = None
        self._submissions: "queue.Queue[Submission]" = queue.Queue()
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None
        self.app = self._build_app()

    def _build_app(self) -> FastAPI:
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @app.get("/annotate", response_class=HTMLResponse)
        def annotate_page():
            return HTMLResponse(UI_HTML.read_text(encoding="utf-8"), headers=NO_STORE)

        @app.get("/image")
        def current_image():
            media_type = mimetypes.guess_type(self.image_path.name)[0] or "application/octet-stream"
            return FileResponse(self.image_path, media_type=media_type, headers=NO_STORE)

        @app.post("/save-template")
        async def save_template(request: Request):
            try:
                data = await request.json()
            except ValueError:
                return JSONResponse({"status": "error", "message": "Invalid JSON"}, status_code=400)
            if not isinstance(data, dict):
                return JSONResponse({"status": "error", "message": "Invalid payload"}, status_code=400)

            try:
                name = validate_name(str(data.get("name", "")))
            except ValueError as e:
                return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

            x_pct, y_pct = data.get("x_pct"), data.get("y_pct")
            if not (_is_fraction(x_pct) and _is_fraction(y_pct)):
                return JSONResponse(
                    {"status": "error", "message": "Click on the image first"}, status_code=400
                )

            crop_size = data.get("crop_size", DEFAULT_CROP_SIZE)
            if not isinstance(crop_size, int) or not 8 <= crop_size <= 200:
                return JSONResponse({"status": "error", "message": "Invalid crop size"}, status_code=400)

            sub = Submission("save", name=name, x_pct=float(x_pct), y_pct=float(y_pct), crop_size=crop_size)
            self._submissions.put(sub)
            try:
                result = await _wait_for_reply(sub)
            except queue.Empty:
                return JSONResponse({"status": "error", "message": "Timed out waiting for the CLI"}, status_code=504)
            return JSONResponse(result)

        @app.post("/cancel")
        def cancel():
            self._submissions.put(Submission("cancel"))
            return JSONResponse({"status": "cancelled", "message": "Annotation cancelled"})

        return app

    def start(self) -> int:
        """Start the server in a background thread. Returns the port used (8080, 8081, ...)."""
        for port in range(self.base_port, self.base_port + self.port_tries):
            if not _port_free(port):
                continue
            config = uvicorn.Config(
                self.app, host=HOST, port=port, log_level="warning",
                access_log=False, lifespan="off",
            )
            server = uvicorn.Server(config)
            thread = threading.Thread(target=server.run, name="annotation-server", daemon=True)
            thread.start()

            deadline = time.time() + 5
            while time.time() < deadline and not server.started and thread.is_alive():
                time.sleep(0.05)
            if server.started:
                self._server, self._thread, self.port = server, thread, port
                return port
            server.should_exit = True

        raise RuntimeError(
            f"No free port between {self.base_port} and {self.base_port + self.port_tries - 1}"
        )

    @property
    def url(self) -> str:
        if self.port is None:
            raise RuntimeError("Server not started")
        return f"http://{HOST}:{self.port}/annotate"

    def next_submission(self, timeout: float = 1.0) -> Optional[Submission]:
        """Wait up to `timeout` seconds for a browser request. Returns None if nothing arrived."""
        try:
            return self._submissions.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            if self._thread is not None:
                self._thread.join(timeout=5)
            self._server = None
            self._thread = None

    def schedule_stop(self, delay: float = 5.0) -> None:
        """Shut the server down after `delay` seconds, so the browser can show its final message."""
        timer = threading.Timer(delay, self.stop)
        timer.daemon = True
        timer.start()


async def _wait_for_reply(sub: Submission) -> dict:
    # Block a worker thread (not the event loop) until the CLI responds.
    return await asyncio.to_thread(sub._reply.get, True, REPLY_TIMEOUT)
