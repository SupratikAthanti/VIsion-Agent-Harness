"""
TEV1 decision router for the hybrid backend.

Asks the local TEV1 model (served by Ollama at /v1/systemone) which perception
method fits a request:
  - "ocr"    : target is identified by visible text (button, link, label)
  - "yolo"   : target is an icon or graphic without readable text
  - "vision" : general question about the screen, not a click target
"""

import json
import urllib.request
from typing import Dict

OLLAMA_BASE = "http://127.0.0.1:11434"
TEV1_URL = f"{OLLAMA_BASE}/v1/systemone"
TEV1_MODEL = "tev1:0.8b"
TEV1_TIMEOUT = 300  # seconds; CPU inference took ~44s on the first call

METHOD_CRITERIA = {
    "ocr": "The target is a button, link, tab, label or field identified by visible text on screen.",
    "yolo": "The target is an icon or graphic with no readable text, such as a hamburger menu, close X or gear icon.",
    "vision": "The request is a general question about the screen content, not a click target.",
}


class TEV1Error(RuntimeError):
    """Raised when TEV1 is unreachable, returns an error, or gives an unknown answer."""


def check_tev1_available(timeout: int = 5) -> str:
    """
    Healthcheck: confirm Ollama is up and the TEV1 model is installed.
    Returns the installed model name, or raises TEV1Error.
    """
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=timeout) as resp:
            models = json.loads(resp.read().decode("utf-8")).get("models", [])
    except Exception as e:
        raise TEV1Error(f"Ollama not reachable on 127.0.0.1:11434: {e}") from e

    for m in models:
        name = m.get("name", "")
        if name.startswith("tev1"):
            return name
    raise TEV1Error("TEV1 model not found in Ollama (expected tev1:0.8b)")


def route_task(user_query: str) -> Dict:
    """
    Decide which perception method to use for `user_query`.

    Returns a dict:
        method        - "ocr", "yolo" or "vision"
        confidence    - TEV1 probability of the chosen method (0.0-1.0)
        probabilities - probability per method
    Raises TEV1Error on any failure, so the caller can fall back.
    """
    payload = {
        "model": TEV1_MODEL,
        "state": user_query,
        "questions": {
            "method": {
                "type": "choice",
                "instructions": (
                    "Which perception method should locate the target in the user's request? "
                    "Choose ocr for visible text targets, yolo for icon-only targets, "
                    "vision for general questions."
                ),
                "criteria": METHOD_CRITERIA,
            }
        },
    }
    req = urllib.request.Request(
        TEV1_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TEV1_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise TEV1Error(f"TEV1 request failed: {e}") from e

    if "error" in data:
        raise TEV1Error(f"TEV1 error: {data['error']}")

    answer = data.get("answers", {}).get("method", {})
    choice = answer.get("choice")
    probabilities = answer.get("probabilities", {})
    if choice not in METHOD_CRITERIA:
        raise TEV1Error(f"TEV1 returned unknown method: {choice!r}")

    return {
        "method": choice,
        "confidence": float(probabilities.get(choice, 0.0)),
        "probabilities": probabilities,
    }
