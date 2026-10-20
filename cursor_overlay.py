"""A small, click-through cursor preview for desktop actions.

Tk runs on a dedicated daemon thread so callers can show and hide the overlay
without taking over the application's main thread. On X11, XShape makes the
window's visible region match the arrow and its input region empty.
"""

import ctypes
import ctypes.util
import math
import os
import queue
import threading
from typing import Any, List, Optional, Tuple


_WIDTH = 28
_HEIGHT = 38
_BLUE = "#168cff"
_ARROW = ((0, 0), (0, 29), (8, 21), (14, 35), (19, 33), (13, 19), (25, 19))


class _XRectangle(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_short),
        ("y", ctypes.c_short),
        ("width", ctypes.c_ushort),
        ("height", ctypes.c_ushort),
    ]


def _arrow_rectangles() -> List[_XRectangle]:
    """Rasterize the arrow silhouette into one XShape rectangle per scanline."""
    rectangles = []
    for y in range(_HEIGHT):
        scan_y = y + 0.5
        intersections = []
        for index, (x1, y1) in enumerate(_ARROW):
            x2, y2 = _ARROW[(index + 1) % len(_ARROW)]
            if (y1 <= scan_y < y2) or (y2 <= scan_y < y1):
                intersections.append(x1 + (scan_y - y1) * (x2 - x1) / (y2 - y1))
        intersections.sort()
        for left_edge, right_edge in zip(intersections[::2], intersections[1::2]):
            left = max(0, math.floor(left_edge))
            right = min(_WIDTH, math.ceil(right_edge))
            if right > left:
                rectangles.append(_XRectangle(left, y, right - left, 1))
    return rectangles


def _make_x11_clickthrough(window_id: int) -> bool:
    """Shape an X11 window to the arrow and give it an empty input region."""
    x11_name = ctypes.util.find_library("X11")
    xext_name = ctypes.util.find_library("Xext")
    if not x11_name or not xext_name:
        return False

    x11 = ctypes.CDLL(x11_name)
    xext = ctypes.CDLL(xext_name)
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    x11.XFlush.argtypes = [ctypes.c_void_p]
    display_name = os.environ.get("DISPLAY")
    display = x11.XOpenDisplay(display_name.encode() if display_name else None)
    if not display:
        return False

    try:
        event_base, error_base = ctypes.c_int(), ctypes.c_int()
        xext.XShapeQueryExtension.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        ]
        if not xext.XShapeQueryExtension(display, ctypes.byref(event_base), ctypes.byref(error_base)):
            return False

        major, minor = ctypes.c_int(), ctypes.c_int()
        xext.XShapeQueryVersion.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        ]
        if not xext.XShapeQueryVersion(display, ctypes.byref(major), ctypes.byref(minor)):
            return False
        if (major.value, minor.value) < (1, 1):
            return False

        query_tree = x11.XQueryTree
        query_tree.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
            ctypes.POINTER(ctypes.c_uint),
        ]
        query_tree.restype = ctypes.c_int
        x11.XFree.argtypes = [ctypes.c_void_p]

        def tree_for(window: int) -> Tuple[int, int, List[int]]:
            root_window, parent = ctypes.c_ulong(), ctypes.c_ulong()
            children = ctypes.POINTER(ctypes.c_ulong)()
            child_count = ctypes.c_uint()
            if not query_tree(
                display, window, ctypes.byref(root_window), ctypes.byref(parent),
                ctypes.byref(children), ctypes.byref(child_count),
            ):
                raise RuntimeError("could not inspect overlay X11 window tree")
            try:
                return (
                    root_window.value,
                    parent.value,
                    [children[i] for i in range(child_count.value)],
                )
            finally:
                if children:
                    x11.XFree(children)

        # Tk creates child windows for the toplevel and canvas. Shape the actual
        # toplevel and clear input on every child so none can capture the click.
        top_level = window_id
        while True:
            root_window, parent_window, _ = tree_for(top_level)
            if parent_window == root_window:
                break
            top_level = parent_window

        combine = xext.XShapeCombineRectangles
        combine.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(_XRectangle), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        combine.restype = None
        arrow_rectangles = _arrow_rectangles()
        rectangles = (_XRectangle * len(arrow_rectangles))(*arrow_rectangles)
        # ShapeBounding = 0, ShapeInput = 2, ShapeSet = 0, Unsorted = 0.
        combine(display, top_level, 0, 0, 0, rectangles, len(rectangles), 0, 0)
        pending = [top_level]
        while pending:
            window = pending.pop()
            _, _, children = tree_for(window)
            combine(display, window, 2, 0, 0, None, 0, 0, 0)
            pending.extend(children)
        x11.XFlush(display)
        return True
    finally:
        x11.XCloseDisplay(display)


class CursorOverlay:
    """Show a blue arrow at screen coordinates without intercepting input."""

    def __init__(self) -> None:
        self._commands: "queue.Queue[Tuple[str, Optional[int], Optional[int], threading.Event]]" = queue.Queue()
        self._ready = threading.Event()
        self._available = False
        self._thread = threading.Thread(target=self._run, name="cursor-overlay", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=2.0)

    def _run(self) -> None:
        root: Any = None
        try:
            import tkinter as tk

            root = tk.Tk()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.attributes("-alpha", 0.0)
            root.configure(bg=_BLUE)
            root.geometry(f"{_WIDTH}x{_HEIGHT}+0+0")
            canvas = tk.Canvas(
                root, width=_WIDTH, height=_HEIGHT, bg=_BLUE,
                borderwidth=0, highlightthickness=0,
            )
            canvas.pack()
            canvas.create_polygon(*sum(_ARROW, ()), fill=_BLUE, outline="#073f83", width=2)
            root.deiconify()
            root.update_idletasks()

            # Fail closed: do not display a window that could steal the real click.
            frame_id = int(str(root.tk.call("wm", "frame", root._w)), 0)
            if not _make_x11_clickthrough(frame_id):
                root.withdraw()
                self._ready.set()
                return

            self._available = True
            self._ready.set()

            def process_commands() -> None:
                while True:
                    try:
                        command, x, y, completed = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        if command == "show":
                            root.geometry(f"{_WIDTH}x{_HEIGHT}+{x}+{y}")
                            root.deiconify()
                            root.attributes("-alpha", 1.0)
                            root.lift()
                            root.update_idletasks()
                        else:
                            root.attributes("-alpha", 0.0)
                    finally:
                        completed.set()
                root.after(20, process_commands)

            root.after(20, process_commands)
            root.mainloop()
        except Exception:
            self._available = False
            self._ready.set()
            if root is not None:
                try:
                    root.destroy()
                except Exception:
                    pass

    def _send(self, command: str, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        if not self._ready.wait(timeout=0.1) or not self._available:
            return False
        completed = threading.Event()
        self._commands.put((command, x, y, completed))
        return completed.wait(timeout=1.0)

    def show_cursor(self, x: int, y: int) -> bool:
        """Show the blue arrow with its tip at the given absolute pixel position."""
        return self._send("show", int(x), int(y))

    def hide_cursor(self) -> bool:
        """Make the overlay fully transparent again."""
        return self._send("hide")
