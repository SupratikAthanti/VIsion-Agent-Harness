"""AT-SPI accessibility bridge for structured desktop application automation.

Provides position-independent desktop control modeled after OS accessibility trees:
- Inspect applications and accessibility trees (`list_apps`, `snapshot`, `snapshot_all`)
- Qualified snapshot references: `@<snapshot_id>:e<N>`
- Find elements by name, role, text content, and state
- Direct AT-SPI Action execution (`click`, `press`, `activate`, etc.)
- Direct AT-SPI EditableText setting (`set_text_contents`, `insert_text`)
- Element reading (accessible name, role, state, text content, value)
- Focus management via Component interface (`grab_focus`)
- Headless-first with seamless fallback to xdotool/wmctrl when needed
- Deterministic mock backend support for headless/CI test environments
"""

from functools import lru_cache
import os
import re
import sys
import time
import uuid
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

# Ensure Debian/Ubuntu dist-packages is in sys.path if system python3-gi is installed
for dist_path in ("/usr/lib/python3/dist-packages", "/usr/local/lib/python3/dist-packages"):
    if os.path.exists(dist_path) and dist_path not in sys.path:
        sys.path.append(dist_path)

MAX_TREE_DEPTH = 30
MAX_TREE_ELEMENTS = 10000

# Snapshot storage: snapshot_id -> Dict mapping ref_id -> Dict
_SNAPSHOTS: Dict[str, Dict[str, Any]] = {}
_LAST_SNAPSHOT_ID: Optional[str] = None

# Custom test mock desktop provider (for deterministic tests)
_MOCK_DESKTOP: Optional[Any] = None


def set_mock_desktop(mock_desktop: Optional[Any]) -> None:
    """Set a mock desktop tree provider for testing without a live GUI or D-Bus session."""
    global _MOCK_DESKTOP
    _MOCK_DESKTOP = mock_desktop
    _backend.cache_clear()


def get_mock_desktop() -> Optional[Any]:
    """Return the active mock desktop, if set."""
    return _MOCK_DESKTOP


@lru_cache(maxsize=1)
def _backend():
    """Return (module, api-kind), or None if no AT-SPI bindings or active bus exist."""
    if _MOCK_DESKTOP is not None:
        return _MOCK_DESKTOP, "mock"

    # Avoid fatal SIGTRAP abort when calling Atspi without an active D-Bus session or AT-SPI bus
    if not (os.environ.get("DBUS_SESSION_BUS_ADDRESS") or os.environ.get("AT_SPI_BUS_ADDRESS")):
        return None

    try:
        import gi
        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi
        return Atspi, "gi"
    except (ImportError, ValueError):
        pass
    try:
        import pyatspi
        return pyatspi, "pyatspi"
    except ImportError:
        return None


def _desktop(backend):
    module, api = backend
    if api == "mock":
        return module
    return module.get_desktop(0) if api == "gi" else module.Registry.getDesktop(0)


def _child_count(accessible) -> int:
    try:
        return int(accessible.get_child_count())
    except Exception:
        try:
            return int(accessible.childCount)
        except Exception:
            try:
                children = getattr(accessible, "children", None)
                return len(children) if children is not None else 0
            except Exception:
                return 0


def _child_at(accessible, index: int):
    try:
        return accessible.get_child_at_index(index)
    except Exception:
        try:
            return accessible[index]
        except Exception:
            try:
                return accessible.children[index]
            except Exception:
                return None


def _name(accessible) -> str:
    try:
        return str(accessible.get_name() or "")
    except Exception:
        try:
            return str(accessible.name or "")
        except Exception:
            return ""


def _role(accessible) -> str:
    try:
        return str(accessible.get_role_name() or "")
    except Exception:
        try:
            return str(accessible.getRoleName() or "")
        except Exception:
            return ""


def _description(accessible) -> str:
    try:
        return str(accessible.get_description() or "")
    except Exception:
        try:
            return str(accessible.description or "")
        except Exception:
            return ""


def _states(accessible, backend) -> List[str]:
    """Retrieve string list of active states (e.g., 'focused', 'showing', 'enabled')."""
    states = []
    try:
        if backend[1] == "mock":
            if hasattr(accessible, "get_states"):
                return list(accessible.get_states())
            return list(getattr(accessible, "states", []))

        state_set = accessible.get_state_set()
        if hasattr(state_set, "get_states"):
            for s in state_set.get_states():
                states.append(str(s).lower())
        else:
            # Check common states by name
            for sname in ("FOCUSED", "SHOWING", "VISIBLE", "ENABLED", "CHECKED", "EDITABLE", "ACTIVE"):
                st_attr = getattr(backend[0].StateType, sname, None) if backend[1] == "gi" else getattr(backend[0], f"STATE_{sname}", None)
                if st_attr is not None and state_set.contains(st_attr):
                    states.append(sname.lower())
    except Exception:
        pass
    return states


def _text_content(accessible) -> Optional[str]:
    """Retrieve text content via AtspiText interface if supported."""
    try:
        if hasattr(accessible, "get_text"):
            # Check if this is an AtspiText interface or direct mock
            try:
                return str(accessible.get_text(0, -1) or "")
            except TypeError:
                return str(accessible.get_text() or "")
        text_iface = getattr(accessible, "get_text_iface", None) or getattr(accessible, "queryText", None)
        if text_iface:
            ti = text_iface()
            return str(ti.get_text(0, -1) or "")
    except Exception:
        pass
    return None


def _value_content(accessible) -> Optional[str]:
    """Retrieve numeric/text value via AtspiValue interface if supported."""
    try:
        val_iface = getattr(accessible, "get_value_iface", None) or getattr(accessible, "queryValue", None)
        if val_iface:
            vi = val_iface()
            if hasattr(vi, "get_text"):
                txt = vi.get_text()
                if txt:
                    return str(txt)
            if hasattr(vi, "get_current_value"):
                return str(vi.get_current_value())
        if hasattr(accessible, "get_current_value"):
            return str(accessible.get_current_value())
    except Exception:
        pass
    return None


def _actions(accessible, backend) -> List[str]:
    """Retrieve list of available action names on this accessible."""
    actions = []
    try:
        if backend[1] == "mock":
            if hasattr(accessible, "get_action_names"):
                return list(accessible.get_action_names())
            return list(getattr(accessible, "actions", []))

        action_iface = getattr(accessible, "get_action_iface", None) or getattr(accessible, "queryAction", None)
        if action_iface:
            ai = action_iface()
            count = ai.get_n_actions() if backend[1] == "gi" else ai.nActions
            for i in range(int(count)):
                name = ai.get_action_name(i) if backend[1] == "gi" else ai.getName(i)
                if name:
                    actions.append(str(name))
    except Exception:
        pass
    return actions


def _bounds(accessible, backend) -> Optional[Tuple[int, int, int, int]]:
    try:
        if backend[1] == "mock":
            if hasattr(accessible, "get_bounds"):
                return accessible.get_bounds()
            return getattr(accessible, "bounds", None)

        module, api = backend
        if api == "gi":
            component = accessible.get_component_iface()
            rect = component.get_extents(module.CoordType.SCREEN)
            values = (rect.x, rect.y, rect.width, rect.height)
        else:
            rect = accessible.queryComponent().getExtents(module.DESKTOP_COORDS)
            values = (rect.x, rect.y, rect.width, rect.height)
        bounds = tuple(int(value) for value in values)
        return bounds if bounds[2] > 0 and bounds[3] > 0 else None
    except Exception:
        return None


def is_at_spi_available() -> bool:
    """Return whether AT-SPI bindings and a responsive accessibility desktop exist."""
    backend = _backend()
    if backend is None:
        return False
    try:
        desk = _desktop(backend)
        return desk is not None
    except Exception:
        return False


def _normalize(value: str) -> str:
    return "".join(character for character in (value or "").casefold() if character.isalnum())


def _walk_tree(accessible, backend, path: str, depth: int, counter: List[int],
               interactive_only: bool = False) -> Iterator[Dict[str, Any]]:
    """Walk an accessible hierarchy yielding element dictionaries."""
    if depth > MAX_TREE_DEPTH or counter[0] >= MAX_TREE_ELEMENTS or accessible is None:
        return
    counter[0] += 1

    name = _name(accessible)
    role = _role(accessible)
    desc = _description(accessible)
    bounds = _bounds(accessible, backend)
    states = _states(accessible, backend)
    text = _text_content(accessible)
    val = _value_content(accessible)
    acts = _actions(accessible, backend)

    is_interactive = bool(
        acts
        or "editable" in states
        or "selectable" in states
        or any(r in _normalize(role) for r in ("button", "entry", "text", "menu", "item", "link", "tab", "check", "combo"))
    )

    if not interactive_only or is_interactive or role in ("application", "window", "dialog", "frame"):
        if name or role or bounds or text:
            yield {
                "name": name,
                "role": role,
                "description": desc,
                "bounds": bounds,
                "states": states,
                "text": text,
                "value": val,
                "actions": acts,
                "ref_path": path,
                "raw_accessible": accessible,
            }

    count = _child_count(accessible)
    for index in range(count):
        if counter[0] >= MAX_TREE_ELEMENTS:
            break
        child = _child_at(accessible, index)
        if child is not None:
            child_path = f"{path}/{index}" if path else str(index)
            yield from _walk_tree(child, backend, child_path, depth + 1, counter, interactive_only)


def _resolve_path(path: str, backend) -> Any:
    """Resolve a slash-separated index path (e.g. '0/2/1') to an accessible object."""
    accessible = _desktop(backend)
    if not path:
        return accessible
    for part in path.split("/"):
        if not part:
            continue
        accessible = _child_at(accessible, int(part))
        if accessible is None:
            return None
    return accessible


def _resolve_ref(ref: str, backend=None):
    """Legacy helper: resolve ref path or snapshot ref to an accessible object."""
    backend = backend or _backend()
    if backend is None:
        return None
    if ref.startswith("@") or ":" in ref:
        elem = resolve_ref(ref)
        if elem:
            return elem.get("_accessible")
    return _resolve_path(ref, backend)


# ---------------------------------------------------------------------------
# High-Level Inspection & Snapshot API (agent-desktop model)
# ---------------------------------------------------------------------------

def list_apps() -> List[Dict[str, Any]]:
    """List running desktop applications exposed in the accessibility tree."""
    backend = _backend()
    if backend is None:
        return []
    try:
        desktop = _desktop(backend)
        apps = []
        count = _child_count(desktop)
        for i in range(count):
            app = _child_at(desktop, i)
            if app is None:
                continue
            name = _name(app)
            role = _role(app)
            # Find window titles under this app
            window_titles = []
            win_count = _child_count(app)
            for w in range(win_count):
                win = _child_at(app, w)
                if win:
                    wname = _name(win)
                    if wname:
                        window_titles.append(wname)
            apps.append({
                "app_id": str(i),
                "name": name or f"App_{i}",
                "role": role or "application",
                "windows": window_titles,
                "ref_path": str(i),
            })
        return apps
    except Exception:
        return []


def snapshot(app_name: Optional[str] = None, interactive_only: bool = True) -> Dict[str, Any]:
    """
    Take an accessibility snapshot of an app (or all apps).
    Assigns stable qualified refs: `@<snapshot_id>:e<N>`.
    Returns a structured envelope with tree and metadata.
    """
    global _LAST_SNAPSHOT_ID
    backend = _backend()
    if backend is None:
        return {
            "ok": False,
            "error": "AT-SPI accessibility is unavailable in the current environment.",
            "data": {},
        }

    snapshot_id = "s" + uuid.uuid4().hex[:8]
    root = _desktop(backend)
    target_path = ""

    # If app_name given, find that specific application node
    app_info = {"name": app_name or "Desktop", "role": "desktop"}
    if app_name:
        normalized_target = _normalize(app_name)
        count = _child_count(root)
        matched_app = None
        for i in range(count):
            child = _child_at(root, i)
            if child is not None:
                cname = _normalize(_name(child))
                if normalized_target == cname or (len(normalized_target) >= 3 and normalized_target in cname):
                    matched_app = child
                    target_path = str(i)
                    app_info["name"] = _name(child)
                    app_info["role"] = _role(child)
                    break
        if matched_app is not None:
            root = matched_app
        else:
            return {
                "ok": False,
                "error": f"Application '{app_name}' not found in desktop accessibility tree.",
                "data": {},
            }

    raw_elements = list(_walk_tree(root, backend, target_path, 0, [0], interactive_only=interactive_only))

    elements_map: Dict[str, Any] = {}
    tree_items: List[Dict[str, Any]] = []

    for idx, elem in enumerate(raw_elements, 1):
        ref_id = f"@{snapshot_id}:e{idx}"
        elem_data = {
            "ref_id": ref_id,
            "name": elem["name"],
            "role": elem["role"],
            "description": elem["description"],
            "bounds": elem["bounds"],
            "states": elem["states"],
            "text": elem["text"],
            "value": elem["value"],
            "actions": elem["actions"],
            "ref_path": elem["ref_path"],
            "_accessible": elem["raw_accessible"],
        }
        elements_map[ref_id] = elem_data
        # Public item without private _accessible
        public_item = {k: v for k, v in elem_data.items() if not k.startswith("_")}
        tree_items.append(public_item)

    _SNAPSHOTS[snapshot_id] = elements_map
    _LAST_SNAPSHOT_ID = snapshot_id

    # Keep snapshot cache bounded
    if len(_SNAPSHOTS) > 20:
        oldest_id = next(iter(_SNAPSHOTS))
        del _SNAPSHOTS[oldest_id]

    return {
        "ok": True,
        "snapshot_id": snapshot_id,
        "app": app_info["name"],
        "ref_count": len(elements_map),
        "elements": tree_items,
    }


def snapshot_all(interactive_only: bool = True) -> Dict[str, Any]:
    """Capture snapshot across all running desktop applications."""
    return snapshot(app_name=None, interactive_only=interactive_only)


def resolve_ref(ref: str) -> Optional[Dict[str, Any]]:
    """Resolve a qualified ref `@<snapshot_id>:e<N>` or bare `e<N>` to its element data."""
    ref = ref.strip()
    if not ref:
        return None

    if ref.startswith("@") and ":" in ref:
        snapshot_id, ref_num = ref[1:].split(":", 1)
        snap = _SNAPSHOTS.get(snapshot_id)
        if snap:
            return snap.get(ref)
        return None

    # If bare @e8 or e8, use last snapshot if available
    if _LAST_SNAPSHOT_ID and _LAST_SNAPSHOT_ID in _SNAPSHOTS:
        snap = _SNAPSHOTS[_LAST_SNAPSHOT_ID]
        bare = ref.lstrip("@")
        qualified = f"@{_LAST_SNAPSHOT_ID}:{bare}"
        return snap.get(qualified)

    return None


# ---------------------------------------------------------------------------
# Element Lookup by Semantic Attributes
# ---------------------------------------------------------------------------

def find_elements(
    name: Optional[str] = None,
    role: Optional[str] = None,
    text: Optional[str] = None,
    app_name: Optional[str] = None,
    exact: bool = False,
) -> List[Dict[str, Any]]:
    """
    Find matching accessible elements by name, role, or text content.
    Takes a fresh snapshot if none exists or app_name is specified.
    """
    snap_result = snapshot(app_name=app_name, interactive_only=False)
    if not snap_result.get("ok"):
        return []

    elements = snap_result.get("elements", [])
    matches = []

    norm_name = _normalize(name) if name else None
    norm_role = _normalize(role) if role else None
    norm_text = _normalize(text) if text else None

    for elem in elements:
        ename = _normalize(elem.get("name", ""))
        erole = _normalize(elem.get("role", ""))
        etext = _normalize(elem.get("text", "") or "")

        if norm_name:
            if exact and ename != norm_name:
                continue
            if not exact and norm_name not in ename:
                continue

        if norm_role:
            if exact and erole != norm_role:
                continue
            if not exact and norm_role not in erole:
                continue

        if norm_text:
            if exact and etext != norm_text:
                continue
            if not exact and norm_text not in etext:
                continue

        matches.append(elem)

    return matches


def find_element_by_name(name: str, app_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Find an element by name (exact actionable match preferred, then exact, then substring)."""
    norm_query = _normalize(name)
    if len(norm_query) < 2:
        return None

    elements = find_elements(app_name=app_name)
    exact = [e for e in elements if _normalize(e.get("name", "")) == norm_query]
    actionable_roles = ("button", "menuitem", "link", "tab", "pushbutton", "togglebutton", "entry")
    exact_actionable = [e for e in exact if any(r in _normalize(e.get("role", "")) for r in actionable_roles)]
    if exact_actionable:
        return exact_actionable[0]
    if exact:
        return exact[0]

    partial = [e for e in elements if norm_query in _normalize(e.get("name", ""))]
    if partial:
        # Prefer bounded elements
        bounded = [e for e in partial if e.get("bounds")]
        return bounded[0] if bounded else partial[0]

    # Try matching text content
    text_matches = [e for e in elements if norm_query in _normalize(e.get("text", "") or "")]
    if text_matches:
        bounded = [e for e in text_matches if e.get("bounds")]
        return bounded[0] if bounded else text_matches[0]

    return None


def find_element_by_role(role: str, app_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Find elements whose role contains the role query."""
    return find_elements(role=role, app_name=app_name, exact=False)


# ---------------------------------------------------------------------------
# Direct Actions via AT-SPI Interfaces (Action, EditableText, Component)
# ---------------------------------------------------------------------------

def _get_element_accessible(elem_or_ref: Union[str, Dict[str, Any]]) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Resolve element dict or ref to (accessible_object, element_dict)."""
    backend = _backend()
    if backend is None:
        return None, None

    if isinstance(elem_or_ref, dict):
        ref_id = elem_or_ref.get("ref_id")
        if ref_id:
            resolved = resolve_ref(ref_id)
            if resolved and "_accessible" in resolved:
                return resolved["_accessible"], resolved
        if "_accessible" in elem_or_ref:
            return elem_or_ref["_accessible"], elem_or_ref
        ref_path = elem_or_ref.get("ref_path")
        if ref_path is not None:
            acc = _resolve_path(ref_path, backend)
            return acc, elem_or_ref
        return None, elem_or_ref

    if isinstance(elem_or_ref, str):
        # Check if ref or path
        if elem_or_ref.startswith("@") or ":" in elem_or_ref:
            resolved = resolve_ref(elem_or_ref)
            if resolved:
                return resolved.get("_accessible"), resolved
        # Otherwise treat as path or element name
        if "/" in elem_or_ref or elem_or_ref.isdigit():
            acc = _resolve_path(elem_or_ref, backend)
            return acc, {"ref_path": elem_or_ref}
        found = find_element_by_name(elem_or_ref)
        if found:
            return _get_element_accessible(found)

    return None, None


def click_element(elem_or_ref: Union[str, Dict[str, Any]]) -> bool:
    """
    Invoke an accessible element's action (click, press, activate) without physical cursor movement.
    Returns True if an AT-SPI action was successfully invoked.
    """
    backend = _backend()
    if backend is None or not elem_or_ref:
        return False

    try:
        accessible = _resolve_ref(elem_or_ref, backend) if isinstance(elem_or_ref, str) else None
    except Exception:
        accessible = None

    if accessible is None:
        accessible, _ = _get_element_accessible(elem_or_ref)
    if accessible is None:
        return False

    try:
        if backend[1] == "mock":
            if hasattr(accessible, "do_action"):
                return bool(accessible.do_action("click"))
            return False

        if backend[1] == "gi":
            action = accessible.get_action_iface()
            action_count = action.get_n_actions()
            get_name = action.get_action_name
            do_action = action.do_action
        else:
            action = accessible.queryAction()
            action_count = action.nActions
            get_name = action.getName
            do_action = action.doAction

        preferred_actions = ("click", "press", "activate", "jump", "open")
        for preferred in preferred_actions:
            for index in range(int(action_count)):
                if str(get_name(index) or "").casefold() == preferred:
                    return bool(do_action(index))
        return False
    except Exception:
        return False


def set_text_contents(elem_or_ref: Union[str, Dict[str, Any]], text: str) -> bool:
    """
    Directly set text content using Atspi.EditableText interface (no focus or typing needed).
    Returns True on success.
    """
    backend = _backend()
    if backend is None:
        return False

    accessible, elem = _get_element_accessible(elem_or_ref)
    if accessible is None:
        return False

    try:
        if backend[1] == "mock":
            if hasattr(accessible, "set_text_contents"):
                return bool(accessible.set_text_contents(text))
            return False

        if backend[1] == "gi":
            editable = accessible.get_editable_text_iface()
            return bool(editable.set_text_contents(text))
        else:
            editable = accessible.queryEditableText()
            return bool(editable.setTextContents(text))
    except Exception:
        # Try insert_text fallback
        try:
            if backend[1] == "gi":
                editable = accessible.get_editable_text_iface()
                editable.delete_text(0, -1)
                return bool(editable.insert_text(0, text, len(text)))
            else:
                editable = accessible.queryEditableText()
                editable.deleteText(0, -1)
                return bool(editable.insertText(0, text, len(text)))
        except Exception:
            return False


def insert_text(elem_or_ref: Union[str, Dict[str, Any]], text: str, position: int = 0) -> bool:
    """Insert text at specific character index using Atspi.EditableText."""
    backend = _backend()
    if backend is None:
        return False

    accessible, elem = _get_element_accessible(elem_or_ref)
    if accessible is None:
        return False

    try:
        if backend[1] == "mock":
            if hasattr(accessible, "insert_text"):
                return bool(accessible.insert_text(position, text))
            return False

        if backend[1] == "gi":
            editable = accessible.get_editable_text_iface()
            return bool(editable.insert_text(position, text, len(text)))
        else:
            editable = accessible.queryEditableText()
            return bool(editable.insertText(position, text, len(text)))
    except Exception:
        return False


def grab_focus(elem_or_ref: Union[str, Dict[str, Any]]) -> bool:
    """Focus an element using the Atspi.Component interface."""
    backend = _backend()
    if backend is None:
        return False

    accessible, elem = _get_element_accessible(elem_or_ref)
    if accessible is None:
        return False

    try:
        if backend[1] == "mock":
            if hasattr(accessible, "grab_focus"):
                return bool(accessible.grab_focus())
            return False

        if backend[1] == "gi":
            component = accessible.get_component_iface()
            return bool(component.grab_focus())
        else:
            component = accessible.queryComponent()
            return bool(component.grabFocus())
    except Exception:
        return False


def read_element(elem_or_ref: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Read full semantic state from an element:
    name, role, description, states, text content, value, bounds, actions.
    """
    backend = _backend()
    if backend is None:
        return None

    accessible, elem = _get_element_accessible(elem_or_ref)
    if accessible is None:
        return None

    return {
        "name": _name(accessible),
        "role": _role(accessible),
        "description": _description(accessible),
        "states": _states(accessible, backend),
        "text": _text_content(accessible),
        "value": _value_content(accessible),
        "bounds": _bounds(accessible, backend),
        "actions": _actions(accessible, backend),
    }


def get_element_bounds(elem_or_ref: Union[str, Dict[str, Any]]) -> Optional[Tuple[int, int, int, int]]:
    """Return screen-pixel bounds (x, y, width, height) of an accessible element."""
    backend = _backend()
    if backend is None:
        return None

    accessible, elem = _get_element_accessible(elem_or_ref)
    if accessible is None:
        return None

    return _bounds(accessible, backend)


def get_accessible_tree() -> List[Dict[str, Any]]:
    """Legacy helper: return all accessible elements as a flat list."""
    backend = _backend()
    if backend is None:
        return []
    try:
        desktop = _desktop(backend)
        raw = _walk_tree(desktop, backend, "", 0, [0], interactive_only=False)
        return [{"name": r["name"], "role": r["role"], "bounds": r["bounds"], "ref": r["ref_path"]} for r in raw]
    except Exception:
        return []
