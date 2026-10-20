"""Optional AT-SPI accessibility bridge for locating named desktop UI elements.

The bridge has no mandatory dependencies: it prefers the GObject-introspection
Atspi namespace and falls back to the common ``pyatspi`` module when available.
Accessibility is queried lazily so importing the CLI never starts or blocks on
an accessibility bus.
"""

from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Tuple

MAX_TREE_DEPTH = 30
MAX_TREE_ELEMENTS = 10000


@lru_cache(maxsize=1)
def _backend():
    """Return (module, api-kind), or None if no AT-SPI bindings are installed."""
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
    return module.get_desktop(0) if api == "gi" else module.Registry.getDesktop(0)


def _child_count(accessible) -> int:
    try:
        return int(accessible.get_child_count())
    except Exception:
        try:
            return int(accessible.childCount)
        except Exception:
            return 0


def _child_at(accessible, index: int):
    try:
        return accessible.get_child_at_index(index)
    except Exception:
        return accessible[index]


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


def _bounds(accessible, backend) -> Optional[Tuple[int, int, int, int]]:
    module, api = backend
    try:
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


def _walk(accessible, backend, path: str, depth: int, count: List[int]) -> Iterator[Dict]:
    if depth > MAX_TREE_DEPTH or count[0] >= MAX_TREE_ELEMENTS or accessible is None:
        return
    count[0] += 1
    try:
        name = _name(accessible)
        role = _role(accessible)
        bounds = _bounds(accessible, backend)
        if name or bounds:
            yield {"name": name, "role": role, "bounds": bounds, "ref": path}
        children = _child_count(accessible)
        for index in range(children):
            if count[0] >= MAX_TREE_ELEMENTS:
                break
            try:
                child = _child_at(accessible, index)
                yield from _walk(child, backend, f"{path}/{index}", depth + 1, count)
            except Exception:
                continue
    except Exception:
        return


def get_accessible_tree() -> List[Dict]:
    """Return accessible elements as {name, role, bounds, ref}; [] if unavailable."""
    backend = _backend()
    if backend is None:
        return []
    try:
        desktop = _desktop(backend)
        return list(_walk(desktop, backend, "", 0, [0]))
    except Exception:
        return []


def is_at_spi_available() -> bool:
    """Return whether AT-SPI bindings and a responsive accessibility desktop exist."""
    backend = _backend()
    if backend is None:
        return False
    try:
        _desktop(backend)
        return True
    except Exception:
        return False


def _normalize(value: str) -> str:
    return "".join(character for character in (value or "").casefold() if character.isalnum())


def find_element_by_name(name: str) -> Optional[Dict]:
    """Find an accessible element by exact normalized name, then a safe substring match."""
    query = _normalize(name)
    if len(query) < 2:
        return None
    elements = get_accessible_tree()
    exact = [element for element in elements if _normalize(element["name"]) == query and element["bounds"]]
    actionable_roles = ("button", "menuitem", "link", "tab", "pushbutton", "togglebutton")
    exact_actionable = [element for element in exact
                        if any(role in _normalize(element["role"]) for role in actionable_roles)]
    if exact_actionable:
        return exact_actionable[0]
    if exact:
        return exact[0]
    partial = [element for element in elements
               if element["bounds"] and query in _normalize(element["name"])]
    return partial[0] if partial else None


def find_element_by_role(role: str) -> List[Dict]:
    """Return accessible elements whose role name contains the normalized role query."""
    query = _normalize(role)
    if not query:
        return []
    return [element for element in get_accessible_tree()
            if element["bounds"] and query in _normalize(element["role"])]


def _resolve_ref(ref: str, backend):
    """Resolve a tree reference returned by this module to its accessible object."""
    accessible = _desktop(backend)
    for index in (int(part) for part in ref.split("/") if part):
        accessible = _child_at(accessible, index)
    return accessible


def get_element_bounds(ref: str) -> Optional[Tuple[int, int, int, int]]:
    """Resolve a tree reference returned by this module and return screen-pixel bounds."""
    backend = _backend()
    if backend is None:
        return None
    try:
        return _bounds(_resolve_ref(ref, backend), backend)
    except (ValueError, IndexError, TypeError):
        return None
    except Exception:
        return None


def click_element(ref: str) -> bool:
    """Invoke an accessible element's click/press action without moving the pointer."""
    backend = _backend()
    if backend is None or not ref:
        return False
    try:
        accessible = _resolve_ref(ref, backend)
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

        preferred_actions = ("click", "press", "activate")
        for preferred in preferred_actions:
            for index in range(int(action_count)):
                if str(get_name(index) or "").casefold() == preferred:
                    return bool(do_action(index))
        return False
    except Exception:
        return False
