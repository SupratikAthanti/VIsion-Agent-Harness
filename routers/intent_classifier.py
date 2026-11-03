"""
Rule-based intent classifier and router for the Vision Agent Harness.

Turns a request such as "Type hello in the search box" into a structured dict:

    {"action": "TYPE", "target": "search box",
     "params": {"text": "hello", "app": "Notes"}, "needs_clarification": None}

Actions: CLICK, TYPE, NAVIGATE, SCROLL, DRAG, WAIT, READ, FOCUS, SNAPSHOT, LIST_APPS, MULTISTEP, UNKNOWN.
Provides route_context(intent, active_context) to route deterministically between:
- "browser" (Playwright / CDP)
- "desktop" (AT-SPI / xdotool)
- "vision" (OCR / YOLO fallback)
"""

import re
from typing import Dict, List, Optional, Tuple

ACTIONS = (
    "CLICK", "TYPE", "NAVIGATE", "SCROLL", "DRAG", "WAIT",
    "READ", "FOCUS", "SNAPSHOT", "LIST_APPS", "MULTISTEP", "UNKNOWN"
)

# Trailing element-type words are removed from the target and kept in params["kind"].
_KIND_WORDS = ("button", "link", "icon", "tab")
# Words that refer to something on screen without naming it.
_PRONOUNS = {"it", "this", "that", "there", "them", "one", "something", "thing"}
# Keys that can be pressed on their own ("press enter").
_KEYS = {
    "enter": "Return", "return": "Return", "tab": "Tab",
    "escape": "Escape", "esc": "Escape", "space": "space", "backspace": "BackSpace",
}

_STEP_SPLIT = re.compile(
    r"\s*(?:;|[,]?\s*(?:and\s+)?(?:then|after that)\b)\s*",
    re.IGNORECASE,
)
_AND_SPLIT = re.compile(r"\s+and\s+", re.IGNORECASE)
_ACTION_VERB = re.compile(
    r"^(?:please\s+)?(?:click|tap|press|hit|select|mark|type|enter|write|input|go|open|"
    r"scroll|wait|drag|find|locate|navigate|visit|read|extract|show|focus|snapshot|list)\b",
    re.IGNORECASE,
)

# App scoping pattern: "click File in VS Code" / "type hello in search in Notes"
_APP_IN = re.compile(r"\s+(?:in|into|of|on\s+app)\s+([A-Z0-9][A-Za-z0-9_\-\s]+)$")

# Snapshot and inspect commands
_LIST_APPS = re.compile(r"^(?:please\s+)?(?:list\s+apps?|show\s+apps?|get\s+apps?|running\s+apps?)$", re.IGNORECASE)
_SNAPSHOT = re.compile(r"^(?:please\s+)?(?:snapshot|take\s+snapshot)(?:\s+(?:of|for|app)?\s*(.+))?$", re.IGNORECASE)
_FOCUS = re.compile(r"^(?:please\s+)?(?:focus|switch\s+to|bring\s+up)\s+(.+)$", re.IGNORECASE)

_READ = re.compile(
    r"^(?:please\s+)?(?:read|extract|show|get)\s*(?:the\s+)?(page|table|text|content|results?|title\s+bar|window)?(?:\s*(?:from|in|of|on)?\s+(.+))?$",
    re.IGNORECASE,
)
_WAIT = re.compile(
    r"^(?:please\s+)?(?:wait|pause|sleep)(?:\s+for)?(?:\s+(\d+(?:\.\d+)?))?\s*"
    r"(ms|milliseconds?|s|secs?|seconds?)?\s*$",
    re.IGNORECASE,
)
_SCROLL = re.compile(r"^(?:please\s+)?(?:scroll|page\s+(?:up|down))\b(.*)$", re.IGNORECASE)
_DRAG = re.compile(r"^(?:please\s+)?drag\s+(.+?)\s+(?:to|onto|on)\s+(.+)$", re.IGNORECASE)
_NAVIGATE = re.compile(r"^(?:please\s+)?(?:go\s+to|navigate\s+to|visit|browse\s+to|open)\s+(.+)$", re.IGNORECASE)
_NAV_VERB = re.compile(r"^(?:please\s+)?(?:navigate\s+to|visit|browse\s+to)\b", re.IGNORECASE)
_KEY_PRESS = re.compile(r"^(?:please\s+)?(?:press|hit|tap)\s+(enter|return|tab|escape|esc|space|backspace)$", re.IGNORECASE)
_TYPE = re.compile(r"^(?:please\s+)?(?:type|enter|write|input)\s+(.+)$", re.IGNORECASE)
_TYPE_BARE = re.compile(r"^(?:please\s+)?(?:type|enter|write|input)$", re.IGNORECASE)
_TYPE_QUOTED = re.compile(r"""^(["'])(.+?)\1(?:\s+(?:in|into|on)\s+(.+))?$""")
_TYPE_IN = re.compile(r"^(.+?)\s+(?:in|into|on)\s+(.+)$", re.IGNORECASE)
_CLICK = re.compile(
    r"^(?:please\s+)?(where\s+is|where\s+are|where's|click|tap|press|hit|select|mark|push|"
    r"find|locate|choose|open|go\s+to)\s*(?:on\s+)?(.*)$",
    re.IGNORECASE,
)
_URL = re.compile(
    r"^(?:https?://)?(?:localhost(?::\d+)?|(?:[a-z0-9-]+\.)+[a-z]{2,})(?:[/:?#]\S*)?$",
    re.IGNORECASE,
)
_REF_PATTERN = re.compile(r"^@s[a-f0-9]{8}:e\d+$|^@e\d+$|^e\d+$")
_KIND_SUFFIX = re.compile(r"\s+(" + "|".join(_KIND_WORDS) + r")$", re.IGNORECASE)
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an|my|your)\s+", re.IGNORECASE)
_KEYWORD_ELEMENT = re.compile(r"\b(button|link|icon|menu|tab|box|field)\b", re.IGNORECASE)


def _result(action: str, target: Optional[str] = None, params: Optional[Dict] = None,
            clarification: Optional[str] = None) -> Dict:
    return {
        "action": action,
        "target": target,
        "params": params or {},
        "needs_clarification": clarification,
    }


def _clean_target(raw: Optional[str]) -> Optional[str]:
    """Normalize an element name: strip punctuation, leading articles and filler words."""
    if not raw:
        return None
    text = raw.strip().strip("?.!,").strip()
    text = _LEADING_ARTICLE.sub("", text).strip()
    text = re.sub(r"\s+please$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+", " ", text)
    return text or None


def _extract_app_scope(target: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Extract app name if target has 'in <App>': 'File in VS Code' -> ('File', 'VS Code')."""
    if not target:
        return target, None
    m = _APP_IN.search(target)
    if m:
        app_name = m.group(1).strip()
        # Ensure it's not a common UI element like 'in search' or 'in the menu'
        if not re.match(r"^(?:search|box|field|menu|table|window|page)$", app_name, re.IGNORECASE):
            base_target = target[:m.start()].strip()
            return base_target, app_name
    return target, None


def _split_kind(target: Optional[str]) -> Tuple[Optional[str], Dict]:
    """Split a trailing element-type word off a target: 'gemini button' -> ('gemini', {'kind': 'button'})."""
    if not target:
        return None, {}
    m = _KIND_SUFFIX.search(target)
    if m and m.start() > 0:
        return target[:m.start()].strip(), {"kind": m.group(1).lower()}
    return target, {}


def _looks_like_url(token: str) -> bool:
    return bool(_URL.match(token.strip("\"'")))


def is_ref(token: Optional[str]) -> bool:
    """Return True if token matches a snapshot reference (@s12345678:e1, @e1, e1)."""
    return bool(token and _REF_PATTERN.match(token.strip()))


def _split_steps(text: str) -> List[str]:
    """Split 'click A then type B' or 'click A and type B' into individual action steps."""
    raw_parts = [p.strip() for p in _STEP_SPLIT.split(text) if p and p.strip()]
    if not raw_parts:
        return [text]

    results: List[str] = []
    for part in raw_parts:
        pieces = [p.strip() for p in _AND_SPLIT.split(part) if p.strip()]
        if len(pieces) > 1 and all(_ACTION_VERB.match(p) for p in pieces):
            results.extend(pieces)
        else:
            results.append(part)
    return results if results else [text]


def _classify_single(text: str) -> Dict:
    text = text.strip()

    if _LIST_APPS.match(text):
        return _result("LIST_APPS")

    m = _SNAPSHOT.match(text)
    if m:
        app_target = _clean_target(m.group(1))
        return _result("SNAPSHOT", target=app_target, params={"app": app_target} if app_target else {})

    m = _FOCUS.match(text)
    if m:
        target = _clean_target(m.group(1))
        return _result("FOCUS", target=target)

    m = _READ.match(text)
    if m:
        target_kind = (m.group(1) or "page").lower().replace(" ", "_")
        selector = m.group(2)
        base_sel, app_name = _extract_app_scope(selector)
        params = {"kind": target_kind}
        if app_name:
            params["app"] = app_name
        return _result("READ", target=base_sel, params=params)

    m = _WAIT.match(text)
    if m:
        amount = float(m.group(1)) if m.group(1) else 1.0
        unit = (m.group(2) or "s").lower()
        seconds = amount / 1000.0 if unit.startswith("m") else amount
        return _result("WAIT", params={"seconds": seconds})

    m = _SCROLL.match(text)
    if m:
        rest = m.group(1) or ""
        direction_m = re.search(r"\b(up|down|left|right)\b", text, re.IGNORECASE)
        direction = direction_m.group(1).lower() if direction_m else "down"
        amount_m = re.search(r"\b(\d+)\b", rest)
        amount = int(amount_m.group(1)) if amount_m else 1
        return _result("SCROLL", params={"direction": direction, "amount": amount})

    m = _DRAG.match(text)
    if m:
        source, _ = _split_kind(_clean_target(m.group(1)))
        dest, dest_kind = _split_kind(_clean_target(m.group(2)))
        if not source or not dest:
            return _result("DRAG", clarification="Say what to drag and where, e.g. 'drag the file to the trash'.")
        params = {"destination": dest, **dest_kind}
        return _result("DRAG", source, params)

    m = _KEY_PRESS.match(text)
    if m:
        return _result("TYPE", params={"key": _KEYS[m.group(1).lower()]})

    m = _NAVIGATE.match(text)
    if m:
        rest = m.group(1).strip()
        token = rest.split()[0] if rest else ""
        if _looks_like_url(token):
            url = token.strip("\"'").rstrip("?.!,")
            return _result("NAVIGATE", params={"url": url})
        if _NAV_VERB.match(text):
            name = _clean_target(rest)
            return _result("NAVIGATE", name, {"url": None},
                           clarification=f"Give a full address for '{name}', e.g. {name}.com." if name
                           else "Where should I navigate to? e.g. 'go to youtube.com'.")
        # "open settings" / "go to the settings menu" -> click on screen
        return _classify_click(rest, text)

    m = _TYPE_BARE.match(text)
    if m:
        return _result("TYPE", clarification="What should I type?")

    m = _TYPE.match(text)
    if m:
        rest = m.group(1).strip()
        q = _TYPE_QUOTED.match(rest)
        if q:
            typed, target_raw = q.group(2), q.group(3)
        else:
            tm = _TYPE_IN.match(rest)
            if tm:
                typed, target_raw = tm.group(1).strip(), tm.group(2)
            else:
                typed, target_raw = rest, None
        clean_raw = _clean_target(target_raw)
        base_target, app_scope = _extract_app_scope(clean_raw)
        target, kind = _split_kind(base_target)
        params = {"text": typed, **kind}
        if app_scope:
            params["app"] = app_scope
        return _result("TYPE", target, params)

    m = _CLICK.match(text)
    if m:
        return _classify_click(m.group(2), text, locate_only=m.group(1).lower().startswith("where"))

    # Keyword fallback: "the gemini button" with no verb is still a click target.
    if _KEYWORD_ELEMENT.search(text):
        target, kind = _split_kind(_clean_target(text))
        return _result("CLICK", target, kind)

    return _result("UNKNOWN", _clean_target(text),
                   clarification="I couldn't tell what to do. Try 'click <element>', 'type <text> in <field>', "
                                 "'go to <url>' or 'scroll down'.")


def _classify_click(raw_target: str, original: str, locate_only: bool = False) -> Dict:
    clean_raw = _clean_target(raw_target)
    base_target, app_scope = _extract_app_scope(clean_raw)
    target, kind = _split_kind(base_target)
    params = {**kind}
    if app_scope:
        params["app"] = app_scope
    if locate_only:
        params["locate_only"] = True
    if not target or target.lower() in _PRONOUNS:
        return _result("CLICK", None, params,
                       clarification="Which element should I click? Name it, e.g. 'click the gemini button'.")
    return _result("CLICK", target, params)


def classify_intent(query: str) -> Dict:
    """
    Classify a request. Returns {"action", "target", "params", "needs_clarification"}.

    `needs_clarification` is a message for the user when the request is ambiguous or
    incomplete; otherwise it is None. Multi-step requests return MULTISTEP with
    params["steps"] holding one classified dict per step.
    """
    text = re.sub(r"\s+", " ", (query or "")).strip().rstrip("?!.").strip()
    if not text:
        return _result("UNKNOWN", clarification="The request is empty.")

    steps = _split_steps(text)
    if len(steps) == 1:
        return _classify_single(steps[0])

    parsed = [_classify_single(step) for step in steps]
    first_question = next((p["needs_clarification"] for p in parsed if p["needs_clarification"]), None)
    return _result("MULTISTEP", params={"steps": parsed}, clarification=first_question)


# ---------------------------------------------------------------------------
# Routing Policy: Browser vs Desktop vs Fallback
# ---------------------------------------------------------------------------

_BROWSER_KEYWORDS = {
    "web", "website", "url", "page", "browser", "chrome", "chromium", "firefox",
    "search google", "google.com", "wikipedia", "youtube", "tab", "http", "https"
}


# Desktop system / application menus that should never be silently routed to a web page
_DESKTOP_OS_TERMS = {
    "file", "edit", "view", "window", "help", "terminal", "preferences", "settings",
    "save", "save as", "quit", "exit", "minimise", "minimize", "maximize", "close"
}


def route_context(intent: Dict, active_context: Optional[Dict] = None) -> Tuple[str, Optional[str]]:
    """
    Route an intent to 'browser', 'desktop', or 'vision', returning (route, clarification_required).

    Authoritative Routing Policy:
    1. Explicit desktop identifiers:
       - Ref format (@s...:e..., @e..., e...)
       - Action is LIST_APPS, SNAPSHOT, FOCUS
       - Target or params specify a non-browser app ('in VS Code', 'in Notes')
       - Target is an OS system/desktop menu bar item ('File', 'Edit', 'Preferences')
       -> ('desktop', None)
    2. Explicit browser identifiers:
       - Action is NAVIGATE to a URL
       - Target or URL explicitly mentions web page, URL, or browser terms
       - App parameter is explicitly a browser ('in Chrome', 'in Firefox')
       -> ('browser', None)
    3. Authoritative active window / focus verification:
       - If authoritative focused application context is known:
         If focused app is a browser -> ('browser', None)
         If focused app is a native desktop app -> ('desktop', None)
    4. Ambiguity detection:
       - If both browser and desktop apps are active and the target is generic without an app scope,
         returns ('ambiguous', clarification_prompt) to prevent silent misrouted execution.
       - Otherwise defaults safely to ('desktop', None) if AT-SPI available, else ('browser', None).
    """
    action = intent.get("action")
    target = (intent.get("target") or "").strip()
    target_lower = target.lower()
    params = intent.get("params") or {}
    app = (params.get("app") or "").strip()
    app_lower = app.lower()

    # 1. Desktop-specific actions and refs
    if action in ("LIST_APPS", "SNAPSHOT", "FOCUS"):
        return "desktop", None

    if is_ref(target):
        return "desktop", None

    # 2. Explicit app targeting
    if app:
        if any(b in app_lower for b in ("chrome", "chromium", "firefox", "browser", "playwright")):
            return "browser", None
        return "desktop", None

    # 3. Known desktop OS menu items
    if target_lower in _DESKTOP_OS_TERMS:
        return "desktop", None

    # 4. Explicit browser navigation or keywords
    if action == "NAVIGATE" and params.get("url"):
        return "browser", None

    if any(k in target_lower for k in _BROWSER_KEYWORDS) or _looks_like_url(target):
        return "browser", None

    # 5. Authoritative focus context
    ctx = active_context or {}
    focused_app = (ctx.get("focused_app") or ctx.get("current_app") or "").strip().lower()
    browser_open = ctx.get("browser_open", False)

    if focused_app:
        if any(b in focused_app for b in ("chrome", "chromium", "firefox", "browser")):
            return "browser", None
        return "desktop", None

    # 6. Ambiguity check: if browser is open and query is generic, ask for clarification
    if browser_open and target and not any(k in target_lower for k in _BROWSER_KEYWORDS):
        return "ambiguous", f"Target '{target}' is ambiguous between the active browser and desktop apps. Specify an app (e.g. '{action.lower()} {target} in <App>') or a URL."

    if browser_open:
        return "browser", None

    return "desktop", None
