"""
Rule-based intent classifier for the Vision Agent Harness.

Turns a request such as "Type hello in the search box" into a structured dict:

    {"action": "TYPE", "target": "search box",
     "params": {"text": "hello"}, "needs_clarification": None}

Actions: CLICK, TYPE, NAVIGATE, SCROLL, DRAG, WAIT, MULTISTEP, UNKNOWN.
Pure regex matching, so classification takes well under 1 ms. No model is loaded.
"""

import re
from typing import Dict, List, Optional, Tuple

ACTIONS = ("CLICK", "TYPE", "NAVIGATE", "SCROLL", "DRAG", "WAIT", "MULTISTEP", "UNKNOWN")

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
    r"^(?:please\s+)?(?:click|tap|press|hit|select|type|enter|write|input|go|open|"
    r"scroll|wait|drag|find|locate|navigate|visit)\b",
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
    r"^(?:please\s+)?(where\s+is|where\s+are|where's|click|tap|press|hit|select|push|"
    r"find|locate|choose|open|go\s+to)\s*(?:on\s+)?(.*)$",
    re.IGNORECASE,
)
_URL = re.compile(
    r"^(?:https?://)?(?:localhost(?::\d+)?|(?:[a-z0-9-]+\.)+[a-z]{2,})(?:[/:?#]\S*)?$",
    re.IGNORECASE,
)
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


def _split_steps(text: str) -> List[str]:
    """Split 'click A then type B' into ['click A', 'type B']. Returns [text] when there is no split."""
    parts = [p.strip() for p in _STEP_SPLIT.split(text) if p and p.strip()]
    if len(parts) > 1:
        return parts
    # "click login and type bob": split on 'and' only when both sides start with an action verb.
    pieces = [p.strip() for p in _AND_SPLIT.split(text) if p.strip()]
    if len(pieces) == 2 and all(_ACTION_VERB.match(p) for p in pieces):
        return pieces
    return [text]


def _classify_single(text: str) -> Dict:
    text = text.strip()

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
        target, kind = _split_kind(_clean_target(target_raw))
        return _result("TYPE", target, {"text": typed, **kind})

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
    target, kind = _split_kind(_clean_target(raw_target))
    params = {**kind}
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
