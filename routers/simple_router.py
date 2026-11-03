"""Simple keyword-based router to bypass intent classifier for common cases.

This is the fast path for common commands. The intent classifier is only used
as a fallback for complex, ambiguous queries.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

CONFIG_PATH = Path(__file__).parent / "config.json"


def _load_config() -> Dict:
    """Load known websites and apps from config.json."""
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {"known_websites": {}, "known_apps": {}}


_CONFIG = _load_config()


def _is_url(text: str) -> bool:
    """Check if text looks like a URL."""
    url_pattern = re.compile(
        r'^(https?:\/\/)?'  # http:// or https://
        r'((([a-z\d]([a-z\d-]*[a-z\d])*)\.)+[a-z]{2,}|'  # domain
        r'((\d{1,3}\.){3}\d{1,3}))'  # IP
        r'(\:\d+)?(\/[-a-z\d%_.~+]*)*'  # port and path
        r'(\?[;&a-z\d%_.~+=-]*)?'  # query
        r'(\#[-a-z\d_]*)?$', re.IGNORECASE
    )
    return bool(url_pattern.match(text))


def _is_ref(text: str) -> bool:
    """Check if text starts with a ref like @snapshot_id:e1."""
    return text.startswith('@')


def route_simple(text: str) -> Optional[Tuple[str, Dict]]:
    """
    Fast path routing for common commands.

    Returns (route_type, params) or None if intent classifier should handle it.

    Route types:
    - "browser": URL-based or known website
    - "desktop_app": Launch or focus a known app
    - "desktop_ref": Action on a ref
    - "desktop_scope": Desktop action with app scope
    - None: Fall back to intent classifier
    """
    text_lower = text.lower().strip()

    # 1. Explicit commands
    if text_lower.startswith("/"):
        return ("command", {"cmd": text})

    # 2. Ref-based actions (highest priority)
    if _is_ref(text_lower):
        if "click" in text_lower:
            return ("desktop_ref", {"action": "click", "ref": text})
        elif "type" in text_lower:
            return ("desktop_ref", {"action": "type", "ref": text})
        return ("desktop_ref", {"action": "unknown", "ref": text})

    # 3. URL-based (browser)
    if _is_url(text_lower):
        return ("browser", {"url": text})

    # 4. Known websites
    for site_name, site_config in _CONFIG["known_websites"].items():
        if site_name in text_lower:
            # "open google" or just "google"
            if "open" in text_lower or text_lower == site_name:
                return ("browser", {
                    "url": site_config["url"],
                    "browser": site_config.get("browser", "chrome")
                })

    # 5. Known apps with "open <app>"
    for app_name, app_config in _CONFIG["known_apps"].items():
        if app_name in text_lower and "open" in text_lower:
            return ("desktop_app", {
                "app": app_config["name"],
                "launch_cmd": app_config.get("launch_cmd", app_name)
            })

    # 6. Desktop scope: "click X in <app>"
    app_scope_match = re.search(r'\s+(?:in|into|of)\s+([a-z\d_\-\s]+)$', text_lower, re.IGNORECASE)
    if app_scope_match:
        app_name = app_scope_match.group(1).strip()
        # Check if it's a known app
        for known_app in _CONFIG["known_apps"].keys():
            if known_app in app_name.lower():
                return ("desktop_scope", {
                    "text": text,
                    "app": app_name
                })

    # 7. Desktop commands without scope (assume focused app)
    if any(word in text_lower for word in ["click", "type", "read", "focus", "snapshot"]):
        return ("desktop_scope", {"text": text, "app": None})

    # 8. List apps / snapshot
    if "list apps" in text_lower or "show apps" in text_lower:
        return ("desktop_command", {"action": "list_apps"})
    if "snapshot" in text_lower:
        app_match = re.search(r'snapshot\s+(.+)', text_lower)
        app = app_match.group(1) if app_match else None
        return ("desktop_command", {"action": "snapshot", "app": app})

    # Fall back to intent classifier for complex cases
    return None


def add_known_website(name: str, url: str, browser: str = "chrome") -> bool:
    """Add a website to config.json."""
    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        config["known_websites"][name.lower()] = {
            "url": url,
            "browser": browser,
            "actions": {}
        }
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception:
        return False


def add_known_app(name: str, launch_cmd: str, display_name: str = None) -> bool:
    """Add an app to config.json."""
    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        config["known_apps"][name.lower()] = {
            "name": display_name or name,
            "launch_cmd": launch_cmd
        }
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception:
        return False
