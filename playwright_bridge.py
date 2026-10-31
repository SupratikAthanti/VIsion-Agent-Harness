"""
Playwright Browser Bridge for the Vision Agent Harness.

Provides direct browser engine control via Playwright (Chrome DevTools Protocol).
Replaces screenshot-based OCR/coordinate guessing for browser tasks with
structured, position-independent DOM interactions:
- navigate(url)
- click_element(target, kind=None)
- type_text(text, target=None, press_enter=False)
- press_key(key)
- scroll(direction, amount)
- read_content(selector=None)
- extract_table(selector=None)
- screenshot(path)

Supports headed and headless modes, auto-recovering sessions, smart locators
(role, accessible name, label, placeholder, text, CSS), and seamless
integration with context_memory and the intent classifier.
"""

import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from playwright.sync_api import (
        Browser,
        BrowserContext,
        ElementHandle,
        Locator,
        Page,
        Playwright,
        sync_playwright,
    )
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

import context_memory

DEFAULT_TIMEOUT_MS = 6000
NAVIGATE_TIMEOUT_MS = 30000
DEFAULT_SETTLE_MS = 300


def _warn(message: str) -> None:
    print(f"   [playwright] {message}")


def _info(message: str) -> None:
    print(f"   [playwright] {message}")


def _normalize_url(raw_url: str) -> str:
    """Normalize user-entered target into a valid URL."""
    url = raw_url.strip().strip("\"'").rstrip("?.!,")
    if not url:
        return ""
    if url.startswith(("data:", "file:", "about:", "chrome:", "javascript:")):
        return url
    if "://" not in url:
        # If it looks like a domain or localhost
        if url.startswith("localhost") or "." in url:
            url = "https://" + url
        else:
            # e.g. "google" -> "https://www.google.com"
            url = f"https://www.{url}.com"
    return url


_KEY_MAP = {
    "return": "Enter",
    "enter": "Enter",
    "tab": "Tab",
    "escape": "Escape",
    "esc": "Escape",
    "space": " ",
    "backspace": "Backspace",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "home": "Home",
    "end": "End",
}


class PlaywrightBridge:
    """
    Manages a persistent Playwright Chromium browser session and translates
    intent classifier actions into structured DOM actions.
    """

    def __init__(self, headless: Optional[bool] = None, slow_mo: int = 0):
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self.slow_mo = slow_mo

        # Headless configuration:
        # If headless explicitly set, use that.
        # Otherwise: check env var HEADLESS. If not set, check DISPLAY.
        # If DISPLAY is available, default to headed (headless=False) so user sees actions,
        # otherwise headless=True.
        if headless is not None:
            self._headless = headless
        elif os.environ.get("HEADLESS") is not None:
            self._headless = os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes")
        else:
            has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
            self._headless = not has_display

    @property
    def is_active(self) -> bool:
        """Return True if browser and page are currently open and responsive."""
        if self._page is None or self._browser is None:
            return False
        try:
            return not self._page.is_closed() and self._browser.is_connected()
        except Exception:
            return False

    @property
    def current_url(self) -> Optional[str]:
        if self.is_active and self._page:
            try:
                return self._page.url
            except Exception:
                return None
        return None

    @property
    def current_title(self) -> Optional[str]:
        if self.is_active and self._page:
            try:
                return self._page.title()
            except Exception:
                return None
        return None

    def start(self, headless: Optional[bool] = None) -> bool:
        """Launch Playwright browser if not already running."""
        if not HAS_PLAYWRIGHT:
            _warn("Playwright is not installed. Run: pip install playwright && playwright install chromium")
            return False

        if self.is_active:
            return True

        if headless is not None:
            self._headless = headless

        try:
            self._playwright = sync_playwright().start()

            # Launch options:
            launch_args = [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ]

            try:
                self._browser = self._playwright.chromium.launch(
                    headless=self._headless,
                    slow_mo=self.slow_mo,
                    args=launch_args,
                )
            except Exception as launch_err:
                # If headed failed (e.g. DISPLAY refused connection), try headless fallback
                if not self._headless:
                    _warn(f"Headed launch failed ({launch_err}); falling back to headless browser.")
                    self._headless = True
                    self._browser = self._playwright.chromium.launch(
                        headless=True,
                        slow_mo=self.slow_mo,
                        args=launch_args,
                    )
                else:
                    raise launch_err

            self._context = self._browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            )
            self._page = self._context.new_page()
            self._page.set_default_timeout(DEFAULT_TIMEOUT_MS)
            mode_str = "headless" if self._headless else "headed"
            _info(f"Playwright Chromium started ({mode_str} mode)")
            return True
        except Exception as e:
            _warn(f"Failed to start Playwright browser: {e}")
            self.close()
            return False

    def navigate(self, url: str) -> bool:
        """Navigate to a URL and update context memory."""
        clean_url = _normalize_url(url)
        if not clean_url:
            _warn("Invalid or empty URL")
            return False

        if not self.ensure_active():
            return False

        assert self._page is not None
        try:
            _info(f"Navigating to {clean_url}...")
            response = self._page.goto(clean_url, wait_until="domcontentloaded", timeout=NAVIGATE_TIMEOUT_MS)
            # Give dynamic SPAs a brief moment to render
            self._page.wait_for_timeout(DEFAULT_SETTLE_MS)

            title = self._page.title()
            final_url = self._page.url
            status = response.status if response else "OK"
            _info(f"Loaded: '{title}' [{final_url}] (HTTP {status})")

            # Update context memory
            context_memory.update_context({"action": "NAVIGATE", "params": {"url": final_url}})
            context_memory.set_current_page(f"{title} ({final_url})")
            return True
        except Exception as e:
            _warn(f"Navigation failed for {clean_url}: {e}")
            return False

    def ensure_active(self) -> bool:
        """Ensure browser and page are open, starting them if needed."""
        if not self.is_active:
            return self.start()
        return True

    def find_element_locator(self, target: str, kind: Optional[str] = None) -> Optional[Locator]:
        """
        Smart element location strategy.
        Searches DOM by role, accessible text, label, placeholder, title, and selectors.
        """
        if not self.is_active or not self._page:
            return None

        clean = target.strip()
        if not clean:
            return None

        page = self._page

        # 1. If target looks like a CSS selector or XPath
        if (clean.startswith(("#", ".", "//", "xpath=")) or
                any(clean.startswith(prefix) for prefix in ("input[", "button[", "a[", "div["))):
            try:
                loc = page.locator(clean)
                if loc.count() > 0 and loc.first.is_visible():
                    return loc.first
            except Exception:
                pass

        escaped = re.escape(clean)
        regex = re.compile(escaped, re.IGNORECASE)

        # 2. Kind-specific searches
        kind_lower = (kind or "").lower()

        if kind_lower in ("button", "btn"):
            try:
                loc = page.get_by_role("button", name=regex)
                if loc.count() > 0 and loc.first.is_visible():
                    return loc.first
            except Exception:
                pass

        if kind_lower in ("link", "tab"):
            try:
                role = "tab" if kind_lower == "tab" else "link"
                loc = page.get_by_role(role, name=regex)
                if loc.count() > 0 and loc.first.is_visible():
                    return loc.first
            except Exception:
                pass

        if kind_lower in ("input", "field", "box", "search", "textbox"):
            for role in ("searchbox", "textbox", "combobox"):
                try:
                    loc = page.get_by_role(role, name=regex)
                    if loc.count() > 0 and loc.first.is_visible():
                        return loc.first
                except Exception:
                    pass

        # 3. Accessible Name / Role search (buttons, links, headings, inputs)
        for role in ("button", "link", "searchbox", "textbox", "menuitem", "tab", "checkbox", "radio"):
            try:
                loc = page.get_by_role(role, name=regex)
                if loc.count() > 0 and loc.first.is_visible():
                    return loc.first
            except Exception:
                pass

        # 4. Form inputs by label or placeholder
        try:
            loc = page.get_by_label(regex)
            if loc.count() > 0 and loc.first.is_visible():
                return loc.first
        except Exception:
            pass

        try:
            loc = page.get_by_placeholder(regex)
            if loc.count() > 0 and loc.first.is_visible():
                return loc.first
        except Exception:
            pass

        # 5. Exact or case-insensitive text match anywhere on page
        try:
            loc = page.get_by_text(regex, exact=False)
            if loc.count() > 0:
                # Find first visible one
                for i in range(min(loc.count(), 5)):
                    item = loc.nth(i)
                    if item.is_visible():
                        return item
        except Exception:
            pass

        # 6. Combined CSS locator matching common interactive tags containing text or attribute
        css_selectors = [
            f"button:has-text('{clean}')",
            f"a:has-text('{clean}')",
            f"[role='button']:has-text('{clean}')",
            f"[role='link']:has-text('{clean}')",
            f"[role='tab']:has-text('{clean}')",
            f"input[placeholder*='{clean}' i]",
            f"input[name*='{clean}' i]",
            f"input[aria-label*='{clean}' i]",
            f"textarea[placeholder*='{clean}' i]",
            f"textarea[aria-label*='{clean}' i]",
            f"[aria-label*='{clean}' i]",
            f"[title*='{clean}' i]",
            f"*:has-text('{clean}')",
        ]

        for sel in css_selectors:
            try:
                loc = page.locator(sel)
                if loc.count() > 0 and loc.first.is_visible():
                    return loc.first
            except Exception:
                continue

        # 7. Fallback for search queries: if target is "search" or "search box"
        if any(term in clean.lower() for term in ("search", "search box", "search bar", "google search")):
            fallback_search_selectors = [
                "input[type='search']",
                "input[name='q']",
                "textarea[name='q']",
                "input[title='Search']",
                "[aria-label='Search']",
                "input[type='text']",
            ]
            for sel in fallback_search_selectors:
                try:
                    loc = page.locator(sel)
                    if loc.count() > 0 and loc.first.is_visible():
                        return loc.first
                except Exception:
                    continue

        return None

    def click(self, target: str, kind: Optional[str] = None, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> bool:
        """Find an element on the current page and click it."""
        if not self.ensure_active():
            return False

        assert self._page is not None
        locator = self.find_element_locator(target, kind=kind)
        if locator is None:
            _warn(f"Could not find element matching '{target}' (kind={kind}) on page")
            return False

        try:
            locator.scroll_into_view_if_needed(timeout=timeout_ms)
            locator.click(timeout=timeout_ms)
            self._page.wait_for_timeout(DEFAULT_SETTLE_MS)
            _info(f"Clicked '{target}' successfully")
            context_memory.update_context({"action": "CLICK", "target": target, "params": {"kind": kind}})
            return True
        except Exception as e:
            _warn(f"Click failed on '{target}': {e}")
            return False

    def type_text(
        self,
        text: str,
        target: Optional[str] = None,
        press_enter: bool = False,
        clear_first: bool = True,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> bool:
        """
        Type text into a target field, or into the currently focused element if target is None.
        Optionally presses Enter after typing.
        """
        if not self.ensure_active():
            return False

        assert self._page is not None
        page = self._page

        if target:
            locator = self.find_element_locator(target, kind="input")
            if locator is None:
                _warn(f"Could not find input field matching '{target}'")
                return False

            try:
                locator.scroll_into_view_if_needed(timeout=timeout_ms)
                locator.click(timeout=timeout_ms)
                if clear_first:
                    try:
                        locator.fill(text, timeout=timeout_ms)
                    except Exception:
                        page.keyboard.type(text, delay=20)
                else:
                    page.keyboard.type(text, delay=20)
                _info(f"Typed {text!r} into '{target}'")
            except Exception as e:
                _warn(f"Failed typing into '{target}': {e}")
                return False
        else:
            # Type into current focus
            try:
                page.keyboard.type(text, delay=20)
                _info(f"Typed {text!r} into active element")
            except Exception as e:
                _warn(f"Failed typing text: {e}")
                return False

        if press_enter:
            try:
                page.keyboard.press("Enter")
                page.wait_for_timeout(DEFAULT_SETTLE_MS)
                _info("Pressed Enter")
            except Exception as e:
                _warn(f"Failed pressing Enter: {e}")

        context_memory.update_context({"action": "TYPE", "target": target, "params": {"text": text}})
        return True

    def press_key(self, key_name: str) -> bool:
        """Press a keyboard key."""
        if not self.ensure_active():
            return False

        assert self._page is not None
        mapped = _KEY_MAP.get(key_name.lower().strip(), key_name)
        try:
            self._page.keyboard.press(mapped)
            self._page.wait_for_timeout(DEFAULT_SETTLE_MS)
            _info(f"Pressed key '{mapped}'")
            return True
        except Exception as e:
            _warn(f"Failed pressing key '{key_name}': {e}")
            return False

    def scroll(self, direction: str = "down", amount: int = 1) -> bool:
        """Scroll the page in the specified direction."""
        if not self.ensure_active():
            return False

        assert self._page is not None
        dir_clean = direction.lower()
        delta_y = 0
        delta_x = 0
        step = 300 * max(1, amount)

        if dir_clean == "down":
            delta_y = step
        elif dir_clean == "up":
            delta_y = -step
        elif dir_clean == "right":
            delta_x = step
        elif dir_clean == "left":
            delta_x = -step
        else:
            _warn(f"Unknown scroll direction: {direction}")
            return False

        try:
            self._page.mouse.wheel(delta_x, delta_y)
            self._page.wait_for_timeout(DEFAULT_SETTLE_MS)
            _info(f"Scrolled {dir_clean} ({amount} steps)")
            return True
        except Exception as e:
            _warn(f"Scroll failed: {e}")
            return False

    def read_content(self, selector: Optional[str] = None, max_chars: int = 2000) -> str:
        """Read text content from the active page or a specific selector."""
        if not self.is_active or not self._page:
            return ""

        try:
            if selector:
                loc = self._page.locator(selector).first
                text = loc.inner_text(timeout=DEFAULT_TIMEOUT_MS)
            else:
                text = self._page.inner_text("body", timeout=DEFAULT_TIMEOUT_MS)

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            cleaned = "\n".join(lines)
            if len(cleaned) > max_chars:
                cleaned = cleaned[:max_chars] + f"\n... [truncated {len(cleaned) - max_chars} chars]"
            return cleaned
        except Exception as e:
            _warn(f"Failed reading content: {e}")
            return ""

    def read_table(self, selector: Optional[str] = "table", max_rows: int = 20) -> List[List[str]]:
        """Extract table rows as structured text."""
        if not self.is_active or not self._page:
            return []

        try:
            table_loc = self._page.locator(selector or "table").first
            if table_loc.count() == 0:
                return []
            rows = table_loc.locator("tr")
            count = min(rows.count(), max_rows)
            result = []
            for i in range(count):
                cells = rows.nth(i).locator("th, td")
                row_vals = [cells.nth(j).inner_text().strip() for j in range(cells.count())]
                if any(row_vals):
                    result.append(row_vals)
            return result
        except Exception as e:
            _warn(f"Failed extracting table: {e}")
            return []

    def screenshot(self, path: Union[str, Path]) -> bool:
        """Capture a screenshot of the active web page."""
        if not self.is_active or not self._page:
            return False
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._page.screenshot(path=str(p))
            return p.exists()
        except Exception as e:
            _warn(f"Failed to capture screenshot: {e}")
            return False

    def close(self) -> None:
        """Close page, context, and browser."""
        try:
            if self._page:
                self._page.close()
        except Exception:
            pass
        self._page = None

        try:
            if self._context:
                self._context.close()
        except Exception:
            pass
        self._context = None

        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        self._browser = None

        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._playwright = None
        _info("Browser session closed")


# Module-level singleton
_default_bridge: Optional[PlaywrightBridge] = None


def get_bridge() -> PlaywrightBridge:
    """Get or create the singleton PlaywrightBridge instance."""
    global _default_bridge
    if _default_bridge is None:
        _default_bridge = PlaywrightBridge()
    return _default_bridge


def is_browser_open() -> bool:
    """Check if the singleton browser bridge is active."""
    return _default_bridge is not None and _default_bridge.is_active


def navigate(url: str, headless: Optional[bool] = None) -> bool:
    """Navigate to a URL using the Playwright bridge."""
    bridge = get_bridge()
    if headless is not None:
        bridge._headless = headless
    return bridge.navigate(url)


def click_element(target: str, kind: Optional[str] = None) -> bool:
    """Click an element on the active web page."""
    return get_bridge().click(target, kind=kind)


def type_text(text: str, target: Optional[str] = None, press_enter: bool = False) -> bool:
    """Type text into a target or active field."""
    return get_bridge().type_text(text, target=target, press_enter=press_enter)


def press_key(key: str) -> bool:
    """Press a key in the browser."""
    return get_bridge().press_key(key)


def scroll(direction: str = "down", amount: int = 1) -> bool:
    """Scroll the browser page."""
    return get_bridge().scroll(direction=direction, amount=amount)


def read_content(selector: Optional[str] = None) -> str:
    """Read text content from the browser page."""
    return get_bridge().read_content(selector=selector)


def read_table(selector: Optional[str] = "table") -> List[List[str]]:
    """Extract table rows from the browser page."""
    return get_bridge().read_table(selector=selector)


def close_browser() -> None:
    """Close the active browser session."""
    global _default_bridge
    if _default_bridge is not None:
        _default_bridge.close()
        _default_bridge = None
