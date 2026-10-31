#!/usr/bin/env python3
"""
Vision Agent - Interactive Automation CLI

A conversational command-line assistant for local desktop and browser automation.
Uses the structured-data-first architecture:
1. Direct AT-SPI accessibility tree inspection and actions for desktop applications.
2. Playwright / CDP for high-speed, position-independent browser automation.
3. OCR + YOLO vision pipeline as on-demand fallback.

Usage:
  ./agent.py                                Interactive prompt (real execution)
  ./agent.py "open google.com"              Run single command and exit
  ./agent.py --dry-run                      Interactive prompt in simulation mode
  ./agent.py --headless                     Force headless browser mode
  ./agent.py --headed                       Force headed browser window
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
from intent_classifier import classify_intent, route_context, is_ref
import context_memory
import playwright_bridge
import action_executor
import atspi_bridge

# Explicit security & safety enforcement policy
SENSITIVE_ACTIONS_PATTERN = re.compile(
    r"\b(delete|remove|close|quit|kill|terminate|shutdown|reboot|restart|poweroff|format|reset|wipe|destroy|uninstall|rmdir|unlink)\b",
    re.IGNORECASE,
)

# System critical apps protected from unintended automated interference
PROTECTED_SYSTEM_APPS = {
    "login", "lockscreen", "gnome-screensaver", "systemd", "polkit", "sudo", "auth"
}


def is_sensitive_action(step: Dict) -> bool:
    """Return True if the instruction targets a potentially destructive or protected action."""
    target = (step.get("target") or "").lower()
    app = ((step.get("params") or {}).get("app") or "").lower()
    text = ((step.get("params") or {}).get("text") or "").lower()
    combined = f"{target} {app} {text}"

    if any(p in combined for p in PROTECTED_SYSTEM_APPS):
        return True
    return bool(SENSITIVE_ACTIONS_PATTERN.search(combined))


def _print_banner(execute: bool, headless: bool):
    display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or "None (Headless)"
    atspi_status = "Available" if atspi_bridge.is_at_spi_available() else "Inactive/Fallback"
    print("\n" + "=" * 68)
    print("  Vision Agent CLI (v0.4 - Structured-First Architecture)")
    print("=" * 68)
    print(f" • Desktop Engine:   AT-SPI ({atspi_status}) + xdotool")
    print(" • Browser Engine:   Playwright Chromium (fast, position-independent)")
    print(f" • Display Server:   {display}")
    print(f" • Execution Mode:   {'LIVE (Actions executed)' if execute else 'DRY RUN (Simulation)'}")
    print("=" * 68)
    print("Commands:")
    print("  Desktop:")
    print("    • 'list apps'")
    print("    • 'snapshot Text Editor' or 'snapshot'")
    print("    • 'click @snapshot_id:e1' or 'click Save in Text Editor'")
    print("    • 'type hello in Notes' or 'read title bar of Text Editor'")
    print("  Browser:")
    print("    • 'open google.com'")
    print("    • 'type formula 1 standings in search box and press enter'")
    print("    • 'read page' or 'read table'")
    print("  Controls: /desktop [apps|snapshot|click|type|read|focus], /browser, /context, /exit\n")


def execute_instruction(instruction: str, execute: bool = True) -> bool:
    """Classify and run an instruction (single or multi-step)."""
    text = instruction.strip()
    if not text:
        return True

    intent = classify_intent(text)
    if intent["action"] == "UNKNOWN":
        if intent.get("needs_clarification"):
            print(f"   {intent['needs_clarification']}")
        else:
            print("   I didn't recognise that command. Try 'open <url>', 'type <text> in <box>', 'click <button>', or 'snapshot'.")
        return False

    steps = intent["params"]["steps"] if intent["action"] == "MULTISTEP" else [intent]

    for idx, step in enumerate(steps, 1):
        if len(steps) > 1:
            step_target = f" '{step['target']}'" if step.get("target") else ""
            print(f"\n[Step {idx}/{len(steps)}] {step['action']}{step_target}")

        # Safety enforcement gate: interactive confirmation or non-interactive fail-safe
        if execute and is_sensitive_action(step):
            target_desc = step.get('target') or 'this target'
            # Check if explicit allowlist environment flag is set
            allow_sensitive = os.environ.get("ALLOW_SENSITIVE_ACTIONS") == "1"
            if not allow_sensitive:
                if sys.stdin.isatty():
                    try:
                        conf = input(f"   [Security Gate] '{step['action']} {target_desc}' is classified as a sensitive/destructive action. Confirm execution? (y/N): ").strip().lower()
                        if conf not in ("y", "yes"):
                            print("   Action denied by user policy.")
                            return False
                    except (KeyboardInterrupt, EOFError):
                        print("\n   Action cancelled.")
                        return False
                else:
                    # Non-interactive / headless script mode fails closed by default to prevent unsupervised damage
                    print(f"   [Security Policy Enforced] Action '{step['action']} {target_desc}' blocked in automated/non-interactive mode without ALLOW_SENSITIVE_ACTIONS=1.")
                    return False

        ok = _execute_single_step(step, execute)
        if not ok:
            print(f"   [Step {idx} stopped]")
            return False

    return True


def _execute_single_step(step: Dict, execute: bool) -> bool:
    action = step["action"]
    target = step.get("target")
    params = step.get("params") or {}
    app = params.get("app")

    # Authoritative window and browser state detection
    active_window = action_executor.get_active_window_title()
    browser_active = playwright_bridge.is_browser_open()
    focused_context = active_window
    if browser_active and not active_window:
        focused_context = playwright_bridge.get_bridge().current_title

    ctx = {
        "browser_open": browser_active,
        "focused_app": focused_context,
    }
    route, clarification = route_context(step, ctx)

    if route == "ambiguous":
        print(f"   [Routing Ambiguity] {clarification}")
        return False

    if action == "WAIT":
        secs = params.get("seconds", 1.0)
        if execute:
            action_executor.wait(secs)
        print(f"   {'Waited' if execute else 'Would wait'} {secs:g}s")
        return True

    if action == "LIST_APPS":
        apps = atspi_bridge.list_apps()
        if not apps:
            print("   No accessible desktop applications detected (or accessibility bus inactive).")
            return False
        print(f"   --- Desktop Applications ({len(apps)}) ---")
        for a in apps:
            wins = f" - Windows: {', '.join(a['windows'])}" if a.get("windows") else ""
            print(f"   [{a['app_id']}] {a['name']}{wins}")
        return True

    if action == "SNAPSHOT":
        app_name = target or app
        snap = atspi_bridge.snapshot(app_name=app_name, interactive_only=True)
        if not snap.get("ok"):
            print(f"   Snapshot failed: {snap.get('error')}")
            return False
        elems = snap.get("elements", [])
        print(f"   --- Snapshot {snap['snapshot_id']} for '{snap['app']}' ({len(elems)} elements) ---")
        for e in elems[:30]:
            name_str = f" \"{e['name']}\"" if e.get("name") else ""
            bounds_str = f" {e['bounds']}" if e.get("bounds") else ""
            acts_str = f" [{', '.join(e['actions'])}]" if e.get("actions") else ""
            print(f"   {e['ref_id']:<15} {e['role']:<15}{name_str}{bounds_str}{acts_str}")
        if len(elems) > 30:
            print(f"   ... and {len(elems) - 30} more elements")
        return True

    if action == "FOCUS":
        if not target:
            print("   No focus target specified.")
            return False
        if not execute:
            print(f"   Would focus '{target}'")
            return True
        if atspi_bridge.is_at_spi_available():
            if atspi_bridge.grab_focus(target):
                print(f"   Focused '{target}' via AT-SPI.")
                return True
        print(f"   Could not focus '{target}'.")
        return False

    if action == "NAVIGATE":
        url = params.get("url") or target
        if not url:
            print("   No URL specified.")
            return False
        if not execute:
            print(f"   Would navigate to {url}")
            return True

        print(f"   Navigating to {url} via Playwright...")
        ok = playwright_bridge.navigate(url)
        if not ok:
            print("   Playwright navigation failed; trying system browser fallback...")
            ok = action_executor.navigate_to(url)
        return ok

    if action == "READ":
        kind = params.get("kind", "page")
        # 1. Desktop read
        if route == "desktop" or is_ref(target) or app:
            elem_data = atspi_bridge.read_element(target) if target else None
            if elem_data:
                print(f"   --- Accessible Element: {elem_data.get('name') or target} ---")
                for k, v in elem_data.items():
                    if v:
                        print(f"   {k}: {v}")
                return True
            # Read all elements of app or window
            snap = atspi_bridge.snapshot(app_name=app)
            if snap.get("ok") and snap.get("elements"):
                print(f"   --- Desktop Content for '{snap['app']}' ---")
                for e in snap["elements"][:20]:
                    t = e.get("text") or e.get("name")
                    if t:
                        print(f"   [{e['role']}] {t}")
                return True
            print(f"   Could not read desktop element '{target}'.")
            return False

        # 2. Browser read
        if not playwright_bridge.is_browser_open():
            print("   'read' requires an open web page or desktop target.")
            return False

        if kind == "table":
            rows = playwright_bridge.read_table(target)
            if rows:
                print(f"   --- Table Content ({len(rows)} rows) ---")
                for r in rows[:20]:
                    print("   " + " | ".join(r))
                return True
            print("   No table found on active page.")
            return False
        else:
            content = playwright_bridge.read_content(target)
            if content:
                print("   --- Page Content ---")
                for line in content.splitlines()[:30]:
                    print("   " + line)
                return True
            print("   No content could be read.")
            return False

    if action == "SCROLL":
        direction = params.get("direction", "down")
        amount = params.get("amount", 1)
        if not execute:
            print(f"   Would scroll {direction} by {amount}")
            return True
        if route == "browser" and playwright_bridge.is_browser_open():
            return playwright_bridge.scroll(direction, amount)
        return action_executor.scroll(direction, amount)

    if action == "TYPE":
        # Key press like "press enter"
        if params.get("key"):
            key = params["key"]
            if not execute:
                print(f"   Would press key '{key}'")
                return True
            if route == "browser" and playwright_bridge.is_browser_open():
                return playwright_bridge.press_key(key)
            return action_executor.press_key(key)

        typed_text = params.get("text", "")
        if not typed_text:
            print("   Nothing to type.")
            return False

        if not execute:
            dest = f"'{target}'" if target else "active focus"
            print(f"   Would type {typed_text!r} into {dest}")
            return True

        # Desktop route (AT-SPI primary)
        if route == "desktop" or is_ref(target) or app:
            try:
                elem = atspi_bridge.find_element_by_name(target, app_name=app, require_single=True) if (target and not is_ref(target)) else None
            except ValueError as e:
                print(f"   [Error] {e}. Specify the app or use a qualified ref.")
                return False

            ref_or_target = elem.get("ref_id") if elem else target
            if ref_or_target:
                # Strict verification: verify field is editable before modifying
                if not atspi_bridge.is_editable(ref_or_target):
                    print(f"   [Error] Target '{target}' is not an editable field. Aborting to prevent misdirected typing.")
                    return False

                # Direct AT-SPI EditableText
                if atspi_bridge.set_text_contents(ref_or_target, typed_text):
                    print(f"   Typed {typed_text!r} into '{target}' via AT-SPI editable text.")
                    return True
                # Verified editable but direct API failed: focus then xdotool typing
                atspi_bridge.grab_focus(ref_or_target)
                if action_executor.type_text(typed_text, accessible_ref=ref_or_target):
                    print(f"   Typed {typed_text!r} into '{target}' via focused input.")
                    return True
                return False

            print(f"   Could not find desktop field '{target}'.")
            return False

        # Browser route (Playwright primary)
        if playwright_bridge.is_browser_open():
            ok = playwright_bridge.type_text(typed_text, target=target)
            if ok:
                return True

        # If a specific target was requested, do not type blindly into active window
        if target:
            print(f"   [Error] Specific field '{target}' could not be resolved. Aborting typing.")
            return False

        # General active focus typing only if no target was specified
        return action_executor.type_text(typed_text)

    if action == "CLICK":
        if not target:
            print("   No target specified to click.")
            return False

        if not execute:
            print(f"   Would click '{target}' (routed to: {route})")
            return True

        # Desktop route (AT-SPI primary)
        if route == "desktop" or is_ref(target) or app:
            try:
                elem = atspi_bridge.find_element_by_name(target, app_name=app, require_single=True) if (target and not is_ref(target)) else None
            except ValueError as e:
                print(f"   [Error] {e}. Specify the app or use a qualified ref.")
                return False

            ref_or_target = elem.get("ref_id") if elem else target
            if ref_or_target:
                if atspi_bridge.click_element(ref_or_target):
                    print(f"   Clicked '{target}' via AT-SPI desktop action.")
                    return True
                # Try bounding box click via xdotool if action invocation failed
                bounds = atspi_bridge.get_element_bounds(ref_or_target)
                if bounds:
                    screen = action_executor.get_screen_size()
                    if screen:
                        x_pct = (bounds[0] + bounds[2] / 2) / screen[0]
                        y_pct = (bounds[1] + bounds[3] / 2) / screen[1]
                        if action_executor.click_at(x_pct, y_pct, use_atspi_action=False):
                            print(f"   Clicked '{target}' at live desktop coordinates.")
                            return True

        # Browser route (Playwright primary)
        if playwright_bridge.is_browser_open():
            ok = playwright_bridge.click_element(target, kind=params.get("kind"))
            if ok:
                return True

        # Desktop fallback if routed to browser but failed
        if route != "desktop" and atspi_bridge.is_at_spi_available():
            try:
                elem = atspi_bridge.find_element_by_name(target, app_name=app, require_single=True)
            except ValueError:
                elem = None
            if elem and atspi_bridge.click_element(elem.get("ref_id") or target):
                print(f"   Clicked '{target}' via AT-SPI fallback.")
                return True

        print(f"   Could not find clickable element '{target}'.")
        return False

    print(f"   Unsupported action: {action}")
    return False


def run_repl(execute: bool = True, headless: Optional[bool] = None):
    """Run interactive agent REPL."""
    if headless is not None:
        os.environ["HEADLESS"] = "1" if headless else "0"

    _print_banner(execute, bool(headless))

    try:
        while True:
            # Context-sensitive prompt
            if playwright_bridge.is_browser_open():
                title = playwright_bridge.get_bridge().current_title or "Browser"
                if len(title) > 28:
                    title = title[:25] + "..."
                prompt = f"Vision Agent [Browser: {title}] > "
            else:
                prompt = "Vision Agent > "

            try:
                user_input = input(prompt).strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting. Goodbye!")
                break

            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in ("exit", "quit", "/exit", "/quit"):
                print("Exiting. Goodbye!")
                break

            if cmd in ("/execute on", "/execute off"):
                execute = cmd.endswith("on")
                print(f"Execution mode: {'LIVE' if execute else 'DRY RUN'}")
                continue

            if cmd == "/context":
                ctx = context_memory.get_context()
                print(f"Current page: {ctx['current_page'] or '(none)'}")
                print(f"Recent actions: {len(ctx['last_actions'])}")
                for a in ctx["last_actions"][-5:]:
                    print(f"  - {a.get('action')} {a.get('target') or a.get('url') or ''}")
                continue

            if cmd.startswith("/desktop"):
                parts = user_input.split(maxsplit=2)
                sub = parts[1].lower() if len(parts) > 1 else "apps"
                arg = parts[2].strip() if len(parts) > 2 else None
                if sub == "apps":
                    _execute_single_step({"action": "LIST_APPS"}, execute)
                elif sub == "snapshot":
                    _execute_single_step({"action": "SNAPSHOT", "target": arg, "params": {"app": arg}}, execute)
                elif sub == "click" and arg:
                    _execute_single_step({"action": "CLICK", "target": arg, "params": {}}, execute)
                elif sub == "type" and arg:
                    # e.g. /desktop type @s123:e1 hello or /desktop type hello
                    type_parts = arg.split(maxsplit=1)
                    if len(type_parts) == 2 and is_ref(type_parts[0]):
                        _execute_single_step({"action": "TYPE", "target": type_parts[0], "params": {"text": type_parts[1]}}, execute)
                    else:
                        _execute_single_step({"action": "TYPE", "target": None, "params": {"text": arg}}, execute)
                elif sub == "read":
                    _execute_single_step({"action": "READ", "target": arg, "params": {"kind": "desktop"}}, execute)
                elif sub == "focus" and arg:
                    _execute_single_step({"action": "FOCUS", "target": arg, "params": {}}, execute)
                else:
                    print("Usage: /desktop [apps | snapshot [app] | click <ref|name> | type [ref] <text> | read [ref] | focus <ref>]")
                continue

            if cmd.startswith("/browser"):
                parts = user_input.split(maxsplit=2)
                sub = parts[1].lower() if len(parts) > 1 else "info"
                if sub in ("info", "status"):
                    if playwright_bridge.is_browser_open():
                        b = playwright_bridge.get_bridge()
                        print(f"Browser active: '{b.current_title}' [{b.current_url}]")
                    else:
                        print("Browser is not open.")
                elif sub == "close":
                    if playwright_bridge.is_browser_open():
                        playwright_bridge.close_browser()
                        print("Browser closed.")
                    else:
                        print("Browser is not open.")
                elif sub == "read":
                    sel = parts[2].strip() if len(parts) > 2 else None
                    if playwright_bridge.is_browser_open():
                        content = playwright_bridge.read_content(sel)
                        print(content if content else "(empty)")
                    else:
                        print("Browser is not open.")
                continue

            t0 = time.time()
            execute_instruction(user_input, execute=execute)
            print(f"   [completed in {time.time() - t0:.2f}s]\n")

    finally:
        playwright_bridge.close_browser()


def main():
    parser = argparse.ArgumentParser(
        description="Vision Agent CLI - Local Browser & Desktop Automation"
    )
    parser.add_argument("command", nargs="*", help="Direct command to execute (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate actions without executing")
    parser.add_argument("--headless", action="store_true", help="Force headless browser")
    parser.add_argument("--headed", action="store_true", help="Force visible browser window")
    args = parser.parse_args()

    headless = None
    if args.headless:
        headless = True
    elif args.headed:
        headless = False

    execute = not args.dry_run

    if args.command:
        # Run single instruction passed from terminal
        instruction = " ".join(args.command)
        if headless is not None:
            os.environ["HEADLESS"] = "1" if headless else "0"
        try:
            execute_instruction(instruction, execute=execute)
        finally:
            playwright_bridge.close_browser()
    else:
        # Interactive REPL
        run_repl(execute=execute, headless=headless)


if __name__ == "__main__":
    main()
