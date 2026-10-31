#!/usr/bin/env python3
"""
Vision Agent - Interactive Automation CLI

A conversational command-line assistant for local desktop and browser automation.
Uses the structured-data-first architecture:
1. Playwright / CDP for high-speed, position-independent browser automation.
2. AT-SPI for desktop UI trees and coordinates.
3. OCR + YOLO vision pipeline as on-demand fallback.

Usage:
  ./agent.py                                Interactive prompt (real execution)
  ./agent.py "open google.com"              Run single command and exit
  ./agent.py --dry-run                      Interactive prompt in simulation mode
  ./agent.py --headless                     Force headless browser mode
  ./agent.py --headed                       Force headed browser window
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intent_classifier import classify_intent
import context_memory
import playwright_bridge
import action_executor

# Try importing fallback vision modules lazily if needed
try:
    import atspi_bridge
    HAS_ATSPI = atspi_bridge.is_at_spi_available()
except Exception:
    HAS_ATSPI = False


def _print_banner(execute: bool, headless: bool):
    display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or "None (Headless)"
    print("\n" + "=" * 68)
    print("  Vision Agent CLI (v0.3 - Structured-First Architecture)")
    print("=" * 68)
    print(" • Browser Engine:   Playwright Chromium (fast, position-independent)")
    print(f" • Desktop Engine:   AT-SPI ({'Available' if HAS_ATSPI else 'Inactive/Fallback'}) + xdotool")
    print(f" • Display Server:   {display}")
    print(f" • Execution Mode:   {'LIVE (Actions executed)' if execute else 'DRY RUN (Simulation)'}")
    print("=" * 68)
    print("Commands:")
    print("  Just type what to do:")
    print("    • 'open google.com'")
    print("    • 'type formula 1 standings in search box and press enter'")
    print("    • 'click Images'")
    print("    • 'read page' or 'read table'")
    print("    • 'scroll down'")
    print("  Controls: /browser [info|close|read], /context, /execute on|off, /exit\n")


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
            print("   I didn't recognise that command. Try 'open <url>', 'type <text> in <box>', 'click <button>', or 'read page'.")
        return False

    steps = intent["params"]["steps"] if intent["action"] == "MULTISTEP" else [intent]

    for idx, step in enumerate(steps, 1):
        if len(steps) > 1:
            step_target = f" '{step['target']}'" if step.get("target") else ""
            print(f"\n[Step {idx}/{len(steps)}] {step['action']}{step_target}")

        ok = _execute_single_step(step, execute)
        if not ok:
            print(f"   [Step {idx} stopped]")
            return False

    return True


def _execute_single_step(step: Dict, execute: bool) -> bool:
    action = step["action"]
    target = step.get("target")
    params = step.get("params") or {}

    if action == "WAIT":
        secs = params.get("seconds", 1.0)
        if execute:
            action_executor.wait(secs)
        print(f"   {'Waited' if execute else 'Would wait'} {secs:g}s")
        return True

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
        if not playwright_bridge.is_browser_open():
            print("   'read' requires an open web page.")
            return False
        kind = params.get("kind", "page")
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
        if playwright_bridge.is_browser_open():
            return playwright_bridge.scroll(direction, amount)
        return action_executor.scroll(direction, amount)

    if action == "TYPE":
        # Key press like "press enter"
        if params.get("key"):
            key = params["key"]
            if not execute:
                print(f"   Would press key '{key}'")
                return True
            if playwright_bridge.is_browser_open():
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

        # Try Playwright first if browser is active
        if playwright_bridge.is_browser_open():
            ok = playwright_bridge.type_text(typed_text, target=target)
            if ok:
                return True
            print(f"   Field '{target}' not found in web page; trying desktop...")

        # Desktop fallback
        return action_executor.type_text(typed_text)

    if action == "CLICK":
        if not target:
            print("   No target specified to click.")
            return False

        if not execute:
            print(f"   Would click '{target}'")
            return True

        # Try Playwright first if browser is active
        if playwright_bridge.is_browser_open():
            ok = playwright_bridge.click_element(target, kind=params.get("kind"))
            if ok:
                return True
            print(f"   Element '{target}' not found in web page; trying desktop...")

        # Desktop fallback via AT-SPI or xdotool
        if HAS_ATSPI:
            elem = atspi_bridge.find_element_by_name(target)
            if elem and elem.get("ref"):
                if atspi_bridge.click_element(elem["ref"]):
                    print(f"   Clicked '{target}' via AT-SPI desktop action.")
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
                prompt = f"Vision Agent [{title}] > "
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
