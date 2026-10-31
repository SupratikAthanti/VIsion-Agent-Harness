# Vision Agent Harness - Implementation Plan

## Project Vision
Build a voice-controlled computer automation system that can navigate any interface by clicking, typing, and interacting with UI elements. The system must run entirely locally on a Raspberry Pi 5 with 4GB RAM.

## Architectural Pivot (2026-10-29)

**We are changing our primary approach.** The original plan relied on a vision-first pipeline: take a screenshot, run OCR/YOLO to find elements, then click at pixel coordinates. While this works, it is slow (1.5-3.7s per action), fragile (breaks when windows move or resize), and resource-heavy (~2.6GB RAM).

**The new approach uses structured data instead of pixels:**
- **Browser tasks → Playwright / CDP:** Talks directly to the browser engine internally. No screenshots, no OCR, no coordinate guessing. Can click, type, fill forms, navigate — all position-independent. Uses ~50MB RAM.
- **Desktop app tasks → AT-SPI (Linux Accessibility API):** Queries each application's UI tree for elements by name and role, gets their live screen coordinates regardless of window position. Works for GTK apps, Qt apps, and Electron apps (VS Code, Slack, etc.). Already partially implemented in atspi_bridge.py.
- **Fallback → Existing vision pipeline (OCR + YOLO + templates):** Kept as a last resort for apps with poor accessibility support (games, custom-rendered UIs, old X11 apps). This code is NOT being deleted — it is still valuable for edge cases.

**Why the change:**
- Playwright and AT-SPI are position-independent — they don't care where a window is on screen
- They return structured data (element names, roles, states) instead of raw pixels, which is faster and more reliable than OCR
- They use a fraction of the RAM compared to the vision pipeline
- This is how most successful open-source desktop agents actually work (hybrid structured-data + vision fallback)

## Current Status (Legacy Vision Pipeline — Retained as Fallback)
- OCR-based text detection working (94% accuracy on text elements)
- TEV1 decision router operational
- YOLO icon detection operational (OmniParser ONNX model)
- Hybrid backend routing working (OCR for text, YOLO for icons)
- Memory usage: ~2.6GB (within constraints)
- Grid overlay system functional (20x20)
- Target marking system functional
- System can now detect both text labels and unlabeled icons
- Intent classifier (rule-based regex) operational
- Action executor (xdotool + cursor overlay) operational
- Template matching system working for static elements

**Note:** All of the above is being kept. It still works and serves as the fallback detection path when AT-SPI returns empty/incomplete trees or when automating apps that don't expose accessibility data. The vision pipeline is demoted from "primary" to "last resort."

---

## Implementation Roadmap

### Phase 1: Heuristic Rules (Completed - Failed)
Add pattern-based rules to handle common UI elements that OCR alone cannot solve.

**Status:** Implemented but unsuccessful - icons are not OCR'd as readable text characters.

**What was tried:**
- New tab detection (find + icon near "New Tab" text)
- Close tab detection (find X icon near tab labels)
- Menu button detection (hamburger icon patterns)

**Result:** Heuristics cannot work because OCR does not recognize graphical icons as text.

**Decision:** Skip to Phase 2 for proper icon detection.

### Phase 2: YOLO Icon Detection (Completed)
Integrate YOLO with ONNX Runtime for general icon and logo recognition.

**Research completed:**
- Identified UI detection models: ScreenParser (YOLO11-L), GPA-GUI-Detector, OmniParser icon_detect
- OmniParser icon_detect available in ONNX format directly (no PyTorch/NCNN conversion needed)
- Selected OmniParser icon_detect (12MB ONNX model) for CPU efficiency

**Implementation completed:**
- Downloaded OmniParser icon_detect ONNX model from Hugging Face
- Integrated ONNX Runtime for CPU inference (no PyTorch dependency)
- Created yolo_detector.py with:
  - Preprocessing (resize, padding, normalization)
  - Postprocessing (NMS, coordinate conversion)
  - Position-based icon detection (top_left, top_right, top_bar, small)
  - Text-adjacent icon detection (for new tab + icon near "New Tab" text)
- Integrated YOLO into hybrid backend routing in vision_cli.py
- TEV1 router now routes icon queries to YOLO, text queries to OCR
- Implemented heuristic for "new tab icon": OCR finds "New Tab" text, YOLO finds nearby icon

**Testing results:**
- "Where is the new tab icon": Routes to YOLO (61.9% confidence), finds icon near "New Tab" (38% confidence)
- "Where is the close icon": Routes to YOLO (89.5% confidence), finds icon (78% confidence)
- "Where is the menu icon": Routes to YOLO (80.9% confidence), finds icon (45% confidence)
- "Where is the Ask Gemini button": Routes to OCR (79.1% confidence), finds text (94% confidence)
- Response time: 1.5-3.7 seconds (includes TEV1 routing + YOLO/OCR inference)
- Memory impact: +12MB (ONNX model), minimal additional runtime memory

**Current status:**
- YOLO icon detection is functional and integrated
- Icon detection works for UI elements without text labels
- System can now distinguish between text labels and icons
- Hybrid backend successfully routes between OCR and YOLO based on query type
- **Demoted to fallback role** — only used when AT-SPI/Playwright are unavailable for a given app

---

### Phase 3: Playwright Browser Bridge (COMPLETED)
Replaced screenshot-based browser interaction with direct browser engine control via Playwright / Chrome DevTools Protocol.

**Why this replaces the old approach for browser tasks:**
- Playwright talks to Chrome internally — it doesn't see or care about the screen
- It can click any element, type in any text box, fill any form, navigate any page
- Position-independent: works regardless of where Chrome is on screen, even if minimized
- Uses ~50MB RAM instead of ~2.6GB for the vision pipeline
- Element lookup is milliseconds, not 1.5-3.7 seconds

**Implementation completed:**
- Installed Playwright and Chromium build for Linux ARM64 (`playwright-1.64.1`)
- Built `playwright_bridge.py` supporting:
  - `navigate(url)` with URL normalization
  - `click_element(target, kind)` with smart role/label/text/CSS locator matching
  - `type_text(text, target, press_enter)` with clear-before-type and focus detection
  - `press_key(key)` with keyboard mapping
  - `scroll(direction, amount)` with mouse wheel deltas
  - `read_content(selector)` and `read_table(selector)` for content extraction
  - Headless/headed auto-detection based on DISPLAY
- Connected Playwright bridge into `intent_classifier.py` and `action_executor.py`
- Added `READ` action support and multi-step verb chaining ("type X in search and press enter")
- Integrated into `vision_cli.py` and created interactive executable `./vision` / `agent.py`
- Tested and verified with Google Search, Wikipedia F1 standings, and unit test suite

**Expected Outcome:**
- All browser automation handled through Playwright, not screenshots
- Memory impact: ~50MB (Playwright + CDP connection)
- Latency: milliseconds per action instead of seconds

**Limitations to be aware of:**
- CAPTCHAs and anti-bot detection on some sites
- Needs Chromium installed (Playwright bundles its own, but it takes disk space)
- Only works for browser content — desktop apps need a different path (Phase 4)

### Phase 4: AT-SPI Desktop Automation (COMPLETED)
Enable position-independent control of native desktop applications via the Linux Accessibility API.

**Why this is better than the old coordinate-based approach:**
- AT-SPI asks each app "where are your UI elements right now?" — live coordinates, not hardcoded positions
- Elements found by name, role, and text content ("search box", "submit button"), not by pixel location
- Works for GTK apps (file manager, terminal, settings), Qt apps (VLC, KDE apps), and Electron apps (VS Code, Slack, Discord)
- Modeled after the `agent-desktop` snapshot/ref architecture (`@snapshot_id:e<N>`)

**Implementation completed:**
- Installed AT-SPI system packages: `python3-pyatspi`, `python3-gi`, `gir1.2-atspi-2.0`, `at-spi2-core`, `xdotool`, `wmctrl`
- Completed `atspi_bridge.py` supporting:
  - App discovery (`list_apps()`) and window enumerations
  - Structured accessibility snapshotting (`snapshot(app_name)`, `snapshot_all()`)
  - Qualified reference addressing (`@<snapshot_id>:e<N>`) and resolution
  - Element searching by name, role, and text content
  - Direct headless action invocation via `Atspi.Action` (`click`, `press`, `activate`, etc.)
  - Direct semantic text setting via `Atspi.EditableText` (`set_text_contents`, `insert_text`) without focus stealing
  - Accessible element state and text reading via `read_element`
  - Focus management via `Atspi.Component` (`grab_focus`)
  - Clean fallbacks to live bounding box clicks and typing via `xdotool`
  - Deterministic mock backend support (`set_mock_desktop()`) allowing 100% test pass rate in headless/CI environments
- Updated `intent_classifier.py`:
  - Added app-scoping regex (`click File in VS Code`, `type hello in Notes`)
  - Added snapshot/ref actions (`LIST_APPS`, `SNAPSHOT`, `FOCUS`, `READ`)
  - Added `route_context()` to route deterministically between Browser (Playwright), Desktop (AT-SPI), and Vision fallback
- Updated `action_executor.py`:
  - Enhanced `type_text` to support direct AT-SPI EditableText setting before falling back to focus + typing
- Updated `agent.py`:
  - Added `/desktop [apps|snapshot|click|type|read|focus]` commands
  - Connected desktop routing so desktop tasks invoke AT-SPI direct actions
- Added comprehensive unit and integration test suite (`tests/test_atspi_desktop.py`) with 20 mock tests covering all AT-SPI capabilities

**Expected Outcome:**
- Position-independent desktop app control (window in any position/size)
- Memory impact: negligible (AT-SPI is a system service, pyatspi/gi is lightweight)
- Latency: milliseconds for element lookup + action invocation

### Phase 5: Voice Input Integration (NEW)
Add speech-to-text capability to drive the system via voice commands.

**Decision: Vosk (not Whisper)**

Research results:
- Vosk: ~200-400MB RAM, real-time streaming latency, processes word-by-word as you speak
- Whisper tiny: ~800MB-1.2GB RAM, 2-3 second delay after speaking stops
- Given our memory constraints, Vosk is the clear choice
- Vosk is less accurate with accents/noise, but for command-style input ("open google.com", "click submit") it is sufficient
- Can constrain Vosk's vocabulary/grammar to boost accuracy for common commands

**Tasks:**
- Install Vosk and download a small English model (~50MB)
- Integrate audio capture (microphone input)
- Connect Vosk streaming output to the intent classifier
- Add wake word detection (optional — could use a simple keyword like "hey vision" or just push-to-talk)
- Handle command boundaries (know when the user has finished speaking a command)

**Expected Outcome:**
- Voice-controlled automation: say "open google.com" and it happens
- Memory impact: ~300MB (Vosk model + audio pipeline)
- Latency: real-time (words recognized as spoken)

### Phase 6: Agent Loop / Multi-Step Tasks (NEW)
Enable multi-step task execution for complex commands like "go to the F1 website and check who's winning."

**Tasks:**
- Evaluate whether the existing rule-based intent classifier can handle multi-step decomposition (it already parses "then" / "and then" connectors)
- If rule-based isn't enough, integrate a small local LLM for task planning:
  - Candidate: Qwen2.5 1.5B (Q4 quantized, ~1.3GB RAM, ~10-18 tokens/sec on Pi 5)
  - Load on demand only for complex multi-step tasks, unload when done
  - The LLM plans the steps, the structured data paths (Playwright/AT-SPI) execute them
- Implement state tracking: what page are we on, what app is focused, what did we just do
- Add error recovery: if an action fails, retry or ask the user what to do
- Create task templates for common workflows (e.g., "check F1 standings" = navigate to formula1.com → click Standings → read table)

**Expected Outcome:**
- Complex voice commands that chain multiple actions
- Memory impact: 0 if rule-based is sufficient, ~1.3GB if LLM is needed (loaded on demand)

### Phase 7: Optimization and Refinement
Improve performance, accuracy, and reliability across the whole system.

**Tasks:**
- Determine if the legacy vision pipeline (OCR + YOLO + TEV1) can be unloaded from memory when not needed, freeing ~2GB
- Only load vision models on demand when AT-SPI/Playwright can't handle a specific app
- Cache AT-SPI tree queries for frequently-used apps
- Optimize Playwright page state caching
- Comprehensive end-to-end testing: voice → intent → action → verification
- Set up VNC for remote viewing of the Pi desktop during testing

**Expected Outcome:**
- Faster response times across all paths
- Lower idle memory usage (load models only when needed)
- More reliable multi-step task execution

---

## New Architecture (Structured Data First)

### Detection/Interaction Pipeline (in priority order)
1. **Intent Classifier** — rule-based regex parses voice/text command into action + target
2. **Router** — determines if target is a browser task or desktop app task
3. **Browser tasks → Playwright/CDP** — direct browser engine control, no screen interaction
4. **Desktop app tasks → AT-SPI** — query accessibility tree, get live coordinates, xdotool click/type
5. **Fallback → Vision Pipeline** — OCR + YOLO + templates for apps with poor/no accessibility support
6. **Last resort → ask user** — if nothing works, ask for clarification or suggest manual teaching

### Memory Budget (New Target)
- Vosk (voice input): ~300MB
- Playwright (browser bridge): ~50MB
- AT-SPI (desktop bridge): negligible
- Intent classifier (regex): negligible
- xdotool + cursor overlay: negligible
- **Primary path total: ~350-400MB**
- OS + desktop environment: ~1.5GB
- Chrome/browser: ~500MB-1GB
- **Comfortable total: ~2.5GB — well within 4GB**
- Legacy vision pipeline (if loaded on demand): +2.6GB — only loaded when needed as fallback
- Optional LLM for planning (if needed): +1.3GB — loaded on demand

### Technology Stack
- Voice input: **Vosk** (selected — low RAM, real-time streaming)
- Intent parsing: **Rule-based regex** (existing intent_classifier.py)
- Browser control: **Playwright / CDP** (new — replaces screenshot-based browser interaction)
- Desktop control: **AT-SPI + xdotool** (existing atspi_bridge.py — needs completion and testing)
- Cursor visualization: **Tkinter overlay** (existing cursor_overlay.py)
- Fallback detection: **Tesseract OCR + OmniParser YOLO + OpenCV templates** (existing — demoted to fallback)
- Task planning: **Rule-based first, small LLM (Qwen2.5 1.5B Q4) if needed** (Phase 6)

### Constraints
- Must run entirely local (no cloud APIs)
- Must work on Raspberry Pi 5 (4GB RAM)
- Must be CPU-only (no GPU acceleration)
- Latency should be under 5 seconds for simple actions (new target: under 1 second for Playwright/AT-SPI paths)

---

## Next Steps
1. Phase 1: Heuristic rules — COMPLETED (unsuccessful, icons not OCR'd as text)
2. Phase 2: YOLO Icon Detection — COMPLETED (demoted to fallback)
3. Phase 3: Playwright Browser Bridge — COMPLETED
   - Installed Playwright & Chromium ARM64
   - Built playwright_bridge.py
   - Integrated with intent classifier and action pipeline
   - Added interactive `./vision` CLI entrypoint
4. **Phase 4: AT-SPI Desktop Automation — NEXT**
   - Install AT-SPI packages (`python3-pyatspi`, `python3-gi`, `gir1.2-atspi-2.0`)
   - Test accessibility trees in a real desktop session
   - Complete atspi_bridge.py for full element finding + interaction
5. Phase 5: Voice Input (Vosk) — after Phase 4
6. Phase 6: Agent Loop / Multi-Step — after Phase 5
7. Phase 7: Optimization — ongoing
