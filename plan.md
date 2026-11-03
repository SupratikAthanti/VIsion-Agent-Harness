# Vision Agent Harness - Complete Plan & Development Log

## Project Vision
Build a voice-controlled computer automation system that can navigate any interface by clicking, typing, and interacting with UI elements. The system must run entirely locally on a Raspberry Pi 5 with 4GB RAM.

---

## Architectural Pivot (2026-10-29)

**We changed our primary approach from vision-first to structured-data-first.**

**Old approach (vision-first):**
- Take screenshot → run OCR/YOLO → find elements → click at pixel coordinates
- Problems: Slow (1.5-3.7s per action), fragile (breaks when windows move), resource-heavy (~2.6GB RAM)

**New approach (structured-data-first):**
- **Browser tasks → Playwright / CDP:** Talks directly to browser engine internally. No screenshots, no OCR, no coordinate guessing. Position-independent. Uses ~50MB RAM.
- **Desktop app tasks → AT-SPI (Linux Accessibility API):** Queries each application's UI tree for elements by name and role, gets live screen coordinates regardless of window position. Works for GTK apps, Qt apps, and Electron apps.
- **Fallback → Vision pipeline (OCR + YOLO + templates):** Kept as last resort for apps with poor accessibility support (games, custom-rendered UIs, old X11 apps).

**Why the change:**
- Playwright and AT-SPI are position-independent — they don't care where a window is on screen
- They return structured data (element names, roles, states) instead of raw pixels
- They use a fraction of the RAM compared to the vision pipeline
- This is how most successful open-source desktop agents actually work (agent-desktop, AutoControlGUI, OSWright)

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

---

### Phase 2: YOLO Icon Detection (Completed - Demoted to Fallback)
Integrate YOLO with ONNX Runtime for general icon and logo recognition.

**Research completed:**
- Identified UI detection models: ScreenParser (YOLO11-L), GPA-GUI-Detector, OmniParser icon_detect
- OmniParser icon_detect available in ONNX format directly (no PyTorch/NCNN conversion needed)
- Selected OmniParser icon_detect (12MB ONNX model) for CPU efficiency

**Implementation completed:**
- Downloaded OmniParser icon_detect ONNX model from Hugging Face
- Integrated ONNX Runtime for CPU inference (no PyTorch dependency)
- Created yolo_detector.py with preprocessing, postprocessing, position-based detection, text-adjacent detection
- Integrated YOLO into hybrid backend routing in vision_cli.py
- TEV1 router now routes icon queries to YOLO, text queries to OCR

**Testing results:**
- "Where is the new tab icon": Routes to YOLO (61.9% confidence), finds icon near "New Tab" (38% confidence)
- "Where is the close icon": Routes to YOLO (89.5% confidence), finds icon (78% confidence)
- "Where is the menu icon": Routes to YOLO (80.9% confidence), finds icon (45% confidence)
- "Where is the Ask Gemini button": Routes to OCR (79.1% confidence), finds text (94% confidence)
- Response time: 1.5-3.7 seconds
- Memory impact: +12MB (ONNX model)

**Current status:**
- YOLO icon detection is functional and integrated
- **Demoted to fallback role** — only used when AT-SPI/Playwright are unavailable

---

### Phase 3: Playwright Browser Bridge (Completed)
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

**Limitations:**
- CAPTCHAs and anti-bot detection on some sites
- Needs Chromium installed (Playwright bundles its own, but it takes disk space)
- Only works for browser content — desktop apps need AT-SPI

---

### Phase 4: AT-SPI Desktop Automation (Implemented in PR)
Enable position-independent control of native desktop applications via the Linux Accessibility API.

**Why this is better than the old coordinate-based approach:**
- AT-SPI asks each app "where are your UI elements right now?" — live coordinates, not hardcoded positions
- Elements found by name and role ("search box", "submit button"), not by pixel location
- Works for GTK apps (file manager, terminal, settings), Qt apps (VLC, KDE apps), and Electron apps (VS Code, Slack, Discord)
- Already partially implemented in atspi_bridge.py — enhanced with structured snapshots and direct actions

**PR Implementation (feat/phase-4-atspi-desktop-automation):**

**Key additions:**
1. **Enhanced `atspi_bridge.py`:**
   - App discovery (`list_apps`) and window listing
   - Structured snapshots (`snapshot`, `snapshot_all`) with qualified refs (`@snapshot_id:e<N>`) and resolution
   - Semantic element lookup by name, role, and text content
   - Direct headless action invocation (`click`, `press`, `activate`) via `Atspi.Action`
   - Direct headless text setting (`set_text_contents`, `insert_text`) via `Atspi.EditableText`
   - Semantic state reading (`read_element`) and focus control (`grab_focus`)
   - Mock desktop support (`set_mock_desktop`) for deterministic headless CI testing

2. **Enhanced `intent_classifier.py`:**
   - App-scoped targets (e.g. `click File in VS Code`, `type hello in Notes`)
   - Snapshot & inspection intents (`LIST_APPS`, `SNAPSHOT`, `FOCUS`, `READ`)
   - `route_context()` prioritizing explicit browser vs desktop targets and focused application contexts
   - Ambiguity detection when targets lack explicit context across concurrent apps

3. **Enhanced `action_executor.py`:**
   - `type_text` direct AT-SPI editable text support before falling back to focus + typing
   - Strict `is_editable()` validation before typing to eliminate misdirected input

4. **Enhanced `agent.py`:**
   - Interactive `/desktop` REPL controls (`apps`, `snapshot`, `click`, `type`, `read`, `focus`)
   - Natural language execution routing desktop targets to AT-SPI

5. **Tests:**
   - `tests/test_atspi_desktop.py` with 135 passing unit and mock integration tests
   - Covers all AT-SPI actions, snapshots, refs, and routing

6. **Security & Hardening:**
   - Interactive confirmation safety gates for destructive desktop actions
   - Fail-closed blocking in automated runs unless `ALLOW_SENSITIVE_ACTIONS=1` is explicitly supplied
   - AMBIGUOUS_TARGET detection when multiple candidates match a name query
   - Snapshot caching with 15-second TTL expiration and node mutation verification

**Tasks to complete in real environment:**
- Install AT-SPI system packages: `python3-pyatspi`, `python3-gi`, `gir1.2-atspi-2.0`
- Enable accessibility on the desktop: `gsettings set org.gnome.desktop.interface toolkit-accessibility true`
- For Electron apps (VS Code, etc.): launch with `--force-renderer-accessibility` flag
- Test with real apps: VS Code, terminal, file manager in a real desktop session

**Expected Outcome:**
- Position-independent desktop app control (window in any position/size)
- Memory impact: negligible (AT-SPI is a system service, pyatspi is lightweight)
- Latency: milliseconds for element lookup + direct action

**Known limitations:**
- Terminal windows expose content as one text blob — can focus and type into them but can't click specific lines
- Some apps have lazy/broken accessibility trees (games, custom-rendered UIs, old X11 apps)
- For those apps, fall back to the existing vision pipeline (OCR + YOLO + templates)
- The "which text box do you mean?" disambiguation problem — when multiple inputs are visible, the system needs context to pick the right one

---

### Phase 5: Voice Input Integration (Future)
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

---

### Phase 6: Agent Loop / Multi-Step Tasks (Future)
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

---

### Phase 7: Optimization and Refinement (Ongoing)
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

## Current System Architecture

### Detection/Interaction Pipeline (in priority order)
1. **Intent Classifier** — rule-based regex parses voice/text command into action + target
2. **Router** — determines if target is a browser task or desktop app task
3. **Browser tasks → Playwright/CDP** — direct browser engine control, no screen interaction
4. **Desktop app tasks → AT-SPI** — query accessibility tree, get live coordinates, direct action or xdotool click/type
5. **Fallback → Vision Pipeline** — OCR + YOLO + templates for apps with poor/no accessibility support
6. **Last resort → ask user** — if nothing works, ask for clarification or suggest manual teaching

### Memory Budget (New Target)
- Vosk (voice input): ~300MB (not yet implemented)
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
- Voice input: **Vosk** (selected — low RAM, real-time streaming) — NOT YET IMPLEMENTED
- Intent parsing: **Rule-based regex** (existing intent_classifier.py)
- Browser control: **Playwright / CDP** (new — replaces screenshot-based browser interaction)
- Desktop control: **AT-SPI + xdotool** (existing atspi_bridge.py — enhanced in PR)
- Cursor visualization: **Tkinter overlay** (existing cursor_overlay.py)
- Fallback detection: **Tesseract OCR + OmniParser YOLO + OpenCV templates** (existing — demoted to fallback)
- Task planning: **Rule-based first, small LLM (Qwen2.5 1.5B Q4) if needed** (Phase 6 — NOT YET IMPLEMENTED)

### Constraints
- Must run entirely local (no cloud APIs)
- Must work on Raspberry Pi 5 (4GB RAM)
- Must be CPU-only (no GPU acceleration)
- Latency should be under 5 seconds for simple actions (new target: under 1 second for Playwright/AT-SPI paths)

---

## What Was Attempted & Results

### ✅ SUCCESSFUL IMPLEMENTATIONS

#### 1. Template Matching System
- **What**: OpenCV template matching for UI elements
- **Status**: ✅ Working
- **Bug Fixed**: Template name matching now handles spaces/punctuation correctly
- **Limitation**: Only works for static, position-stable elements (logos, fixed buttons)
- **Lesson**: Template matching is great for icons but fails for flat/low-contrast areas (search bars) and dynamic elements (tabs that move)

#### 2. OCR Engine (Tesseract)
- **What**: Text extraction with coordinates
- **Status**: ✅ Working
- **Use Case**: Finding text labels, buttons with text
- **Performance**: Fast (~1s)
- **Lesson**: Reliable for text-based elements, but can't recognize icons without text

#### 3. Spatial Heuristics for Icons
- **What**: OCR-based icon detection (finds + for new tab, x for close, ≡ for menu)
- **Status**: ✅ Working
- **Implementation**: Finds reference text, then looks for nearby symbolic characters
- **Lesson**: Works better than pure YOLO for common icon patterns, but limited to known symbols

#### 4. YOLO Icon Detection
- **What**: ONNX model for icon detection
- **Status**: ⚠️ Partially working
- **Improvement**: Added confidence threshold (50%) and size filtering
- **Problem**: Position-based detection, not semantically aware. Returns same location for different queries.
- **Lesson**: YOLO is good fallback but not reliable as primary detection. Confidence ≠ correctness.

#### 5. AT-SPI Accessibility Bridge
- **What**: Linux accessibility API integration
- **Status**: ✅ Implemented but ⚠️ Untested in real desktop session
- **Enhancement**: Phase 4 PR adds structured snapshots, direct actions, mock desktop support
- **Problem**: Requires system packages (`python3-pyatspi`, `python3-gi`, `gir1.2-atspi-2.0`) and active desktop session
- **Potential**: Could provide instant, position-independent element finding
- **Lesson**: Needs actual GUI desktop session to test. Promising but requires installation.

#### 6. Cursor Overlay (Phase 1)
- **What**: Blue arrow cursor that shows where system will click
- **Status**: ✅ Implemented
- **Purpose**: Debugging - see where clicks are landing
- **Tech**: Tkinter + X11 click-through window
- **Lesson**: Essential for debugging automation issues.

#### 7. AT-SPI Focus Fix (Phase 2)
- **What**: Try AT-SPI Action first (background-friendly), fall back to wmctrl + xdotool
- **Status**: ✅ Implemented
- **Purpose**: Prevent window stealing when clicking across applications
- **Lesson**: Accessibility APIs can click without bringing windows to foreground.

#### 8. Position-Independent Template Strategy (Phase 3)
- **What**: Reorder pipeline: AT-SPI → OCR → static templates → YOLO
- **Status**: ✅ Implemented
- **Restriction**: Only train templates for truly static elements (logos, fixed buttons)
- **Blocked**: Dynamic elements (tabs, search bars, menus) must use OCR/AT-SPI
- **Lesson**: Position-dependent templates break when layout changes. Always have semantic fallbacks.

#### 9. Playwright Browser Bridge (Phase 3)
- **What**: Direct browser engine control via CDP
- **Status**: ✅ Implemented
- **Performance**: Milliseconds per action instead of 1.5-3.7s
- **Memory**: ~50MB instead of ~2.6GB
- **Lesson**: Structured data beats vision for browser automation.

---

### ❌ FAILED ATTEMPTS

#### 1. Moondream Vision Model
- **What**: 4-bit quantized Moondream via Ollama for semantic vision understanding
- **Attempted**: Queries like "Where is the plus icon?" "What icons do you see?"
- **Result**: ❌ Failed
  - Empty responses (3/5 queries returned almost nothing)
  - Hallucinations (said screen was black when it wasn't)
  - Wrong locations (placed icons in wrong positions)
  - Slow (7-16 seconds vs 1-2 second current pipeline)
- **Why**: Small VLMs are sensitive to domain gaps. Benchmark numbers (ScreenSpot 80.4%) don't translate to real-world screenshots.
- **Lesson**: Current small VLMs are not ready for reliable UI automation. Hybrid system is better.

#### 2. Search Box Template Training
- **What**: Trained template for search box
- **Result**: ❌ Failed
- **Why**: Search bars are flat/low-contrast areas. Template matching's MIN_STD check rejects them as "too flat"
- **Lesson**: Don't train templates for flat input fields. Use OCR or AT-SPI instead.

#### 3. Pure Coordinate-Based Templates
- **What**: Training templates at specific (x, y) coordinates
- **Result**: ❌ Failed for dynamic elements
- **Why**: Tabs move when new tabs open, windows move when resized, icons shift in different layouts
- **Lesson**: Coordinates are last resort. Always use semantic approaches first.

#### 4. Heuristic Rules for Icons (Phase 1)
- **What**: Pattern-based rules to find icons via OCR
- **Result**: ❌ Failed
- **Why**: OCR does not recognize graphical icons as text characters
- **Lesson**: OCR is for text, not icons. Use YOLO or AT-SPI for icons.

---

## DO NOT Try Again

1. **Moondream for UI automation** - Already tested, hallucinates and unreliable
2. **Gemma Vision on RPi 4GB** - Doesn't fit RAM (needs 5GB+)
3. **Training templates for dynamic elements** - Will break when layout changes
4. **Pure coordinate-based approaches** - Fragile and position-dependent
5. **OCR-based icon detection** - Icons are not recognized as text characters

---

## WORTH Exploring Later

1. **AT-SPI in real desktop session** - Install packages, test in GUI environment (Phase 4 PR ready)
2. **VNC/remote desktop** - View RPi from Chromebook to see cursor movements
3. **Better small VLMs** - Wait 1-2 years for models to improve (currently not ready)
4. **Modular approach (Nemo-style)** - Keep YOLO for detection, small LLM for reasoning (what we already have)
5. **Voice input (Vosk)** - After AT-SPI is tested and working (Phase 5)

---

## Current Focus

1. **Merge Phase 4 PR** - AT-SPI desktop automation with structured snapshots
2. **Test AT-SPI in real desktop session** - Install packages, test with VS Code, terminal, file manager
3. **Set up VNC** to view RPi desktop remotely
4. **Train more static templates** for frequently-used elements
5. **Improve OCR spatial heuristics** for more icon patterns

---

## Computer Use Readiness Assessment

### What's Working Now
✅ **Browser automation** - Playwright bridge fully functional
✅ **Natural language commands** - Intent classifier handles complex queries
✅ **Multi-step tasks** - "type X in search and press enter" works
✅ **Interactive CLI** - `./agent.py` provides REPL with context-aware prompts
✅ **Dry-run mode** - Safe simulation before real execution
✅ **Cursor overlay** - Visual feedback before clicking
✅ **AT-SPI bridge** - Structured snapshots and direct actions (needs real desktop testing)

### What's Missing for Full Computer Use
⚠️ **AT-SPI in real desktop session** - Code is ready, needs system packages and GUI testing
⚠️ **Voice input** - Not yet implemented (Phase 5)
⚠️ **VNC/remote viewing** - Not set up yet
⚠️ **Real-world testing** - Split-screen, multi-app scenarios not tested

### What an AI Agent Can Do Right Now
An AI agent can:
1. **Call the CLI** - `./agent.py "open google.com"` or `./agent.py --dry-run "click Images"`
2. **Use natural language** - "type formula 1 standings in search box and press enter"
3. **Browse the web** - Navigate, click, type, read, scroll via Playwright
4. **Automate desktop apps** - AT-SPI code is ready, needs real desktop session to test
5. **See cursor movements** - Blue overlay shows where clicks will land

### What's Needed for "Just Open Terminal and Type Commands"
1. **Install AT-SPI packages** in real desktop session:
   ```bash
   sudo apt install python3-pyatspi python3-gi gir1.2-atspi-2.0
   gsettings set org.gnome.desktop.interface toolkit-accessibility true
   ```
2. **Test AT-SPI** with real apps (VS Code, terminal, file manager)
3. **Set up VNC** to view the desktop remotely
4. **Enable voice input** (optional - Phase 5)

---

## Key Lessons

1. **Layered approaches beat single models**: AT-SPI → OCR → templates → YOLO is more reliable than any single vision model
2. **Position independence matters**: Templates break when layouts change, semantic approaches don't
3. **Benchmark numbers ≠ real performance**: ScreenSpot 80.4% doesn't mean it works on your screenshots
4. **Small VLMs aren't ready yet**: They hallucinate and struggle with domain gaps
5. **Cursor visualization is essential**: Can't debug automation without seeing where it clicks
6. **Accessibility APIs are powerful**: AT-SPI can click without window focus - use when available
7. **RAM constraints are real**: 4GB RPi limits model choices, work within it
8. **Your current system is state-of-the-art**: Matches what agent-desktop, AutoControlGUI, OSWright use locally
9. **Structured data beats vision**: Playwright/AT-SPI are faster, more reliable, use less RAM than screenshot-based approaches
10. **agent-desktop architecture is sound**: The PR correctly implements structured snapshots, qualified refs, and direct actions

---

## Commands Reference

### Interactive Agent CLI
```bash
./agent.py                                Interactive prompt (real execution)
./agent.py "open google.com"              Run single command and exit
./agent.py --dry-run                      Interactive prompt in simulation mode
./agent.py --headless                     Force headless browser mode
./agent.py --headed                       Force visible browser window
```

### CLI Commands (in REPL)
- Just type natural language: "open google.com", "click Images", "read page"
- `/browser [info|close|read]` - Browser controls
- `/context` - Show current page and recent actions
- `/execute on|off` - Toggle real execution vs dry-run
- `/exit` - Exit the CLI

### Detection Pipeline Query
- `click the gmail button` - Uses template
- `click the new tab icon` - Uses OCR spatial heuristics
- `type cats in the search box` - Uses OCR for location, then types

### AT-SPI Desktop Commands (after PR merge)
- `/desktop apps` - List available desktop applications
- `/desktop snapshot` - Take structured accessibility snapshot
- `/desktop click <target>` - Click via AT-SPI
- `/desktop type <text> in <target>` - Type via AT-SPI
- `/desktop read <target>` - Read element state via AT-SPI

---

## Date Log
- **2026-10-18**: Initial system, template matching bug
- **2026-10-19**: AT-SPI integration, YOLO confidence filtering, spatial heuristics
- **2026-10-19**: Phase 1-3 implementation (cursor overlay, focus fix, template strategy)
- **2026-10-20**: Moondream testing - failed
- **2026-10-20**: Reverted Moondream, decided to stick with hybrid system
- **2026-10-29**: Architectural pivot to structured-data-first (Playwright + AT-SPI)
- **2026-10-29**: Phase 3 - Playwright browser bridge completed
- **2026-10-31**: Phase 4 PR created - AT-SPI desktop automation with agent-desktop-inspired enhancements

---

## Final Verdict

**Current hybrid system (Playwright → AT-SPI → OCR → templates → YOLO) is the best approach for local UI automation on Raspberry Pi 5.**

Vision models (Moondream, Gemma) are not reliable enough yet. Focus on:
1. Merging Phase 4 PR (AT-SPI desktop automation)
2. Testing AT-SPI in real desktop session
3. Setting up VNC for remote viewing
4. Training more static templates
5. Adding voice input (Phase 5) when ready

**The system is nearly ready for full computer use.** With AT-SPI packages installed and tested, an AI agent can control both browser and desktop apps via natural language commands.
