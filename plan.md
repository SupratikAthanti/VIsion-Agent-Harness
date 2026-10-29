# Vision Agent Harness - Development Log & Lessons Learned

## System Overview
**Goal**: Build a local "Vision Agent Harness" for voice-driven OS and browser automation on Raspberry Pi 5 (4GB RAM).
**Constraint**: Local-only, no cloud APIs, keep memory under ~3GB.

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
- **Status**: ✅ Implemented but ⚠️ Untested
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

---

## Current System Architecture

### Detection Pipeline (in order)
1. **AT-SPI** (if available) - Find by role/name, position-independent
2. **OCR with spatial heuristics** - Find text, then nearby icons (+, x, ≡, 🔍)
3. **Static templates** - Only for logos/fixed buttons (Gmail, YouTube, AI mode)
4. **YOLO** - Fallback with confidence filtering (>50%)
5. **Failure** - Suggest `/annotate` to train

### Action Execution
- **Cursor overlay**: Shows blue arrow at target before clicking (0.6s preview)
- **AT-SPI Action**: Tries accessibility-based clicking first (background-friendly)
- **Window focus**: Falls back to wmctrl to focus window before xdotool click
- **Retry logic**: Click with small pixel offsets if first click fails

### Trained Templates (static only)
- ai mode button
- gmail button
- youtube
- x (close button)

### Blocked from Training (dynamic)
- new tab icon (use OCR spatial heuristics)
- search box (use OCR/AT-SPI)
- menu buttons (use OCR spatial heuristics)

---

## What Works Well

1. **Text-based elements**: OCR finds them reliably
2. **Static icons**: Template matching works for logos and fixed buttons
3. **Common icon patterns**: Spatial heuristics handle +, x, ≡, 🔍
4. **Cursor visualization**: Can see where clicks will land
5. **Window focus**: Doesn't steal focus unnecessarily (with AT-SPI fix)

---

## What Doesn't Work Yet

1. **AT-SPI**: Not tested - needs system packages and GUI desktop session
2. **Vision models**: Moondream failed, Gemma Vision too large for 4GB RAM
3. **Semantic understanding**: No model that can "understand" UI context reliably
4. **Flat input fields**: Template matching fails for search bars/text fields

---

## Hardware & RAM

- **Raspberry Pi 5**: 4GB RAM
- **OS usage**: ~1.5GB
- **Available for models**: ~2.5GB
- **Current system**: Fits within limit (OCR + YOLO + templates)
- **Gemma Vision**: Would need ~5GB total (too large)
- **Moondream 4-bit**: Fits (2.4GB) but performance is poor

---

## Future Recommendations

### DO NOT Try Again
1. **Moondream for UI automation** - Already tested, hallucinates and unreliable
2. **Gemms Vision on RPi 4GB** - Doesn't fit RAM (needs 5GB+)
3. **Training templates for dynamic elements** - Will break when layout changes
4. **Pure coordinate-based approaches** - Fragile and position-dependent

### WORTH Exploring Later
1. **AT-SPI in real desktop session** - Install packages, test in GUI environment
2. **VNC/remote desktop** - View RPi from Chromebook to see cursor movements
3. **Better small VLMs** - Wait 1-2 years for models to improve (currently not ready)
4. **Modular approach (Nemo-style)** - Keep YOLO for detection, small LLM for reasoning (what we already have)

### Current Focus
1. **Test Phase 1-3 improvements** with real execution mode
2. **Set up VNC** to view RPi desktop remotely
3. **Train more static templates** for frequently-used elements
4. **Improve OCR spatial heuristics** for more icon patterns

---

## Key Lessons

1. **Layered approaches beat single models**: AT-SPI → OCR → templates → YOLO is more reliable than any single vision model
2. **Position independence matters**: Templates break when layouts change, semantic approaches don't
3. **Benchmark numbers ≠ real performance**: ScreenSpot 80.4% doesn't mean it works on your screenshots
4. **Small VLMs aren't ready yet**: They hallucinate and struggle with domain gaps
5. **Cursor visualization is essential**: Can't debug automation without seeing where it clicks
6. **Accessibility APIs are powerful**: AT-SPI can click without window focus - use when available
7. **RAM constraints are real**: 4GB RPi limits model choices, work within it
8. **Your current system is state-of-the-art**: Matches what AutoControlGUI, OSWright use locally

---

## Commands Reference

### Test the System
```bash
venv/bin/python vision_cli.py --backend hybrid
```

### CLI Commands
- `/annotate` - Train a new template (browser tool)
- `/execute on` - Enable real mouse/keyboard control
- `/execute off` - Dry-run mode only
- `/list-templates` - Show trained templates
- `/delete-template <name>` - Remove a template
- `/context` - Show current page and recent actions

### Detection Pipeline Query
- `click the gmail button` - Uses template
- `click the new tab icon` - Uses OCR spatial heuristics
- `type cats in the search box` - Uses OCR for location, then types

---

## Date Log
- **2026-10-18**: Initial system, template matching bug
- **2026-10-19**: AT-SPI integration, YOLO confidence filtering, spatial heuristics
- **2026-10-19**: Phase 1-3 implementation (cursor overlay, focus fix, template strategy)
- **2026-10-20**: Moondream testing - failed
- **2026-10-20**: Reverted Moondream, decided to stick with hybrid system

---

## Final Verdict

**Current hybrid system (AT-SPI → OCR → templates → YOLO) is the best approach for local UI automation on Raspberry Pi 5.**

Vision models (Moondream, Gemma) are not reliable enough yet. Focus on:
1. Testing AT-SPI in real desktop session
2. Setting up VNC for remote viewing
3. Training more static templates
4. Improving OCR spatial heuristics

Don't waste time on vision models until small VLMs improve significantly (1-2 years).
