# Vision Agent Harness - Implementation Plan

## Project Vision
Build a voice-controlled computer automation system that can navigate any interface by clicking, typing, and interacting with UI elements. The system must run entirely locally on a Raspberry Pi 5 with 4GB RAM.

## Current Status
- OCR-based text detection working (94% accuracy on text elements)
- TEV1 decision router operational
- YOLO icon detection operational (OmniParser ONNX model)
- Hybrid backend routing working (OCR for text, YOLO for icons)
- Memory usage: ~2.6GB (within constraints)
- Grid overlay system functional (20x20)
- Target marking system functional
- System can now detect both text labels and unlabeled icons

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

### Phase 3: Voice Input Integration
Add speech-to-text capability to drive the system via voice commands.

**Tasks:**
- Select lightweight STT model (Vosk, Whisper tiny, or similar)
- Integrate audio capture
- Add voice command parsing
- Connect STT output to existing query pipeline

**Expected Outcome:**
- Voice-controlled UI navigation
- Memory impact: TBD based on model choice

### Phase 4: Desktop Automation
Connect coordinate detection to actual mouse/keyboard control.

**Tasks:**
- Integrate pyautogui or accessibility API
- Implement click, type, scroll actions
- Add action validation and safety checks
- Create action queue for multi-step tasks

**Expected Outcome:**
- Actual computer control (not just detection)
- Memory impact: Minimal (<100MB)

### Phase 5: Agent Loop
Create autonomous agent that can execute multi-step tasks.

**Tasks:**
- Design task planning system
- Implement state tracking (what's on screen, what's been done)
- Add error recovery and retry logic
- Create task completion verification

**Expected Outcome:**
- "Open Gmail and send email" style commands
- Architecture TBD based on remaining memory budget

### Phase 6: Optimization and Refinement
Improve performance, accuracy, and reliability.

**Tasks:**
- Cache common UI element locations
- Optimize model loading/unloading
- Add parallel processing where possible
- Reduce latency through algorithmic improvements
- Comprehensive testing across applications

**Expected Outcome:**
- Faster response times
- Better accuracy on edge cases
- More reliable operation

## Architecture Notes

### Memory Budget
- Current: 2.6GB
- Maximum: 4GB
- Remaining: 1.4GB for future phases

### Technology Stack Decisions
- OCR: Tesseract (fixed)
- Decision router: TEV1:0.8B (fixed)
- Icon detection: OmniParser icon_detect + ONNX Runtime (completed, 12MB model)
- Voice input: TBD (Phase 3)
- Desktop control: TBD (Phase 4)
- Agent system: TBD (Phase 5)

### Constraints
- Must run entirely local (no cloud APIs)
- Must work on Raspberry Pi 5 (4GB RAM)
- Must be CPU-only (no GPU acceleration)
- Latency should be under 5 seconds for simple actions

## Next Steps
1. Phase 1: Heuristic rules - COMPLETED (unsuccessful, icons not OCR'd as text)
2. Phase 2: YOLO Icon Detection - COMPLETED
   - Selected OmniParser icon_detect ONNX model
   - Integrated ONNX Runtime inference pipeline
   - Added hybrid backend routing with TEV1
   - Tested on new tab, close, menu icons
3. Phase 3: Voice Input Integration - NEXT
   - Select lightweight STT model
   - Integrate audio capture
   - Add voice command parsing
   - Connect STT to existing query pipeline
