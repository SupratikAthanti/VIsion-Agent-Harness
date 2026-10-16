# Vision Agent Harness - Implementation Plan

## Project Vision
Build a voice-controlled computer automation system that can navigate any interface by clicking, typing, and interacting with UI elements. The system must run entirely locally on a Raspberry Pi 5 with 4GB RAM.

## Current Status
- OCR-based text detection working (94% accuracy on text elements)
- TEV1 decision router operational
- Memory usage: 2.6GB (within constraints)
- Grid overlay system functional (20x20)
- Target marking system functional
- Heuristic rules implemented but failed (icons not OCR'd as text)

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

### Phase 2: YOLO Icon Detection (Current)
Integrate YOLOv8n with NCNN for general icon and logo recognition.

**Research completed:**
- Identified UI detection models: ScreenParser (YOLO11-L), GPA-GUI-Detector, UI element datasets
- Found NCNN conversion workflow: PyTorch -> TorchScript -> PNNX -> NCNN
- Available UI models mostly in PyTorch format, need conversion to NCNN

**Tasks:**
- Select appropriate UI detection model (ScreenParser or custom fine-tune on icon dataset)
- Convert model to NCNN format:
  1. Export PyTorch model to TorchScript
  2. Use PNNX to convert TorchScript to NCNN
  3. Modify PNNX script for dynamic shape inference
  4. Re-export and convert with proper input shapes
- Integrate NCNN inference pipeline in Python
- Add YOLO results to hybrid backend routing
- Implement icon-to-text mapping for common logos
- Test on icons (new tab +, close X, menu) and logos (YouTube, Gmail, etc.)

**Expected Outcome:**
- Recognize icons without text
- Detect logos and branded elements
- Memory impact: +300-400MB (total ~3GB)

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
- Icon detection: YOLOv8n + NCNN (planned)
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
2. Phase 2: YOLO Icon Detection - IN PROGRESS
   - Select UI detection model
   - Convert model to NCNN format
   - Integrate inference pipeline
   - Test on common icons
