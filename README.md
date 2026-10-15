# Vision-Agent-Harness

An evaluation guide for building a CPU-only vision agent on a Raspberry Pi, using `llama.cpp` where the selected model and its vision components are supported.

## Goal

The agent receives an image (for example, from a camera), answers a constrained visual question, and returns a small structured action for the host application to validate and execute. Inference runs locally on the Pi CPU; there is no dependency on a GPU or hosted model API.

## Candidate models

| Model | Approx. size | Fit for this project | Important compatibility note |
| --- | ---: | --- | --- |
| SmolVLM-256M | 256M parameters | Smallest VLM candidate; start here when memory and latency are tight. | Confirm the exact model revision is supported by the installed `llama.cpp` multimodal path and that a compatible GGUF and vision projector are available. |
| SmolVLM-500M | 500M parameters | A larger SmolVLM candidate to compare for answer quality. | Same artifact and runtime checks as SmolVLM-256M; expect higher memory use and latency. |
| Moondream 0.5B | About 0.5B parameters | Compact VLM candidate with image-question-answering use cases. | Model name and packaging vary. Verify the precise checkpoint, GGUF conversion, and multimodal runtime support rather than assuming any Moondream checkpoint will load. |
| Quantized Moondream2 | Depends on quantization | Potentially the most practical Moondream option if a compatible quantized artifact is available. | Check the artifact's architecture and vision-component requirements. “Quantized” alone does not guarantee compatibility with `llama.cpp`. |
| YOLO nano | Varies by YOLO version | Useful for fast object detection, such as finding people or objects in a frame. | YOLO is not a vision-language model and does not run as a model in `llama.cpp`. Use a separate CPU-capable runtime and let the agent call it as a detection tool. |

Model support changes over time and can differ between checkpoints, quantizations, and `llama.cpp` revisions. Treat compatibility as a test to perform on the exact files you intend to deploy, not as a property of a model family name.

## Recommended approach

1. Begin with SmolVLM-256M or a quantized Moondream2 artifact only after confirming its exact `llama.cpp` multimodal support. Compare SmolVLM-500M if the smaller model is too inaccurate and the Pi has enough memory and latency headroom.
2. Keep YOLO nano as an optional perception tool when the task is object detection rather than open-ended image understanding. It requires its own model/runtime and increases system complexity.
3. Test on the actual Raspberry Pi model, OS, and camera input. Record peak resident memory, warm and cold latency, sustained throughput, and task accuracy before choosing a default.

## Compatibility smoke test

Before building the agent, verify the following for the specific model revision and quantization:

- The checkpoint can be represented by an artifact format supported by your `llama.cpp` build.
- The build supports the model's architecture and its multimodal image path. Vision models may require a separate projector or other vision artifact in addition to the language-model file.
- A minimal image-question-answering prompt works on the target Pi, not just on a development machine.
- The model's license and usage terms suit the intended deployment.

Use the current `llama.cpp` documentation and the model's own model card for the supported command, required files, and conversion steps. Do not assume that a successful text-only load proves image input works. If the model fails the multimodal test, select another supported checkpoint or use a separate inference runtime; do not silently fall back to text-only behavior.

## Agent shape

Keep model inference separate from action execution:

1. Capture a frame and resize or crop it to the model's supported input dimensions.
2. Ask one focused question with a short prompt and request a constrained response, such as JSON with an `observation` and an `action` field.
3. Parse and validate the response against an allowlist and application rules.
4. Execute only validated actions. On invalid output, timeout, or inference failure, take a defined no-op or safe fallback.

For YOLO, replace or augment the VLM observation step with detections from its separate runtime. Pass only useful detections to the agent, and retain the same validation boundary before any action is executed. A model response is untrusted input, not authorization to control hardware.

## Raspberry Pi CPU considerations

- Start with a low-bit quantization supported by the selected model and runtime; verify output quality as well as memory savings.
- Limit image resolution, prompt length, and generated tokens. These directly affect latency and memory use.
- Benchmark thread counts on the target Pi. More threads are not always faster, particularly under thermal throttling.
- Measure sustained performance with the case and cooling configuration you expect to use. Monitor temperature and throttling during longer runs.
- Leave memory for the OS, camera pipeline, and application. Include the vision encoder/projector and runtime buffers when measuring peak memory.
- Run inference locally with network access disabled during the test to confirm the runtime has no hidden service dependency.

## Evaluation checklist

Test the same set of representative images on each viable candidate and record:

| Measurement | What to record |
| --- | --- |
| Hardware and software | Pi model, RAM, OS, `llama.cpp` revision, model revision, quantization |
| Compatibility | Image input works; required model and vision files are documented |
| Latency | Cold start and warm end-to-end response time |
| Memory | Peak resident memory under the intended workload |
| Quality | Correct observations and safe structured outputs on the test set |
| Stability | Sustained run duration, temperature, throttling, and failure behavior |

Choose the smallest model that meets the task's accuracy and response-time requirements on the target hardware. No model is selected as the default until these checks pass.

