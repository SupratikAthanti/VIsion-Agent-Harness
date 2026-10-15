# Vision Agent Harness - Minimal Setup

To get started with your simple workflow (downloading a vision model, capturing/placing an image in `images/`, and querying the model), follow these steps:

## 1. Create the Images Folder
A script has been provided to set up the environment:
```bash
python3 setup.py
```
This creates an `images/` directory. Place your screenshots in here (e.g., `images/test.jpg`).

## 2. Download a Vision Model
For a CPU-only Raspberry Pi / local environment, we recommend a compact multimodal model such as **SmolVLM-256M** or **Moondream2** in GGUF format via Hugging Face.

You can download model weights into a `models/` directory using `huggingface-cli` or python:
```bash
pip install huggingface_hub
huggingface-cli download ggml-org/SmolVLM-256M-Instruct-GGUF --local-dir models/
```

## 3. Run the Interactive CLI (v0.1)

Activate your virtual environment and start the CLI:
```bash
source venv/bin/activate
python3 vision_cli.py
```

- Drop any screenshot or image into the `images/` folder (e.g. `images/screenshot.png`).
- By default, it automatically picks the first image alphabetically.
- Type any question directly at the prompt:
  ```
  [screenshot.png] > Describe the layout of this web page
  ```
- Helpful CLI commands:
  - `/reload` : Re-scans `images/` after you drop in a new screenshot
  - `/list` : Lists all images currently in `images/`
  - `/select <filename>` : Switches active image
  - `/exit` : Exits the CLI

