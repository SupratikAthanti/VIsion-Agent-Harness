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

## 3. Run Inference with `llama.cpp`
Once you have `llama.cpp` built and your model weights ready, you can run inference pointing to your image in the `images/` folder:
```bash
llama-minicpmv-cli -m models/smolvlm-256m-instruct.gguf --mmproj models/mmproj-smolvlm-256m-instruct.gguf --image images/test.jpg -p "Describe this image in one sentence."
```
*(Note: Exact binary names and flags depend on your specific model architecture and `llama.cpp` version).*
