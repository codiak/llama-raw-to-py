## llama-raw-to-py

### Goal

This repository is intended to be a quickstart for taking Llama 4 weights directly from Meta, and preparing them to use in Python on a Mac.

## 1. Requirements

Set up Python env and install requirements. Highly recommend using a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
```

Install PyTorch dependencies:
```
pip install transformers 'transformers[torch]' tiktoken blobfile sentencepiece llama-models
```

Install llama.cpp dependencies (note the environment variable enables using Metal to accelerate on Apple Silicon)
```
export CMAKE_ARGS="-DLLAMA_METAL=on"
export FORCE_CMAKE=1
pip install llama-cpp-python --no-cache-dir
```


## 2. Downloading the Model

To get the Llama 4 weights:

### Step 1: Request Access
- Complete the form on Meta's website - https://www.llama.com/llama-downloads/
- Read and accept the license agreement
- Once approved, you'll receive a signed URL via email
- **Important:** Download links expire after 24 hours and have usage limits. If you encounter "403: Forbidden" errors, re-request a new link from the website.
- When copying the URL from the email, copy the URL text itself (starts with https://download.llamameta.net), not using 'Copy link address'

### Step 2: List Available Models
After installing the `llama-models` package (in step 1), you can view available models:
```
llama-model list
```

Or to see all versions including older releases:
```
llama-model list --show-all
```

### Step 3: Download the Model
Run the download command and provide your signed URL when prompted:
```
llama-model download --source meta --model-id Llama4-Scout-17B-16E-Instruct
```

Available Llama 4 models:
- **Llama4-Scout-17B-16E** - 17B active parameters (109B total), 10M token context window
- **Llama4-Maverick-17B-128E** - 17B active parameters (400B total), 1M token context window

Both models are available in Base and Instruct variants. For local purposes, especially on a MacBook M1-M2, use the Scout model with Instruct variant for usability.

**Note:** Llama 4 models require significant resources - at least 4 GPUs to run at full (bf16) precision. Quantization is highly recommended for Mac usage.

### Step 4: Verify Download (Optional)
Verify the integrity of downloaded files:
```
llama-model verify-download
```

The weights will be saved to your llama-models cache directory. The typical structure looks like:
```
llama-raw-to-py/
    ├─ ~/.llama/checkpoints/Llama4-Scout-17B-16E-Instruct/  <-- the folder of model metadata and weights
    ├─ llama-cpp.py
    ├─ llama-torch.py
    ├─ ...
```
## 3. Converting Weights

In order to use Llama weights with llama.cpp, they need to be in GGUF format. As an intermediary step, we will convert them to HuggingFace's safetensors format, which will also make them usable in PyTorch. Make sure to update this command to reflect your version of Python (`python --version`) and the path to your downloaded model:

```
python .venv/lib/python3.12/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ~/.llama/checkpoints/Llama4-Scout-17B-16E-Instruct/ --model_size 17B --output_dir hf_weights --llama_version 4 --instruct True
```

At this point you can test running `llama-torch.py` or `llama-torch-cli.py`, and it should be functioning, albeit quite slow on most Macs. I recommend continuing on to quantize the weights and run the model via llama.cpp for a more efficient integration.

We will use a conversion utility from llama.cpp to convert to GGUF. To simplify accessing llama.cpp scripts, build it directly in the repo:
```
git clone https://github.com/ggerganov/llama.cpp.git
make -C llama.cpp/
```

Convert to GGUF:
```
python ./llama.cpp/convert_hf_to_gguf.py hf_weights/ --outtype f32 --outfile llama-4-scout-17B-instruct.gguf
```

## 4. Quantize and Run

At this point you have a workable GGUF file! Now we'll want to quantize it to run it more efficiently:
```
./llama.cpp/llama-quantize llama-4-scout-17B-instruct.gguf llama-4-scout-17B-instruct-Q8.gguf Q8_0
```

Quantizing makes a big difference, here are response times for the same number of tokens on my M2 MacBook Air:
- Not quantized, llama_cpp: 8-9 minutes
- Quantized, llama_cpp: 7-10 seconds

Here is a good explanation from Ricardo Pascal, who made this guide/repo possible:
> Quantization simplifies the model by representing its internal data with smaller numbers. This makes mathematical operations easier and faster. However, this simplification can lead to a slight decrease in the model’s accuracy, such as less certainty about the next word it should output.

You can test out your quantized weights using llama.cpp directly:
```
# Test run via CLI / interactive mode
./llama.cpp/llama-cli -m llama-4-scout-17B-instruct-Q8.gguf -n 512 --n-gpu-layers 0 --repeat_penalty 1.0 --color -i -r "User:" -f llama.cpp/prompts/chat-with-bob.txt
```

Or go ahead and use the included Python implementations:
```
# Simple test
python ./llama-cpp.py
# Interactive chat
python ./llama-cpp-cli.py
```

### Sources

- Official Meta Llama download page: https://www.llama.com/llama-downloads/
- Meta Llama models repository: https://github.com/meta-llama/llama-models
- Meta Llama 4 announcement: https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- Inspiration for this repo: https://github.com/ggerganov/llama.cpp/issues/8808
- Quantizing and running with llama.cpp: https://voorloopnul.com/blog/quantize-and-run-the-original-llama3-8b-with-llama-cpp/
- PyTorch MPS backend out of memory fix: https://pnote.eu/notes/pytorch-mac-setup/
