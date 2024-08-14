## llama-raw-to-py

### Goal

This repository is intended to be a quickstart for taking Llama 3.1 weights directly from Meta, and preparing them to use in Python on a Mac.

## 1. Requirements

Set up Python env and install requirements. Highly recommend using a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
```

Install PyTorch dependencies:
```
pip install transformers 'transformers[torch]' tiktoken blobfile sentencepiece
```

Install llama.cpp dependencies (note the environment variable enables using Metal to accelerate on Apple Silicon)
```
export CMAKE_ARGS="-DLLAMA_METAL=on"
export FORCE_CMAKE=1
pip install llama-cpp-python --no-cache-dir
```

## 2. Downloading the Model

To get the Llama 3 weights:
- Complete the form on Meta’s website - https://llama.meta.com/llama-downloads/
- A download link will be generated for you, and a link to a repo will be provided.
- Clone the [provided Llama repo](https://github.com/meta-llama/llama-models/blob/main/README.md), and locate the Llama 3.1 download.sh file.
  - Here is the direct link to the download.sh file: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/download.sh
- You will need `wget` installed on your machine to download the weights.
  - If you don't have `wget`, you can install it with [Homebrew](https://brew.sh/): `brew install wget`
- Run `download.sh` and paste the generated link when prompted.
- *Note:* The smaller model with 8 billion parameters is **16GB**! For local purposes, especially on a MacBook M1-M2, I recommend downloading the 8B weights, and the Instruct variant for usability.
- Save the weights to a `llama_weights` folder in this repo, so it looks something like:
```
llama-raw-to-py/
    ├─ llama_weights/
      │  ├─ api/
      │  ├─ Meta-Llama-3.1-8B-Instruct/  <-- the folder of model metadata and weights you downloaded
      │  ├─ ...
    ├─ llama-cpp.py
    ├─ llama-torch.py
    ├─ ...
```

## 3. Converting Weights

In order to use Llama weights with llama.cpp, they need to be in GGUF format. As an intermediary step, we will convert them to HuggingFace's safetensors format, which will also make them usable in PyTorch. Make sure to update this command to reflect your version of Python (`python --version`) and the version of the model you downloaded:

```
python .venv/lib/python3.12/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir llama_weights/Meta-Llama-3.1-8B-Instruct/ --model_size 8B --output_dir hf_weights --llama_version 3.1 --instruct True
```

At this point you can test running `llama-torch.py` or `llama-torch-cli.py`, and it should be functioning, albeit quite slow on most Macs. I recommend continuing on to quantize the weights and run the model via llama.cpp for a more efficient integration.

We will use a conversion utility from llama.cpp to convert to GGUF. To simplify accessing llama.cpp scripts, build it directly in the repo:
```
git clone https://github.com/ggerganov/llama.cpp.git
make -C llama.cpp/
```

Convert to GGUF:
```
python ./llama.cpp/convert_hf_to_gguf.py hf_weights/ --outtype f32 --outfile meta-llama-3-8B-instruct.gguf
```

## 4. Quantize and Run

At this point you have a workable GGUF file! Now we'll want to quantize it to run it more efficiently:
```
./llama.cpp/llama-quantize meta-llama-3-8B-instruct.gguf meta-llama-3-8B-instruct-Q8.gguf Q8_0
```
Quantizing makes a big difference, here are response times for the same number of tokens on my M2 MacBook Air:
- Not quantized, llama_cpp: 8-9 minutes
- Quantized, llama_cpp: 7-10 seconds

Here is a good explanation from Ricardo Pascal, who made this guide possible:
> Quantization simplifies the model by representing its internal data with smaller numbers. This makes mathematical operations easier and faster. However, this simplification can lead to a slight decrease in the model’s accuracy, such as less certainty about the next word it should output.

You can test out your quantized weights using llama.cpp directly:
```
# Test run via CLI / interactive mode
./llama.cpp/llama-cli -m meta-llama-3-8B-instruct-Q8.gguf -n 512 --n-gpu-layers 0 --repeat_penalty 1.0 --color -i -r "User:" -f llama.cpp/prompts/chat-with-bob.txt
```

Or go ahead and use the included Python implementations:
```
# Simple test
python3 ./llama-cpp.py
# Interactive chat
python3 ./llama-cpp-cli.py
```

### Sources

- Inspiration for this repo https://github.com/ggerganov/llama.cpp/issues/8808
- Downloading weights https://discuss.huggingface.co/t/how-to-use-gated-models/53234/8
- Quantizing and running with llama.cpp https://voorloopnul.com/blog/quantize-and-run-the-original-llama3-8b-with-llama-cpp/
- PyTorch MPS backend out of memory fix https://pnote.eu/notes/pytorch-mac-setup/
