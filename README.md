## llama-raw-to-py

### Goal

This repository is intended to be a quickstart for taking Llama 3.1 weights directly from Meta, and preparing them to use in Python on a Mac.

## Requirements

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

- Fill the form on Meta’s website - https://llama.meta.com/llama-downloads/
- You will promptly get a email with a URL
- Go the repo in the email, and you will find a download.sh file.
- Run `download.sh` and paste the link from the email.
- Save the weights to a `llama_weights` folder in this repo, so it looks something like:
```
llama-raw-to-py/
    ├─ llama_weights/
      │  ├─ api/
      │  ├─ Meta-Llama-3.1-8B/
      │  ├─ Meta-Llama-3.1-8B-Instruct/
      │  ├─ ...
    ├─ llama-cpp.py
    ├─ llama-torch.py
    ├─ ...
```

## Converting Weights

In order to use Llama weights with llama.cpp, they need to be in GGUF format. As an intermediary step, we will convert them to HuggingFace's format, which will also make them usable in PyTorch:

```
python3 .venv/lib/python3.12/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir llama_weights/Meta-Llama-3.1-8B-Instruct/ --model_size 8B --output_dir hf_weights --llama_version 3.1 --instruct True
```

At this point you can test running `llama-torch.py` or `llama-torch-cli.py`, and it should be functioning, albeit slow on most Macs. I recommend continuing on to quantize the weights and run the model via llama.cpp for a more efficient integration.

We will use a conversion utility from llama.cpp to convert to GGUF. To simplify accessing llama.cpp scripts, build it directly in the repo:
```
git clone https://github.com/ggerganov/llama.cpp.git
make -C llama.cpp/
```

Convert to GGUF:
```
python3 ./llama.cpp/convert_hf_to_gguf.py hf_weights/ --outtype f32 --outfile meta-llama-3-8B-instruct.gguf
```

## Quantize and Run

At this point you have a workable .gguf file! Now we'll want to quantize it to run it more efficiently:
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
