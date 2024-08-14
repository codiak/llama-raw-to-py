from llama_cpp import Llama

# Minimal example from:
# https://github.com/abetlen/llama-cpp-python

llm = Llama(
    model_path="./meta-llama-3-8B-instruct-Q8.gguf",
    n_gpu_layers=-1, # Use all available GPUs
    # seed=1337, # Uncomment to set a specific seed
    # n_ctx=2048, # Uncomment to increase the context window
)
output = llm(
    # Prompt
    "Context: You are a helpful assistant who always responds in a friendly manner.\nUser: Who are you?\nAssistant:",
    max_tokens=128, # Set to None to generate up to the end of the context window
    stop=["User:", "\n"],
    echo=True
)
print(output['choices'][0]['text'])
