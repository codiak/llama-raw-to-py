import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os

model_path = "./hf-weights"

# Use Faster Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Convert model to float32 for potential CPU speed-up (depends on application)
# model = model.to(torch.float32)

# Convert to TorchScript for speed-up
# model = torch.jit.trace(model, torch.zeros(1, 1, dtype=torch.long))

# Implement Quantization
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
)

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

if torch.backends.mps.is_available():
    device = torch.device("mps")
    # model = model.to(device)
    model = model.to(device, torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float32)
else:
    device = torch.device("cpu")

# Create pipeline with the quantized model
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)

# Pre-defined system message to establish the chatbot's persona
system_message = {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"}

# Interactive CLI loop
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the pirate chatbot. Farewell, matey!")
        break

    messages = [
        system_message,
        {"role": "user", "content": user_input},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=120,
        eos_token_id=pipe.tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    assistant_response = outputs[0]["generated_text"][-1]["content"]
    print(f"Pirate Bot: {assistant_response}")
