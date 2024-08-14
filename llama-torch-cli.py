import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os

model = "./hf_weights"

pipe = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.bfloat16,
    device="cpu"
)

# System prompt
system_message = {"role": "system", "content": "You are a helpful assistant who always responds in a friendly manner."}
messages = [system_message]

# Interactive CLI loop
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting CLI. Goodbye!")
        break
    if user_input.lower() == "reset":
        messages = [system_message]
        print("Chat reset.")
        continue

    messages.append({"role": "user", "content": user_input})

    outputs = pipe(
        messages,
        max_new_tokens=256,
        eos_token_id=pipe.tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    assistant_response = outputs[0]["generated_text"][-1]["content"]
    messages.append({"role": "assistant", "content": assistant_response})
    print(f"Assistant: {assistant_response}")
