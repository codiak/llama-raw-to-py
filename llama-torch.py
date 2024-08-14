import torch
from transformers import pipeline

model = "./hf_weights"

# Pipeline use: https://huggingface.co/docs/transformers/en/main_classes/pipelines
pipe = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.bfloat16,
    device="cpu"
)

messages = [
    # System prompt
    {"role": "system", "content": "You are a helpful assistant who always responds in a friendly manner."},
    {"role": "user", "content": "Who are you?"},
]

terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipe(
    messages,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
assistant_response = outputs[0]["generated_text"][-1]["content"]
print(f"\nResponse:\n{assistant_response}")
