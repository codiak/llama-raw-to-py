from llama_cpp import Llama

# Initialize the Llama model
llm = Llama(
      model_path="./meta-llama-3-8B-instruct-Q8.gguf",
      n_gpu_layers=-1, # Comment to disable GPU acceleration
      # seed=1337,
      # n_ctx=2048,
)
exit_msg = "\nExiting CLI. Goodbye!"
context = "Context: You are a helpful assistant who always responds in a friendly manner.\n"
conversation = context

while True:
    try:
        # Read user input
        user_input = input("Enter your prompt (or 'exit' to quit, 'reset' for new chat): ")

        if user_input.lower() == 'exit':
            print(exit_msg)
            break

        if user_input.lower() == 'reset':
            conversation = context
            print("Chat reset.")
            continue

        conversation += f"User: {user_input} \nAssistant: "
        output = llm(
              conversation,
              max_tokens=256, # 'None' to generate up to the end of the context window
              stop=["User:"],
              echo=True # Echo the prompt back in the output
        )
        print("\nResponse:")
        conversation = output['choices'][0]['text']
        print(conversation)

    except KeyboardInterrupt:
        print(exit_msg)
        break
