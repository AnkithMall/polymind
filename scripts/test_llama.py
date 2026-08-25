from pathlib import Path

from llama_cpp import Llama


MODEL_PATH = (
    Path.home()
    / ".cache"
    / "polymind"
    / "models"
    / "Llama-3.2-3B-Instruct.Q5_K_M.gguf"
)


def main() -> None:
    print(f"Model: {MODEL_PATH}")
    print(f"Exists: {MODEL_PATH.exists()}")
    print()

    print("Loading model...")

    llm = Llama(
        model_path=str(MODEL_PATH),

        # Start conservatively.
        n_ctx=2048,

        # Offload as many layers as possible.
        n_gpu_layers=-1,

        # Keep output manageable.
        verbose=True,
    )

    print()
    print("Model loaded.")
    print("Generating response...")
    print()

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": "Explain what an operating system does in three sentences.",
            }
        ],
        max_tokens=100,
    )

    content = response["choices"][0]["message"]["content"]

    print("Response:")
    print(content)


if __name__ == "__main__":
    main()
