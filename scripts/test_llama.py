from pathlib import Path

from llama_cpp import Llama

from polymind.core.paths import model_dir
from polymind.core.model.registry import ModelRegistry


def find_model(name: str | None = None) -> Path | None:
    """Find a model by name or return the first available."""
    registry = ModelRegistry()
    models = registry.load()
    
    if not models:
        return None
    
    if name:
        # Try to find by name
        for model in models:
            if name.lower() in model.filename.lower():
                return Path(model.local_path)
    
    # Return first available model
    for model in models:
        path = Path(model.local_path)
        if path.exists():
            return path
    
    return None


def main() -> None:
    # Find a model to test
    model_path = find_model()
    
    if model_path is None:
        print("No models found. Run 'polymind model scan' first.")
        print()
        print("Available commands:")
        print("  polymind model search <query>  - Search for models")
        print("  polymind model download <repo> <file> - Download a model")
        print("  polymind model scan            - Scan for local models")
        return
    
    print(f"Model: {model_path}")
    print(f"Exists: {model_path.exists()}")
    print()
    
    print("Loading model...")
    
    llm = Llama(
        model_path=str(model_path),
        
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
