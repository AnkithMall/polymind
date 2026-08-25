from collections.abc import Iterator
from pathlib import Path
from typing import cast

from llama_cpp import (
    ChatCompletionRequestMessage,
    CreateChatCompletionStreamResponse,
    Llama,
)

from polymind.core.runtime.types import RuntimeConfig


class RuntimeRunner:
    def __init__(
        self,
        config: RuntimeConfig,
        model_path: Path,
    ):
        self.config = config
        self.model_path = model_path

        self.llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=config.gpu_layers,
            n_threads=config.threads,
            n_ctx=config.context_size,
            n_batch=config.batch_size,
            verbose=False,
        )

    def chat(self) -> None:
        print("Polymind runtime")
        print("Type /exit to quit.")
        print()

        messages: list[ChatCompletionRequestMessage] = []

        while True:
            try:
                prompt = input(">>> ")
            except (KeyboardInterrupt, EOFError):
                print()
                break

            if prompt.strip() == "/exit":
                break

            if not prompt.strip():
                continue

            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            response_stream = cast(
                Iterator[CreateChatCompletionStreamResponse],
                self.llm.create_chat_completion(
                    messages=messages,
                    stream=True,
                ),
            )

            content_parts: list[str] = []

            for chunk in response_stream:
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content")

                if content:
                    print(
                        content,
                        end="",
                        flush=True,
                    )
                    content_parts.append(content)

            print()
            print()

            assistant_content = "".join(content_parts)

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                }
            )
