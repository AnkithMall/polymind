from polymind.core.providers import ProviderInfo, ProviderType, resolve_model_string


def test_resolve_ollama():
    info = resolve_model_string("ollama/llama3.2:1b")
    assert info.provider == ProviderType.ollama
    assert info.model_name == "llama3.2:1b"
    assert info.litellm_string == "ollama/llama3.2:1b"


def test_resolve_openai():
    info = resolve_model_string("openai/gpt-4o")
    assert info.provider == ProviderType.openai
    assert info.litellm_string == "openai/gpt-4o"


def test_resolve_anthropic():
    info = resolve_model_string("anthropic/claude-3-opus-20240229")
    assert info.provider == ProviderType.anthropic
    assert info.litellm_string == "anthropic/claude-3-opus-20240229"


def test_resolve_openrouter():
    info = resolve_model_string("openrouter/mistral-7b")
    assert info.provider == ProviderType.openrouter


def test_resolve_lm_studio():
    info = resolve_model_string("lm_studio/local-model")
    assert info.provider == ProviderType.lm_studio


def test_resolve_no_slash():
    info = resolve_model_string("llama3.2:1b")
    assert info.provider == ProviderType.ollama
    assert info.model_name == "llama3.2:1b"


def test_resolve_with_provider_override():
    info = resolve_model_string("gpt-4o", provider="openai")
    assert info.provider == ProviderType.openai
    assert info.model_name == "gpt-4o"


def test_build_kwargs():
    info = ProviderInfo(
        provider=ProviderType.ollama,
        model_name="llama3.2:1b",
        base_url="http://localhost:11434",
        api_key=None,
    )
    kwargs = info.build_kwargs()
    assert kwargs["api_base"] == "http://localhost:11434"
    assert "api_key" not in kwargs
