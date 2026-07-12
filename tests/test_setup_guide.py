from polymind.core import check_provider_health, print_setup_guide
from polymind.core.config import Config


def test_check_provider_health_no_config():
    issues = check_provider_health(Config(models=[]))
    assert len(issues) > 0


def test_check_provider_health_no_models():
    issues = check_provider_health(Config(models=[]))
    assert any("model" in i.lower() for i in issues)


def test_print_setup_guide_contains_steps():
    guide = print_setup_guide()
    assert "Setup" in guide or "Step" in guide
    assert "Ollama" in guide or "ollama" in guide
