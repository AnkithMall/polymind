from polymind.core.config import Config
from polymind.core.types import ExecutionStrategy


def test_config_routing_profile_quality():
    config = Config(profile="quality")
    resolved = config.get_resolved_profile()
    assert resolved["scheduler"]["strategy"] == "model_aware"
    assert resolved["scheduler"]["pass_context"] is True


def test_config_routing_profile_fast():
    config = Config(profile="fast")
    resolved = config.get_resolved_profile()
    assert resolved["scheduler"]["strategy"] == "sequential"


def test_config_routing_profile_private():
    config = Config(profile="private")
    resolved = config.get_resolved_profile()
    assert resolved["scheduler"]["strategy"] == "sequential"


def test_config_no_profile():
    config = Config()
    resolved = config.get_resolved_profile()
    assert resolved["scheduler"]["strategy"] == "model_aware"


def test_config_profile_merges_fields():
    config = Config(profile="quality", verbose=True)
    resolved = config.get_resolved_profile()
    assert resolved["verbose"] is True
    assert resolved["scheduler"]["strategy"] == "model_aware"


def test_config_keep_alive():
    config = Config(keep_alive="5m")
    assert config.keep_alive == "5m"
    resolved = config.get_resolved_profile()
    assert resolved["keep_alive"] == "5m"


def test_config_default_yaml_has_new_fields():
    yaml_str = Config.default_yaml()
    assert "profile" in yaml_str
    assert "keep_alive" in yaml_str
