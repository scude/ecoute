import os
from pathlib import Path
from unittest.mock import patch

import yaml
import pytest

from pipeline_config import _deep_merge, load_config, DEFAULT_CONFIG


def test_deep_merge():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"d": 4, "e": 5}, "f": 6}
    expected = {"a": 1, "b": {"c": 2, "d": 4, "e": 5}, "f": 6}
    assert _deep_merge(base, override) == expected


def test_load_config_defaults(tmp_path):
    # Non-existent config file -> should return defaults
    config = load_config(tmp_path / "non_existent.yaml")
    assert config["whisper"]["model_size_or_path"] == "medium"
    assert config["vad"]["threshold"] == 0.45


def test_load_config_with_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    file_data = {
        "whisper": {"model_size_or_path": "large", "language": "en"},
        "paths": {"input_audio_dir": "/tmp/audios"}
    }
    config_file.write_text(yaml.dump(file_data))
    
    config = load_config(config_file)
    assert config["whisper"]["model_size_or_path"] == "large"
    assert config["whisper"]["language"] == "en"
    assert config["whisper"]["device"] == "cpu" # still default
    assert config["paths"]["input_audio_dir"] == "/tmp/audios"


def test_load_config_with_env_overrides(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}") # empty file
    
    env_vars = {
        "WHISPER_MODEL_SIZE_OR_PATH": "tiny",
        "VAD_THRESHOLD": "0.1",
        "WHISPER_CONDITION_ON_PREVIOUS_TEXT": "true",
        "PIPELINE_INTERVAL_SECONDS": "10"
    }
    
    with patch.dict(os.environ, env_vars):
        config = load_config(config_file)
        assert config["whisper"]["model_size_or_path"] == "tiny"
        assert config["vad"]["threshold"] == 0.1
        assert config["whisper"]["condition_on_previous_text"] is True
        assert config["pipeline"]["interval_seconds"] == 10


def test_load_config_invalid_format(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid yaml [")
    # yaml.safe_load will raise YAMLError
    with pytest.raises(Exception):
        load_config(config_file)
    
    config_file.write_text("- not a dict")
    with pytest.raises(ValueError, match="Invalid config format"):
        load_config(config_file)
