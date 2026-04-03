from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")

DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "input_audio_dir": "audios",
        "speech_segments_dir": "speech_segments",
        "processed_audio_dir": "audios_processed",
        "db_path": "transcriptions/transcriptions.sqlite",
        "json_output": "transcriptions/transcriptions.json",
    },
    "pipeline": {
        "interval_seconds": 60,
        "max_backoff_seconds": 300,
    },
    "whisper": {
        "model_size_or_path": "medium",
        "device": "cpu",
        "compute_type": "int8",
        "language": "fr",
        "beam_size": 1,
        "best_of": 1,
        "patience": 1.0,
        "condition_on_previous_text": False,
        "temperature": 0.0,
        "compression_ratio_threshold": 1.8,
        "log_prob_threshold": -0.9,
        "no_speech_threshold": 0.3,
    },
    "vad": {
        "sample_rate": 16000,
        "threshold": 0.5,
        "min_speech_duration_ms": 1200,
        "min_silence_duration_ms": 1200,
        "speech_pad_ms": 150,
    },
}

ENV_OVERRIDES = {
    "PIPELINE_INTERVAL_SECONDS": ("pipeline", "interval_seconds", int),
    "PIPELINE_MAX_BACKOFF_SECONDS": ("pipeline", "max_backoff_seconds", int),
    "INPUT_AUDIO_DIR": ("paths", "input_audio_dir", str),
    "SPEECH_SEGMENTS_DIR": ("paths", "speech_segments_dir", str),
    "PROCESSED_AUDIO_DIR": ("paths", "processed_audio_dir", str),
    "TRANSCRIPTIONS_DB_PATH": ("paths", "db_path", str),
    "TRANSCRIPTIONS_JSON_OUTPUT": ("paths", "json_output", str),
    "WHISPER_MODEL_SIZE_OR_PATH": ("whisper", "model_size_or_path", str),
    "WHISPER_DEVICE": ("whisper", "device", str),
    "WHISPER_COMPUTE_TYPE": ("whisper", "compute_type", str),
    "WHISPER_LANGUAGE": ("whisper", "language", str),
    "WHISPER_BEAM_SIZE": ("whisper", "beam_size", int),
    "WHISPER_BEST_OF": ("whisper", "best_of", int),
    "WHISPER_PATIENCE": ("whisper", "patience", float),
    "WHISPER_CONDITION_ON_PREVIOUS_TEXT": (
        "whisper",
        "condition_on_previous_text",
        lambda v: v.lower() in {"1", "true", "yes", "on"},
    ),
    "WHISPER_TEMPERATURE": ("whisper", "temperature", float),
    "WHISPER_COMPRESSION_RATIO_THRESHOLD": (
        "whisper",
        "compression_ratio_threshold",
        float,
    ),
    "WHISPER_LOG_PROB_THRESHOLD": ("whisper", "log_prob_threshold", float),
    "WHISPER_NO_SPEECH_THRESHOLD": ("whisper", "no_speech_threshold", float),
    "VAD_SAMPLE_RATE": ("vad", "sample_rate", int),
    "VAD_THRESHOLD": ("vad", "threshold", float),
    "VAD_MIN_SPEECH_DURATION_MS": ("vad", "min_speech_duration_ms", int),
    "VAD_MIN_SILENCE_DURATION_MS": ("vad", "min_silence_duration_ms", int),
    "VAD_SPEECH_PAD_MS": ("vad", "speech_pad_ms", int),
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)

    if config_path.exists():
        file_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(file_data, dict):
            raise ValueError(f"Invalid config format in {config_path}")
        config = _deep_merge(config, file_data)

    for env_var, (section, key, caster) in ENV_OVERRIDES.items():
        if env_var not in os.environ:
            continue
        raw_value = os.environ[env_var]
        config[section][key] = caster(raw_value)

    return config
