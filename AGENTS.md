# AI Agent Guidelines - Ecoute Project

This document provides strict standards for AI agents and contributors working on the **Ecoute** codebase (RTSP -> VAD -> Whisper -> UI).

## 1. Technical Standards
- **Language**: Python 3.10+ with mandatory Type Hinting (`from __future__ import annotations`).
- **Style**: Strict PEP 8. Use descriptive English names.
- **Paths**: Use `pathlib.Path` exclusively. No string path manipulation.
- **Config**: Update `pipeline_config.py` for new options (Hierarchy: Env Vars > YAML > Defaults).
- **Core Stack**: FFmpeg (Capture), Silero (VAD), Faster-Whisper (Inference), SQLite (Storage), Streamlit (UI).

## 2. Testing Policy (Pytest)
- **Requirement**: Every logic change (VAD, Transcription, Storage) **must** include/update a `pytest` in `tests/`.
- **Isolation**: **NEVER** load real AI models (Whisper/Silero). Use `unittest.mock`.
- **Fixtures**: Use `tmp_path` for all filesystem operations.
- **Execution**: All tests must pass via `pytest`.

## 3. Git Conventions (Conventional Commits)
Format: `<type>(<scope>): <description>`
- `test`: Adding/updating tests (mandatory for code changes).
- `feat`: New functionality.
- `fix`: Bug correction.
- `docs`: README, AGENTS.md, or comment updates.
- `refactor`: Code change without behavioral impact.
- **Scopes**: `core`, `vad`, `transcribe`, `ui`, `pipeline`, `storage`, `config`.

## 4. Documentation
- **README.md**: Keep it synchronized with architecture, deployment (Docker Compose), and environment variables.
- **AGENTS.md**: This file is the source of truth for AI instructions.

## 5. Implementation Workflow
1. **Analyze**: Identify impacted modules and existing tests.
2. **Implement**: Code with types, PEP 8, and Pathlib.
3. **Test**: Create/update unit tests. Run `pytest`.
4. **Document**: Update `README.md` for any interface or config change.
5. **Commit**: Provide a title following Conventional Commits (e.g., `test(vad): update segment logic`).
