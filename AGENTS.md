# AI Agent Guidelines - Ecoute Project

This document provides strict standards for AI agents and contributors working on the **Ecoute** codebase.

## 1. Project Structure
- `ecoute/` : Core logic (VAD, Transcription, Storage, Config).
- `ui/` : Streamlit user interface.
- `scripts/` : Shell scripts (Capture RTSP, RNNoise).
- `tests/` : Unit tests.

## 2. Technical Standards
- **Language**: Python 3.10+ with mandatory Type Hinting.
- **Style**: Strict PEP 8.
- **Paths**: Use `pathlib.Path` exclusively.
- **Config**: Update `ecoute/pipeline_config.py` for new options.
- **Imports**: Use package-relative imports (e.g., `from .storage import ...`) within `ecoute/`.

## 3. Testing Policy (Pytest)
- **Requirement**: Every logic change **must** include/update a `pytest` in `tests/`.
- **Isolation**: **NEVER** load real AI models. Use mocks.
- **Fixtures**: Use `tmp_path` for all filesystem operations.
- **Execution**: All tests must pass via `pytest`.

## 4. Git Conventions (Conventional Commits)
Format: `<type>(<scope>): <description>`
- `test`: Adding/updating tests.
- `feat`: New functionality.
- `fix`: Bug correction.
- `docs`: README, AGENTS.md, or comment updates.
- `refactor`: Code change without behavioral impact.
- **Scopes**: `core`, `vad`, `transcribe`, `ui`, `pipeline`, `storage`, `config`.

## 5. Documentation
- **README.md**: Keep it synchronized with architecture and deployment.
- **AGENTS.md**: This file is the source of truth for AI instructions.

## 6. Implementation Workflow
1. **Analyze**: Identify impacted modules and existing tests.
2. **Implement**: Code with types, PEP 8, and Pathlib.
3. **Test**: Create/update unit tests. Run `pytest`.
4. **Document**: Update `README.md` for any interface or config change.
5. **Commit**: Provide a title following Conventional Commits.
