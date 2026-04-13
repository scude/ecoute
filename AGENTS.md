# Ecoute - AGENTS.md (agent operating contract)

## Normative keywords
This file uses RFC 2119 keywords: MUST, MUST NOT, SHOULD. In case of conflict, MUST/MUST NOT win.

## Scope
- This file defines repo-level rules for AI coding agents and contributors.
- Goal: make changes that are verifiable (tests), minimal, and consistent with existing architecture.

## Repository map
- ecoute/   : core logic (VAD, Transcription, Storage, Config)
- ui/       : Streamlit UI
- scripts/  : shell helpers (RTSP capture, RNNoise)
- tests/    : pytest unit tests

## Non‑negotiable safety & determinism
- MUST NOT load real AI models in tests or CI. Use mocks/fakes.
- MUST keep tests offline & deterministic (no downloads, no network, no GPU assumptions).
- MUST use tmp_path (pytest) for filesystem I/O in tests. Never write to real user paths.

## Python & code standards
- MUST target Python 3.10+.
- MUST use type hints on all new/modified public functions and non-trivial internals.
- MUST follow PEP 8 (keep code readable and consistent with existing style).
- MUST use pathlib.Path for all path handling (no os.path, no bare strings for paths).

## Imports & module boundaries
- Inside ecoute/, MUST use package-relative imports (e.g., `from .storage import ...`).
- MUST respect the existing package structure; do not introduce new top-level packages unless requested.

## Configuration changes
- Any new pipeline option MUST be added/updated in `ecoute/pipeline_config.py`.
- README.md MUST be updated for any user-facing interface/config change.

## Testing policy (pytest)
- Any change to logic MUST include/update a pytest in `tests/`.
- Tests MUST be unit-level and isolated (mock model loading, external services, and hardware).
- All tests MUST pass with: `pytest`

## Git conventions (Conventional Commits)
Commit title format MUST be: `<type>(<scope>): <description>`

Types:
- feat, fix, test, docs, refactor

Scopes (examples):
- core, vad, transcribe, ui, pipeline, storage, config

## Implementation workflow (mandatory)
1) Analyze: identify impacted modules + existing tests.
2) Implement: minimal diff, typed code, pathlib.
3) Test: add/update pytest(s); run `pytest` and ensure green.
4) Document: update README.md if interface/config changed.
5) Commit: use Conventional Commits title.

## Output expectation (when acting as an agent)
- Summarize what changed and why.
- List the exact tests executed (command + result).
- If something cannot be verified (missing fixture, unclear behavior), state it explicitly.
