# Repository Guidelines

## Project Structure & Module Organization

This Python 3.10+ package uses a `src/` layout. Metrics live in `src/chunking_metrics/`; export public APIs from `__init__.py`. Keep numerical helpers in `utils.py` and input preparation in `preparation.py`. Put tests in `tests/`, experiments in `scripts/`, and reference material in `docs/`.

## Agent Working Style

Work directly in the current checkout. Do not run Git commands, create branches or worktrees, commit, or manage pull requests. Plans may be used in conversation, but never save separate plan, specification, handoff, or process files. Avoid specialized workflows, extra scaffolding, dependencies, and unrelated refactors unless explicitly requested. Implement the smallest complete change, update relevant tests, and run focused checks. Ask only when a missing decision materially affects the result.

## Build, Test, and Development Commands

- `uv sync --dev` installs the package and locked development dependencies.
- `uv run pytest` runs the configured test suite under `tests/` with strict markers and live INFO logging.
- `uv run pytest tests/test_metrics.py -k cohesion` runs a focused subset while developing.
- `uv run ruff check .` checks formatting-independent style, imports, and common correctness issues.
- `uv run ruff format --check .` verifies formatting; use `uv run ruff format .` to apply it.
- `uv build` creates source and wheel distributions through the `uv_build` backend.

## Coding Style & Naming Conventions

Use four-space indentation, a 100-character line limit, and Python 3.10-compatible syntax. Ruff enforces `E`, `F`, `I`, `UP`, `B`, `SIM`, and `RUF`. Use `snake_case` for functions and modules and `PascalCase` for classes. Type public functions and document metric inputs, shapes, and returns. Prefer NumPy vector operations and explicit input validation.

## Testing Guidelines

Tests use pytest and NumPy. Name files `test_*.py` and functions `test_<behavior>`. Cover normal calculations plus empty inputs, invalid dimensions, shape mismatches, and numerical edge cases. Use `pytest.approx` for floating-point results. Add tests alongside every new metric or behavior change; no coverage threshold is currently configured.

## Commit & Pull Request Guidelines

For human contributors, recent history uses short subjects such as `ADD contextual_coherence`, `UPD ...`, or `chore: ...`. Keep commits focused. Pull requests should explain the change, validation performed, linked issues, and any API or dependency impact.

## Security & Configuration

Do not commit virtual environments, generated distributions, credentials, or local datasets. Update both `pyproject.toml` and `uv.lock` when dependencies change.
