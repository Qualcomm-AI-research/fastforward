# Testing Guidelines

Every new function or feature must ship with tests: at least the happy path, plus error/edge cases where behavior is non-obvious.

## Style

Before writing a new test, read at least two existing tests in the same area for style and fixture conventions.

Pytest-style functions, no test classes unless shared state genuinely requires one.

Structure tests with inline `# GIVEN` / `# WHEN` / `# THEN` comments.

## Running Tests

Tests marked `@pytest.mark.slow` are skipped by default. Run them with `--include-slow` when your change touches an area covered by slow tests:

```bash
python3 -m pytest --include-slow tests/path/to/relevant_tests.py
```

Tests marked `@pytest.mark.benchmark` are also skipped by default and are not
included by `--include-slow`. Run them explicitly with `--include-benchmark`:

```bash
python3 -m pytest --include-benchmark tests/path/to/relevant_tests.py
```
