# Agent Guidelines

## Code Style

We follow the Google Python Style Guide with these project-specific rules:

- Use `@override` to mark overridden methods.
- Never use `print(...)` in library code (allowed in tests).
- Define error messages outside the raise statement.
- Don't use implicit string concatenation.

## Testing

See [tests/AGENTS.md](tests/AGENTS.md) for testing conventions and how to run tests.

## Commands

When a command produces large output, redirect it to `./.agents-output/<filename>` and inspect selectively to keep context smaller.

## Commits

Use conventional Commits in format: <type>(<scope>): <description>

- Subject: lowercase start, imperative mood, no period, under 72 chars.
- Types: feat, fix, docs, style, refactor, test, perf, ci, build, chore
- Scopes: autoquant, orchestration, export, ci, docs

Examples: feat(export): introduce export pipeline registry, fix(autoquant): resolve inherited method ownership


Do not add a `Co-authored-by` trailer to commits.

## Security

Never commit secrets, credentials, or company-confidential details; scan your diff before staging and exclude anything sensitive when in doubt.

## Verification

Before considering work done, run:

```bash
./scripts/verify --lint --format --mypy
```

