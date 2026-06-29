# Python Style

In FastForward we follow the [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)
with the following exceptions, modifications and additions:

- We do not require module level docstrings.
- We recommend to not use implicit evaluation.
- Follow the formatter using the provided settings instead of Google's styleguide.
- Use `@override` to mark overridden methods.
- Any comment that refers to future work cross-references an issue.
- Don't use implicit sting concatenation.
- Never use `print(...)` in library code. It is allowed in tests.
- Prefer to define an error message outside of the raise statement:

  ```python
  # Discouraged
  raise TypeError(f"a long error message with {string_interpolation}")

  # Encouraged
  msg = f"a long error message with {string_interpolation}"
  raise TypeError(msg)
  ```

- Try to use `GIVEN/WHEN/THEN` labels in tests. See
  [this blog](https://martinfowler.com/bliki/GivenWhenThen.html) for more info.

# Commit Messages

All commit messages **must** follow the [Conventional Commits](https://www.conventionalcommits.org/)
specification. Every commit message must have the form:

```
<type>(<optional scope>): <description>

<optional body>

<optional footer(s)>
```

## Rules

- **Type** is required and must be one of the allowed types listed below.
- **Scope** is required and must be one of the allowed scopes listed below.
- **Subject** starts with a lowercase letter, uses imperative mood, and does not end with
  a period.
- Keep the first line under 72 characters.
- An optional body and footer may follow a blank line for additional context.

## Allowed Scopes

| Scope            | Area of the codebase                          |
|------------------|-----------------------------------------------|
| `autoquant`      | Automatic quantization                        |
| `orchestration`  | Orchestration and tracing                     |
| `algorithms`     | Quantization algorithms (e.g. GPTQ)           |
| `export`         | Model export pipelines                        |
| `ci`             | CI configuration and scripts                  |
| `docs`           | Documentation                                 |

## Allowed Types

| Type       | Purpose                                              |
|------------|------------------------------------------------------|
| `feat`     | A new feature                                        |
| `fix`      | A bug fix                                            |
| `docs`     | Documentation-only changes                           |
| `style`    | Formatting, missing semicolons, etc. (no code change)|
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test`     | Adding or updating tests                             |
| `perf`     | A performance improvement                            |
| `ci`       | Changes to CI configuration or scripts               |
| `build`    | Changes to the build system or dependencies          |
| `chore`    | Other changes that don't modify src or test files    |

## Examples

```
feat(export): introduce export pipeline registry
fix(autoquant): resolve inherited method ownership for correct forward generation
test(autoquant): add SAM3-inspired end-to-end fixture scaffold
perf(orchestration): reduce graph traversal allocations
ci(docker): enforce weekly rebuilds
docs(export): fix typo in pipeline docstring
```
