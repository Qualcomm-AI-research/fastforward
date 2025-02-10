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
