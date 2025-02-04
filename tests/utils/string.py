import difflib
import textwrap


def dedent_strip(*txt: str) -> tuple[str, ...]:
    """Returns dedented then stripped version of input string(s)."""
    return tuple(textwrap.dedent(t).strip() for t in txt)


def assert_strings_match_verbose(str1: str, str2: str) -> None:
    if not str1 == str2:
        output = "\n".join(difflib.unified_diff(str2.splitlines(), str1.splitlines()))
        raise AssertionError(f"Transformed module does not match expected output:\n{output}")
