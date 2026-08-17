"""Checks on the source files themselves rather than on what they do.

These exist because of a real failure: an editor wrote a UTF-8 byte-order mark
onto the front of `cost.py`. Nothing behavioural broke -- every test passed --
but `ruff format --check` refused the file and CI went red, and the cause is
invisible in a diff viewer, which shows a change to a line that looks identical.

A test that names the problem turns twenty minutes of squinting at an unchanged
line into one failure message.
"""

import pathlib

import pytest

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent

UTF8_BOM = b"\xef\xbb\xbf"


def python_sources() -> list[pathlib.Path]:
    paths = [
        *(REPOSITORY_ROOT / "src").rglob("*.py"),
        *(REPOSITORY_ROOT / "tests").rglob("*.py"),
    ]
    return sorted(path for path in paths if "__pycache__" not in path.parts)


def test_there_are_sources_to_check():
    # Guards against the checks below silently passing over an empty list.
    assert len(python_sources()) > 5


@pytest.mark.parametrize("path", python_sources(), ids=lambda path: path.name)
def test_no_source_file_starts_with_a_byte_order_mark(path):
    raw = path.read_bytes()
    assert not raw.startswith(UTF8_BOM), (
        f"{path.relative_to(REPOSITORY_ROOT)} starts with a UTF-8 BOM. "
        "It is invisible in most diffs and makes `ruff format --check` fail. "
        "Strip the first three bytes."
    )


@pytest.mark.parametrize("path", python_sources(), ids=lambda path: path.name)
def test_no_source_file_has_windows_line_endings(path):
    # Same class of problem: invisible, and it fails the format check.
    assert b"\r\n" not in path.read_bytes(), (
        f"{path.relative_to(REPOSITORY_ROOT)} has CRLF line endings"
    )


@pytest.mark.parametrize("path", python_sources(), ids=lambda path: path.name)
def test_every_source_file_decodes_as_utf8(path):
    try:
        path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - only on a broken file
        pytest.fail(f"{path.relative_to(REPOSITORY_ROOT)} is not valid UTF-8: {exc}")
