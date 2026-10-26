import re
from pathlib import Path

import abcopt

ROOT = Path(__file__).resolve().parents[1]


def _read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_version_is_consistent():
    """pyproject.toml, abcopt.__version__ and CITATION.cff must agree before a release."""
    pyproject = re.search(r'^version = "([^"]+)"', _read("pyproject.toml"), re.M).group(1)
    citation = re.search(r'^version: "([^"]+)"', _read("CITATION.cff"), re.M).group(1)
    assert abcopt.__version__ == pyproject == citation


def test_changelog_has_an_entry_for_the_version():
    assert f"## {abcopt.__version__}\n" in _read("CHANGELOG.md")


def test_public_api():
    assert set(abcopt.__all__) == {
        "CanonicalABCSolver", "ABCSearchCV", "CycleState", "EarlyStopping", "__version__",
    }
