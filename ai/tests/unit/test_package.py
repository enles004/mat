"""The installable application package must expose its semantic version."""

import importlib


def test_package_has_semantic_version() -> None:
    src = importlib.import_module("src")
    assert src.__version__ == "0.1.0"
