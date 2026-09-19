"""AST contracts for the class-only NLP service layer."""

import ast
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[2]
NLP_ROOT = AI_ROOT / "src" / "nlp"

# Pure-data modules carry no service class by construction.
_EXEMPT_MODULE_NAMES = {"constants.py"}


def _concrete_modules() -> list[Path]:
    return sorted(
        path
        for path in NLP_ROOT.rglob("*.py")
        if path.name != "__init__.py" and path.name not in _EXEMPT_MODULE_NAMES
    )


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_nlp_modules_expose_one_public_class_and_no_module_functions() -> None:
    concrete = _concrete_modules()
    assert len(concrete) >= 20

    for path in concrete:
        tree = _parse(path)
        public_classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
        ]
        module_functions = [
            node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert len(public_classes) == 1, path
        assert module_functions == [], path


def test_nlp_modules_do_not_contain_cli_concerns() -> None:
    concrete = _concrete_modules()
    assert len(concrete) >= 20

    for path in concrete:
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    alias.name != "argparse" and not alias.name.startswith("scripts.")
                    for alias in node.names
                ), path
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module != "argparse", path
                assert module != "scripts" and not module.startswith("scripts."), path
            elif isinstance(node, ast.Attribute):
                assert not (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "sys"
                    and node.attr == "argv"
                ), path
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.name != "main", path
