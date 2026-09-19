"""Executable clean-architecture boundaries for the MAT restructure.

These package and dependency rules are the contract every relocation task
must satisfy. They are AST-based (never regex) so the real import graph,
including relative imports, is what gets checked.

Run scoped:

    cd ai && uv run --locked pytest tests/test_boundaries.py -q
"""

import ast
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

AI_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = AI_ROOT / "src"
SCRIPTS_ROOT = AI_ROOT / "scripts"
ROOT_ENTRY_POINTS = (AI_ROOT / "main.py", AI_ROOT / "config.py")

EXPECTED_PACKAGES = {"domain", "nlp", "api", "core", "libs"}

OUTER_LAYERS = ("src.api", "src.nlp")
CONCRETE_ML_ROOTS = frozenset({"sklearn", "joblib"})

# Import roots the final production tree must not contain anywhere.
LEGACY_IMPORT_ROOTS = frozenset(
    {"mat_ai", "adapter", "contracts", "modeling", "evaluation", "data", "utils", "api"}
)

# One primary non-DTO class per OO module (SRP). Related DTO/value-object
# bundles and the verbatim-relocated legacy selection module are explicitly
# exempt; modules without classes (pure functions) are exempt by construction.
SINGLE_PRIMARY_CLASS_MODULES = (
    SRC_ROOT / "domain" / "contracts.py",
    SRC_ROOT / "nlp" / "modeling" / "artifact_model.py",
)
MIDDLEWARES_DIR = SRC_ROOT / "api" / "middlewares"
CLASS_BUNDLE_EXEMPT = frozenset({"entities.py", "schemas.py", "selection.py"})


def _containing_package(path: Path) -> str:
    """Dotted ``src.*`` package that contains ``path`` (``""`` outside ``src/``)."""
    try:
        rel = path.relative_to(SRC_ROOT)
    except ValueError:
        return ""
    parts = list(rel.parts[:-1])
    return ".".join(["src", *parts]) if parts else "src"


def _production_files() -> list[Path]:
    """Every non-test Python file: ``src/**``, root entry points, and ``scripts/**``."""
    files = [f for f in sorted(SRC_ROOT.rglob("*.py")) if "__pycache__" not in f.parts]
    files.extend(p for p in ROOT_ENTRY_POINTS if p.is_file())
    if SCRIPTS_ROOT.is_dir():
        files.extend(f for f in sorted(SCRIPTS_ROOT.rglob("*.py")) if "__pycache__" not in f.parts)
    return files


def _resolved_imports(tree: ast.Module, package: str) -> Iterator[str]:
    """Yield every module reference in ``tree`` with relative imports resolved."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".") if package else []
                if node.level - 1 > len(parts):
                    continue  # escapes the known tree; nothing to resolve
                anchor = parts[: len(parts) - (node.level - 1)]
                base = ".".join([*anchor, *(base.split(".") if base else [])])
            if base == "src":
                for alias in node.names:
                    yield f"src.{alias.name}"
            elif base:
                yield base


def _imports_under(root: Path) -> Iterator[tuple[Path, str]]:
    """Yield ``(file, module)`` for every import below ``root``."""
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _resolved_imports(tree, _containing_package(path)):
            yield path, module


def test_every_expected_package_exists_with_init() -> None:
    missing = [
        package
        for package in sorted(EXPECTED_PACKAGES)
        if not (SRC_ROOT / package / "__init__.py").is_file()
    ]
    assert not missing, f"expected target packages missing __init__.py: {missing}"


def test_src_ships_py_typed() -> None:
    assert (SRC_ROOT / "py.typed").is_file(), "src/py.typed marker is required"


def test_domain_and_core_never_import_outer_layers() -> None:
    for layer in ("domain", "core"):
        violations = [
            f"{path.relative_to(SRC_ROOT)} imports {module}"
            for path, module in _imports_under(SRC_ROOT / layer)
            if module.startswith(OUTER_LAYERS)
        ]
        assert not violations, f"{layer}/ must not import api or nlp: {violations}"


def test_domain_stays_free_of_yaml_filesystem_and_regex() -> None:
    banned_roots = frozenset({"yaml", "re", "pathlib", "os", "io", "glob", "shutil", "json"})
    violations = [
        f"{path.relative_to(SRC_ROOT)} imports {module}"
        for path, module in _imports_under(SRC_ROOT / "domain")
        if module.split(".")[0] in banned_roots
    ]
    assert not violations, f"domain/ must not import yaml/filesystem/regex: {violations}"


def test_api_layer_never_imports_concrete_ml_implementations() -> None:
    violations = [
        f"{path.relative_to(SRC_ROOT)} imports {module}"
        for path, module in _imports_under(SRC_ROOT / "api" / "v1" / "endpoints")
        if module.split(".")[0] in CONCRETE_ML_ROOTS
    ]
    assert not violations, f"endpoints/ must stay abstract over modeling: {violations}"


def test_libs_result_stays_self_contained() -> None:
    result_module = SRC_ROOT / "libs" / "result.py"
    assert result_module.is_file(), "src/libs/result.py is a required target module"
    allowed = set(sys.stdlib_module_names) | {"result"}
    violations = []
    for path, module in _imports_under(SRC_ROOT / "libs"):
        if module.split(".")[0] in allowed or module.startswith("src.libs"):
            continue
        violations.append(f"{path.relative_to(SRC_ROOT)} imports {module}")
    assert not violations, f"libs/ may import stdlib and result only: {violations}"


def test_no_legacy_import_roots_in_production_code() -> None:
    violations = []
    for path in _production_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _resolved_imports(tree, _containing_package(path)):
            if module.split(".")[0] in LEGACY_IMPORT_ROOTS:
                violations.append(f"{path.relative_to(AI_ROOT)} imports {module}")
    assert not violations, f"legacy import roots must disappear: {violations}"


def _is_dto_class(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        if "dataclass" in ast.unparse(decorator):
            return True
    marker_names = {"BaseModel", "Enum", "StrEnum"}
    return any(
        (isinstance(base, ast.Name) and base.id in marker_names)
        or (isinstance(base, ast.Attribute) and base.attr in marker_names)
        for base in node.bases
    )


def _is_failure_class(node: ast.ClassDef) -> bool:
    return any(
        (isinstance(base, ast.Name) and (base.id == "Exception" or base.id.endswith("Error")))
        or (isinstance(base, ast.Attribute) and base.attr.endswith("Error"))
        for base in node.bases
    )


def _primary_classes(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and not _is_dto_class(node)
        and not _is_failure_class(node)
    ]


def test_oo_modules_hold_one_primary_class() -> None:
    """Module creation itself is gated by each relocation task's RED tests."""
    targets = [p for p in SINGLE_PRIMARY_CLASS_MODULES if p.is_file()]
    if MIDDLEWARES_DIR.is_dir():
        targets.extend(sorted(MIDDLEWARES_DIR.glob("*.py")))
    violations = []
    for module_path in targets:
        if module_path.name in CLASS_BUNDLE_EXEMPT:
            continue
        primary = _primary_classes(module_path)
        if len(primary) > 1:
            violations.append(f"{module_path.relative_to(AI_ROOT)}: {primary}")
    assert not violations, f"at most one primary non-DTO class per OO module: {violations}"


def test_modeling_never_imports_evaluation() -> None:
    """The modeling -> evaluation dependency cycle stays broken.

    One edge is sanctioned by the restructure plan: the shared challenge-case
    loader lives in ``src.nlp.evaluation.behavioral`` and the transformer
    adapter in modeling consumes it. Any other modeling -> evaluation import
    is a violation.
    """
    sanctioned = SRC_ROOT / "nlp" / "modeling" / "transformer.py"
    violations = []
    for path, module in _imports_under(SRC_ROOT / "nlp" / "modeling"):
        if not module.startswith("src.nlp.evaluation"):
            continue
        if path == sanctioned and module == "src.nlp.evaluation.behavioral":
            continue
        violations.append(f"{path.relative_to(AI_ROOT)} imports {module}")
    assert not violations, f"modeling/ must not import evaluation: {violations}"


def test_boundary_scanner_sees_real_modules() -> None:
    """Vacuity guard: the AST scanner must resolve real module imports.

    Ported from the legacy ``mat_ai`` walker characterization test: if the
    scanner stopped seeing files it makes rules about, every boundary rule
    above would pass vacuously.
    """
    imports: dict[Path, set[str]] = {}
    for path, module in _imports_under(SRC_ROOT / "nlp"):
        imports.setdefault(path, set()).add(module)
    calibration = SRC_ROOT / "nlp" / "modeling" / "calibration.py"
    metrics = SRC_ROOT / "nlp" / "evaluation" / "metrics.py"
    assert calibration in imports and metrics in imports
    assert "src.domain.entities" in imports[calibration]
    assert "src.domain.scoring" in imports[calibration]
    assert "src.nlp.modeling.baseline" in imports[calibration]
    assert "src.domain.training" in imports[calibration]
    assert "src.domain.entities" in imports[metrics]
    assert "src.domain.scoring" in imports[metrics]


ENDPOINT_BANNED_ROOTS = frozenset(
    {
        # Filesystem and config-parsing roots: routes read runtime state only.
        "pathlib",
        "os",
        "io",
        "glob",
        "shutil",
        "yaml",
        # Model implementations and their serialization formats.
        "joblib",
        "sklearn",
        "pandas",
        "numpy",
        "torch",
        "transformers",
        "huggingface_hub",
    }
)
ENDPOINT_ALLOWED_SRC_PREFIXES = ("src.api", "src.domain", "src.libs")
# The Result services own the one sanctioned ``src.nlp`` touchpoint
# (``src.nlp.modeling.limits``); endpoint modules import none.


def test_endpoint_modules_stay_free_of_filesystem_yaml_and_model_imports() -> None:
    """Route modules compose over injected Result services; they never touch storage.

    Filesystem/YAML roots and concrete model implementations stay behind the
    composition layer, so a drifted endpoint cannot grow secret file access
    or a direct model dependency. Endpoint modules import no ``src.nlp``
    module at all: token accounting lives behind
    ``src.api.services.prediction_service`` and
    ``src.api.services.health_check_service``.
    """
    observed: dict[Path, set[str]] = {}
    violations: list[str] = []
    for path, module in _imports_under(SRC_ROOT / "api" / "v1" / "endpoints"):
        observed.setdefault(path, set()).add(module)
        if module.startswith(ENDPOINT_ALLOWED_SRC_PREFIXES):
            continue
        if module.startswith("src.") or module.split(".")[0] in ENDPOINT_BANNED_ROOTS:
            violations.append(f"{path.name} imports {module}")

    # Vacuity guard: the scan must see the real endpoint modules and imports.
    predict = SRC_ROOT / "api" / "v1" / "endpoints" / "predict.py"
    health = SRC_ROOT / "api" / "v1" / "endpoints" / "health.py"
    assert predict in observed and health in observed, sorted(observed)
    assert "src.api.services.prediction_service" in observed[predict]
    assert "src.api.dependencies" in observed[predict]
    assert "src.api.services.health_check_service" in observed[health]
    assert "src.api.dependencies" in observed[health]
    assert not violations, f"endpoints must stay composition-safe: {violations}"


def test_composition_import_performs_no_artifact_or_catalog_load() -> None:
    """Importing the real composition roots must not load a model or catalog.

    ``main`` wires the production loader into the factory, so a future
    module-level ``load_model_from_artifact(...)`` or catalog activation
    would silently become an import side effect on every worker. The check
    runs in a fresh interpreter so this test process cannot mask it: after
    the import the runtime state must still be completely unloaded.
    """
    probe = "\n".join(
        [
            "import json",
            "import main",
            "state = main.app.state.runtime",
            "print(json.dumps({",
            "    'model_loaded': state.model is not None,",
            "    'ready': state.ready,",
            "    'degraded': state.degraded,",
            "    'backend_declared': state.backend is not None,",
            "    'startup_error': state.startup_error is not None,",
            "    'catalog_activated': state.normalization is not None,",
            "}))",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=AI_ROOT,
        env={**os.environ, "PYTHONPATH": str(AI_ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, f"importing main failed: {result.stderr}"
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed == {
        "model_loaded": False,
        "ready": False,
        "degraded": False,
        "backend_declared": False,
        "startup_error": False,
        "catalog_activated": False,
    }, f"composition import loaded runtime state: {observed}"


APPROVED_OUTPUT_ROOTS = frozenset({"data", "configs", "runs", "artifacts", "reports"})
FROZEN_CORPUS_FILES = frozenset(
    {
        "data/dataset.csv",
        "data/split_manifest.json",
        "data/challenge_set.csv",
        "data/vietnamese_demo_cases.json",
    }
)
_WRITE_RECEIVER_METHODS = frozenset(
    {"write_text", "write_bytes", "mkdir", "to_csv", "to_json", "to_parquet"}
)
_WRITE_MODE_CHARS = frozenset("wax")


def _open_write_target(call: ast.Call, builtin: bool) -> ast.expr | None:
    """Target of ``open(...)``/``Path.open(...)`` when the mode writes."""
    mode = ""
    mode_index = 1 if builtin else 0
    if (
        len(call.args) > mode_index
        and isinstance(call.args[mode_index], ast.Constant)
        and isinstance(call.args[mode_index].value, str)
    ):
        mode = call.args[mode_index].value
    for keyword in call.keywords:
        if (
            keyword.arg == "mode"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        ):
            mode = keyword.value.value
    if not _WRITE_MODE_CHARS.intersection(mode):
        return None
    if builtin:
        return call.args[0]
    assert isinstance(call.func, ast.Attribute)
    return call.func.value


def _write_target_of(call: ast.Call) -> ast.expr | None:
    """The filesystem target a write-shaped call writes to, if any."""
    func = call.func
    if isinstance(func, ast.Attribute):
        if func.attr in _WRITE_RECEIVER_METHODS:
            return func.value
        if func.attr == "open":
            return _open_write_target(call, builtin=False)
        if func.attr == "dump" and len(call.args) >= 2:
            return call.args[1]  # e.g. joblib.dump(payload, path)
        return None
    if isinstance(func, ast.Name) and func.id == "open":
        return _open_write_target(call, builtin=True)
    return None


def _path_literal(target: ast.expr) -> str | None:
    """Compile-time path literal behind a write target (``None`` if derived)."""
    if isinstance(target, ast.BinOp):  # Path("reports") / "file.md"
        return _path_literal(target.left)
    if (
        isinstance(target, ast.Call)
        and isinstance(target.func, ast.Name)
        and target.func.id == "Path"
        and len(target.args) == 1
        and isinstance(target.args[0], ast.Constant)
        and isinstance(target.args[0].value, str)
    ):
        return target.args[0].value
    if isinstance(target, ast.Constant) and isinstance(target.value, str):
        return target.value
    return None


def test_production_writers_resolve_only_approved_output_roots() -> None:
    """Write-path protection for every production CLI writer.

    A write target must come from the writer's own output parameters (the
    ``--output``/``--artifact-dir`` arguments) or from an approved root.
    Hardcoded destinations outside ``data/configs/runs/artifacts/reports``
    are violations, and the frozen corpus files may never appear as a write
    destination at all. Vacuity-guarded: the scan must keep seeing the real
    production write sites, or the rule above would pass vacuously.
    """
    sinks: dict[Path, list[str]] = {}
    violations: list[str] = []
    for path in _production_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = _write_target_of(node)
            if target is None:
                continue
            sinks.setdefault(path, []).append(ast.unparse(target))
            literal = _path_literal(target)
            if literal is None:
                continue  # parameter-derived: args.output, output_path, ...
            if literal in FROZEN_CORPUS_FILES:
                violations.append(
                    f"{path.relative_to(AI_ROOT)} writes a frozen corpus file: {literal}"
                )
            first = PurePosixPath(literal.replace("\\", "/")).parts[:1]
            if not first or first[0] not in APPROVED_OUTPUT_ROOTS:
                violations.append(
                    f"{path.relative_to(AI_ROOT)} hardcodes a write outside the "
                    f"approved roots {sorted(APPROVED_OUTPUT_ROOTS)}: {literal}"
                )

    total = sum(len(targets) for targets in sinks.values())
    assert total >= 20, f"scanner lost sight of production writers: {total} targets"
    for required in (
        SRC_ROOT / "nlp" / "modeling" / "registry.py",  # joblib.dump + manifest write
        SRC_ROOT / "nlp" / "evaluation" / "reporting.py",  # report write_text
        SRC_ROOT / "nlp" / "training" / "dataset.py",  # dataset open("w")
        SCRIPTS_ROOT / "data_split.py",  # split manifest write_text
    ):
        assert required in sinks, f"scanner must see write sites in {required.name}"
    assert not violations, f"write-path protection violated: {violations}"
