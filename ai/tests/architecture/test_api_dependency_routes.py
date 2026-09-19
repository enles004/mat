"""AST contracts for dependency-injected module-level API routes."""

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[2]
ENDPOINT_ROOT = AI_ROOT / "src" / "api" / "v1" / "endpoints"


def _parse(filename: str) -> ast.Module:
    path = ENDPOINT_ROOT / filename
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@dataclass(frozen=True)
class RouterSymbols:
    factory_names: frozenset[str]
    fastapi_module_names: frozenset[str]
    imported_module_names: frozenset[str]
    imported_router_names: frozenset[str]


def _bound_name(alias: ast.alias) -> str:
    return alias.asname or alias.name.split(".")[0]


def _is_fastapi_module(module: str) -> bool:
    return module == "fastapi" or module.startswith("fastapi.")


def _router_symbols(tree: ast.Module) -> RouterSymbols:
    factory_names = {"APIRouter"}
    fastapi_module_names: set[str] = set()
    imported_module_names: set[str] = set()
    imported_router_names: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = _bound_name(alias)
                imported_module_names.add(bound_name)
                if _is_fastapi_module(alias.name):
                    fastapi_module_names.add(bound_name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound_name = alias.asname or alias.name
                imported_module_names.add(bound_name)
                if _is_fastapi_module(node.module or "") and alias.name == "APIRouter":
                    factory_names.add(bound_name)
                if node.module == "fastapi" and alias.name == "routing":
                    fastapi_module_names.add(bound_name)
                if (
                    alias.name == "router"
                    or alias.name.endswith("router")
                    or bound_name.endswith("router")
                ):
                    imported_router_names.add(bound_name)

    return RouterSymbols(
        factory_names=frozenset(factory_names),
        fastapi_module_names=frozenset(fastapi_module_names),
        imported_module_names=frozenset(imported_module_names),
        imported_router_names=frozenset(imported_router_names),
    )


def _attribute_parts(node: ast.expr) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    return (current.id, *reversed(parts))


def _is_api_router_call(node: ast.AST | None, symbols: RouterSymbols) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in symbols.factory_names
    parts = _attribute_parts(node.func)
    return (
        parts is not None
        and len(parts) >= 2
        and parts[-1] == "APIRouter"
        and parts[0] in symbols.fastapi_module_names
    )


def _module_router_assignments(tree: ast.Module, symbols: RouterSymbols) -> list[ast.AST]:
    assignments: list[ast.AST] = []
    for node in tree.body:
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        if _is_api_router_call(value, symbols):
            assignments.append(node)
    return assignments


def _assigns_router(node: ast.AST) -> bool:
    if isinstance(node, ast.Assign):
        return any(
            isinstance(target, ast.Name) and target.id == "router" for target in node.targets
        )
    return (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "router"
    )


def _router_variable_names(tree: ast.Module, symbols: RouterSymbols) -> frozenset[str]:
    router_names = {"router", *symbols.imported_router_names}
    for node in tree.body:
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        if not _is_api_router_call(value, symbols):
            continue
        if isinstance(node, ast.Assign):
            router_names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            router_names.add(node.target.id)
    return frozenset(router_names)


def _is_relevant_router_object(
    node: ast.expr, router_names: frozenset[str], symbols: RouterSymbols
) -> bool:
    if isinstance(node, ast.Name):
        return node.id in router_names
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "router"
        and isinstance(node.value, ast.Name)
        and node.value.id in symbols.imported_module_names
    )


ROUTE_REGISTRATION_METHODS = frozenset(
    {
        "add_api_route",
        "api_route",
        "delete",
        "get",
        "head",
        "options",
        "patch",
        "post",
        "put",
        "websocket",
    }
)


def _route_path(call: ast.Call) -> str | None:
    candidates: list[ast.expr] = list(call.args[:1])
    candidates.extend(keyword.value for keyword in call.keywords if keyword.arg == "path")
    for candidate in candidates:
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
            return candidate.value
    return None


def _registered_route_paths(tree: ast.Module, symbols: RouterSymbols) -> set[str]:
    router_names = _router_variable_names(tree, symbols)
    paths: set[str] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ROUTE_REGISTRATION_METHODS
            and _is_relevant_router_object(node.func.value, router_names, symbols)
        ):
            continue
        path = _route_path(node)
        if path is not None:
            paths.add(path)
    return paths


def _endpoint_path(decorator: ast.expr) -> str | None:
    if not (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and isinstance(decorator.func.value, ast.Name)
        and decorator.func.value.id == "router"
        and decorator.args
        and isinstance(decorator.args[0], ast.Constant)
        and isinstance(decorator.args[0].value, str)
    ):
        return None
    return decorator.args[0].value


def _endpoint_functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    endpoints: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            path = _endpoint_path(decorator)
            if path is not None:
                endpoints[path] = node
    return endpoints


def _parameters(function: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterable[ast.arg]:
    arguments = function.args
    return (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)


def _parameter_uses_depends(parameter: ast.arg) -> bool:
    return (
        isinstance(parameter.annotation, ast.Call)
        and isinstance(parameter.annotation.func, ast.Name)
        and parameter.annotation.func.id == "Depends"
    )


def _has_depends_parameter(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    defaults = (*function.args.defaults, *function.args.kw_defaults)
    return any(
        isinstance(default, ast.Call)
        and isinstance(default.func, ast.Name)
        and default.func.id == "Depends"
        for default in defaults
    ) or any(_parameter_uses_depends(parameter) for parameter in _parameters(function))


def test_endpoint_modules_expose_one_module_level_router_without_factories() -> None:
    for filename in ("predict.py", "health.py"):
        tree = _parse(filename)
        symbols = _router_symbols(tree)
        router_assignments = _module_router_assignments(tree, symbols)
        router_calls = [node for node in ast.walk(tree) if _is_api_router_call(node, symbols)]

        assert len(router_assignments) == 1, filename
        assert _assigns_router(router_assignments[0]), filename
        assert len(router_calls) == 1, filename
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("create_")
            and node.name.endswith("_router")
            for node in ast.walk(tree)
        ), filename
        assert all(
            node in tree.body
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(_endpoint_path(decorator) is not None for decorator in node.decorator_list)
        ), filename


def test_endpoint_routes_and_dependencies_follow_the_public_contract() -> None:
    predict_tree = _parse("predict.py")
    health_tree = _parse("health.py")
    predict_routes = _endpoint_functions(predict_tree)
    health_routes = _endpoint_functions(health_tree)

    assert "/predict" in predict_routes
    assert "/livez" in health_routes
    assert "/health-check" in health_routes
    assert "/readyz" not in _registered_route_paths(predict_tree, _router_symbols(predict_tree))
    assert "/readyz" not in _registered_route_paths(health_tree, _router_symbols(health_tree))
    assert _has_depends_parameter(predict_routes["/predict"])
    assert _has_depends_parameter(health_routes["/health-check"])


def test_ast_helpers_detect_qualified_and_aliased_router_construction() -> None:
    qualified_tree = ast.parse(
        "import fastapi as api\n"
        "router = api.APIRouter()\n"
        "def create_hidden_router():\n"
        "    return api.APIRouter()\n",
        filename="qualified.py",
    )
    qualified_symbols = _router_symbols(qualified_tree)
    qualified_assignments = _module_router_assignments(qualified_tree, qualified_symbols)
    qualified_calls = [
        node for node in ast.walk(qualified_tree) if _is_api_router_call(node, qualified_symbols)
    ]

    assert len(qualified_assignments) == 1
    assert _assigns_router(qualified_assignments[0])
    assert len(qualified_calls) == 2

    aliased_tree = ast.parse(
        "from fastapi import APIRouter, APIRouter as Router\n"
        "router = APIRouter()\n"
        "def create_hidden_router():\n"
        "    return Router()\n",
        filename="aliased.py",
    )
    aliased_symbols = _router_symbols(aliased_tree)
    aliased_assignments = _module_router_assignments(aliased_tree, aliased_symbols)
    aliased_calls = [
        node for node in ast.walk(aliased_tree) if _is_api_router_call(node, aliased_symbols)
    ]

    assert len(aliased_assignments) == 1
    assert len(aliased_calls) == 2


def test_ast_helpers_detect_fastapi_submodule_aliases_and_attribute_chains() -> None:
    submodule_alias_tree = ast.parse(
        "from fastapi import routing as api\n"
        "router = api.APIRouter()\n"
        "def create_hidden_router():\n"
        "    return api.APIRouter()\n",
        filename="submodule_alias.py",
    )
    attribute_chain_tree = ast.parse(
        "import fastapi\n"
        "router = fastapi.routing.APIRouter()\n"
        "def create_hidden_router():\n"
        "    return fastapi.routing.APIRouter()\n",
        filename="attribute_chain.py",
    )

    for tree in (submodule_alias_tree, attribute_chain_tree):
        symbols = _router_symbols(tree)
        assignments = _module_router_assignments(tree, symbols)
        calls = [node for node in ast.walk(tree) if _is_api_router_call(node, symbols)]

        assert len(assignments) == 1
        assert _assigns_router(assignments[0])
        assert len(calls) == 2


def test_ast_helpers_detect_readyz_registered_on_local_and_imported_routers() -> None:
    local_tree = ast.parse(
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        "router.add_api_route(path='/readyz', endpoint=lambda: None)\n",
        filename="local.py",
    )
    imported_tree = ast.parse(
        "from src.api.v1.endpoints.other import health_router as endpoints\n"
        "@endpoints.get('/readyz')\n"
        "def readyz():\n"
        "    return {}\n",
        filename="imported.py",
    )
    module_tree = ast.parse(
        "import src.api.v1.endpoints.other as other\n"
        "other.router.add_api_route('/readyz', lambda: None)\n",
        filename="module.py",
    )

    assert "/readyz" in _registered_route_paths(local_tree, _router_symbols(local_tree))
    assert "/readyz" in _registered_route_paths(imported_tree, _router_symbols(imported_tree))
    assert "/readyz" in _registered_route_paths(module_tree, _router_symbols(module_tree))
