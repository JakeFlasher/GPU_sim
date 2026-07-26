"""Closed registry for sources that discover or directly execute NCU.

Registration is accounting, not authorization. Campaign admission remains owned by
its preflight, descriptor, receipt, and DONE contracts; diagnostic, calibration,
historical, and retired entries cannot mint campaign evidence merely by appearing
here.

Run:  .venv/bin/python -m pytest eval/guards/test_ncu_invoker_registry.py -q
"""
from __future__ import annotations

import ast
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_PREFIXES = (
    Path(".claude"),
    Path(".git"),
    Path(".humanize"),
    Path(".toolchain"),
    Path(".venv"),
    Path("IISWC_conference_latex"),
    Path("data"),
    Path("references"),
    Path("results"),
    Path("sim/target"),
    Path("sim/trace/pbgen/target"),
    Path("tools/nvbit_tracer/nvbit_release"),
)
EXCLUDED_DIRECTORY_NAMES = {
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "env",
    "node_modules",
    "venv",
}
SOURCE_SUFFIXES = {".py", ".sh", ".bash", ".bsh", ".ksh", ".zsh"}
ROLES = {"campaign", "calibration", "diagnostic", "historical", "retired"}
SOURCE_KINDS = {"python", "shell"}
SCOPES = {"production", "test"}
BOUNDARIES = {"discover", "invoke", "both"}
EXECUTOR_CALLS = {
    "execute_ncu_action",
    "execute_ncu_supervised_child",
    "execute_ncu_supervised_executable",
}
PROCESS_APIS = {
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.run",
    "asyncio.create_subprocess_exec",
    "os.execl",
    "os.execle",
    "os.execlp",
    "os.execlpe",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
    "os.posix_spawn",
    "os.posix_spawnp",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
}
EXPLICIT_NCU_PROCESS_FUNCTIONS = {"_run_checked_ncu", "_run_retained_ncu_version"}
BARE_PROCESS_NAMES = {
    "Popen",
    "call",
    "check_call",
    "check_output",
    "run",
}
SHELL_STRING_APIS = {
    "asyncio.create_subprocess_shell",
    "os.popen",
    "os.system",
    "subprocess.getoutput",
    "subprocess.getstatusoutput",
}
DISCOVERY_APIS = {"shutil.which"}
HEREDOC_OPEN = re.compile(
    r"(?<!<)<<(?!<)-?\s*(?:'([^'\n]+)'|\"([^\"\n]+)\"|\\([^\s;&|]+)|([^\s;&|]+))"
)
ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=(.*)$", re.DOTALL)
VARIABLE_TOKEN = re.compile(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$")
SEPARATORS = {";", "&&", "||", "|", "&", "(", ")"}
CONTROL_WORDS = {"if", "then", "do", "while", "until", "time", "!"}
SHELLS = {"bash", "dash", "fish", "ksh", "sh", "zsh"}
SHELL_WRAPPERS = {
    "chrt", "find", "nice", "nohup", "parallel", "stdbuf", "taskset",
    "timeout", "xargs",
}


@dataclass(frozen=True)
class RegistryEntry:
    role: str
    source_kind: str
    scope: str
    boundary: str


NCU_INVOKER_REGISTRY: dict[str, RegistryEntry] = {
    "tools/local_labels/campaign_preflight.py": RegistryEntry(
        "campaign", "python", "production", "both"
    ),
    "tools/local_labels/ncu_action_executor.py": RegistryEntry(
        "campaign", "python", "production", "invoke"
    ),
    "tools/local_labels/label_driver.sh": RegistryEntry(
        "campaign", "shell", "production", "invoke"
    ),
    "tools/local_labels/parse_ncu_metrics.py": RegistryEntry(
        "campaign", "python", "production", "invoke"
    ),
    "tools/local_labels/sm86_capture_chunk.py": RegistryEntry(
        "campaign", "python", "production", "invoke"
    ),
    "tools/local_labels/capture_count_labels.py": RegistryEntry(
        "campaign", "python", "production", "invoke"
    ),
    "tools/local_labels/run_observation_contract_microfixtures.py": RegistryEntry(
        "calibration", "python", "production", "invoke"
    ),
    "tools/trace_sampling/golden_cells/capture_golden_cells.sh": RegistryEntry(
        "calibration", "shell", "production", "both"
    ),
    "tools/local_labels/probe_ncu_metric_availability.py": RegistryEntry(
        "diagnostic", "python", "production", "invoke"
    ),
    "tools/local_labels/validate_static_occupancy_agreement.py": RegistryEntry(
        "diagnostic", "python", "production", "both"
    ),
    "tools/local_labels/read_ncu_report.py": RegistryEntry(
        "diagnostic", "python", "production", "discover"
    ),
    "tools/local_labels/capture_cutlass_successor_receipts.py": RegistryEntry(
        "diagnostic", "python", "production", "both"
    ),
    "tools/local_labels/capture_timing_labels.py": RegistryEntry(
        "retired", "python", "production", "invoke"
    ),
    "tools/local_labels/capture_long_timing_cohort.py": RegistryEntry(
        "retired", "python", "production", "invoke"
    ),
    "tools/sol_baselines_harness/calibration/cache_prediction/ncu_harvester.py": RegistryEntry(
        "retired", "python", "production", "both"
    ),
    "tools/sol_baselines_harness/calibration/cache_prediction/sm86_expanded/emit_expanded_ncu_ledger.py": RegistryEntry(
        "retired", "python", "production", "discover"
    ),
    "tools/local_labels/test_campaign_preflight.py": RegistryEntry(
        "diagnostic", "python", "test", "discover"
    ),
    "tools/local_labels/test_ncu_action_executor.py": RegistryEntry(
        "diagnostic", "python", "test", "invoke"
    ),
}


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == "*" and node.module == "subprocess":
                    for name in BARE_PROCESS_NAMES:
                        aliases[name] = f"subprocess.{name}"
                    continue
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _qualified_name(node: ast.AST, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        owner = _qualified_name(node.value, aliases)
        return f"{owner}.{node.attr}" if owner else node.attr
    return ""


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _bare_ncu_path(value: str) -> bool:
    return Path(value).name.lower() == "ncu" and "/" not in value


def _api_name(node: ast.AST, aliases: dict[str, str]) -> str:
    return _qualified_name(node, aliases)


def _scope_nodes(scope: ast.AST):
    body = getattr(scope, "body", ())
    pending = list(body) if isinstance(body, list) else [body]
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        pending.extend(ast.iter_child_nodes(node))


def _scope_assignments(scope: ast.AST) -> dict[str, tuple[ast.AST, ...]]:
    assignments: dict[str, list[ast.AST]] = {}
    for node in _scope_nodes(scope):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                assignments.setdefault(target.id, []).append(node.value)
    return {name: tuple(values) for name, values in assignments.items()}


def _merge_assignments(
    parent: dict[str, tuple[ast.AST, ...]],
    local: dict[str, tuple[ast.AST, ...]],
) -> dict[str, tuple[ast.AST, ...]]:
    merged = dict(parent)
    merged.update(local)
    return merged


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _enclosing_assignments(
    scope: ast.AST,
    parents: dict[ast.AST, ast.AST],
    module_assignments: dict[str, tuple[ast.AST, ...]],
    *,
    include_scope: bool = False,
) -> dict[str, tuple[ast.AST, ...]]:
    ancestors: list[ast.AST] = []
    current: ast.AST | None = scope if include_scope else parents.get(scope)
    while current is not None:
        if isinstance(
            current,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            ancestors.append(current)
        current = parents.get(current)
    result = dict(module_assignments)
    for ancestor in reversed(ancestors):
        result = _merge_assignments(result, _scope_assignments(ancestor))
    return result


def _callable_aliases(
    assignments: dict[str, tuple[ast.AST, ...]], aliases: dict[str, str]
) -> dict[str, str]:
    result = dict(aliases)
    for name, values in assignments.items():
        qualified = {
            _qualified_name(value, result)
            for value in values
            if isinstance(value, (ast.Name, ast.Attribute))
        }
        if len(qualified) == 1:
            result[name] = qualified.pop()
    return result


def _resolved_literal_strings(
    node: ast.AST,
    assignments: dict[str, tuple[ast.AST, ...]],
    seen: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
        values: list[str] = []
        for value in assignments[node.id]:
            values.extend(_resolved_literal_strings(value, assignments, seen | {node.id}))
        return tuple(values)
    return ()


def _scope_calls(scope: ast.AST):
    return (node for node in _scope_nodes(scope) if isinstance(node, ast.Call))


def _shell_command_boundary(
    node: ast.AST,
    assignments: dict[str, tuple[ast.AST, ...]],
    seen: frozenset[str] = frozenset(),
) -> set[str]:
    if isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
        result: set[str] = set()
        for value in assignments[node.id]:
            result.update(_shell_command_boundary(value, assignments, seen | {node.id}))
        return result
    value = _literal_string(node)
    if value is not None:
        return _shell_boundary_kinds(value)
    if isinstance(node, ast.JoinedStr):
        result: set[str] = set()
        for candidate in ast.walk(node):
            if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
                result.update(_shell_boundary_kinds(candidate.value))
            elif isinstance(candidate, ast.Name):
                for resolved in _resolved_literal_strings(candidate, assignments):
                    if Path(resolved).name.lower() == "ncu":
                        result.add("invoke")
                        if _bare_ncu_path(resolved):
                            result.add("discover")
                if "ncu" in candidate.id.lower():
                    result.add("invoke")
        return result
    return _command_boundary(node, assignments, seen)


def _command_boundary(
    node: ast.AST,
    assignments: dict[str, tuple[ast.AST, ...]],
    seen: frozenset[str] = frozenset(),
) -> set[str]:
    if isinstance(node, ast.Name):
        if node.id in assignments and node.id not in seen:
            result: set[str] = set()
            for value in assignments[node.id]:
                result.update(
                    _command_boundary(value, assignments, seen | {node.id})
                )
            return result
        return {"invoke"} if "ncu" in node.id.lower() else set()
    if isinstance(node, ast.Attribute):
        return {"invoke"} if "ncu" in node.attr.lower() else set()
    if isinstance(node, ast.Starred):
        return _command_boundary(node.value, assignments, seen)
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
        first_values = _resolved_literal_strings(node.elts[0], assignments)
        basenames = {Path(value).name for value in first_values}
        if basenames & SHELLS:
            for index, element in enumerate(node.elts[:-1]):
                if _literal_string(element) == "-c":
                    return _shell_command_boundary(
                        node.elts[index + 1], assignments, seen
                    )
        if basenames & {"find"}:
            for index, element in enumerate(node.elts):
                if _literal_string(element) not in {"-exec", "-execdir", "-ok", "-okdir"}:
                    continue
                if index + 1 >= len(node.elts):
                    continue
                return _command_boundary(node.elts[index + 1], assignments, seen)
        if basenames & {"env", "sudo"}:
            index = 1
            option_arguments = {"-u", "-g", "-h", "-p", "-C", "-D", "-R", "-T", "-r", "-t"}
            while index < len(node.elts):
                values = _resolved_literal_strings(node.elts[index], assignments)
                if values and all(value.startswith("-") for value in values):
                    if values[0] in option_arguments:
                        index += 1
                    index += 1
                    continue
                if values and all("=" in value for value in values):
                    index += 1
                    continue
                break
            if index < len(node.elts):
                return _command_boundary(node.elts[index], assignments, seen)
            return set()
        wrappers = {
            "timeout", "nice", "nohup", "stdbuf", "taskset", "chrt", "xargs", "parallel"
        }
        if basenames & wrappers:
            wrapper = next(iter(basenames))
            index = 1
            if wrapper == "timeout":
                while index < len(node.elts):
                    values = _resolved_literal_strings(node.elts[index], assignments)
                    if values and values[0].startswith("-"):
                        index += 1
                        continue
                    index += 1
                    break
            elif wrapper == "nice":
                if index < len(node.elts) and _literal_string(node.elts[index]) == "-n":
                    index += 2
            elif wrapper in {"taskset", "chrt"}:
                while index < len(node.elts):
                    values = _resolved_literal_strings(node.elts[index], assignments)
                    if values and values[0].startswith("-"):
                        index += 2
                        continue
                    break
            else:
                while index < len(node.elts):
                    values = _resolved_literal_strings(node.elts[index], assignments)
                    if values and values[0].startswith("-"):
                        index += 1
                        continue
                    break
            if index < len(node.elts):
                return _command_boundary(node.elts[index], assignments, seen)
            return set()
        return _command_boundary(node.elts[0], assignments, seen)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if Path(node.value).name.lower() != "ncu":
            return set()
        result = {"invoke"}
        if _bare_ncu_path(node.value):
            result.add("discover")
        return result
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        if _literal_string(node.right) == "ncu":
            return {"invoke"}
    if isinstance(node, ast.Call):
        if _call_name(node) == "str" and node.args:
            return _command_boundary(node.args[0], assignments, seen)
        if _call_name(node) in {"which", "resolver"} and node.args:
            if _literal_string(node.args[0]) == "ncu":
                return {"discover", "invoke"}
    if isinstance(node, ast.JoinedStr):
        result: set[str] = set()
        for candidate in ast.walk(node):
            if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
                value = candidate.value.strip()
                if Path(value).name.lower() == "ncu":
                    result.add("invoke")
                    if _bare_ncu_path(value):
                        result.add("discover")
        identifiers = {
            candidate.id.lower()
            for candidate in ast.walk(node)
            if isinstance(candidate, ast.Name)
        }
        for candidate in ast.walk(node):
            if isinstance(candidate, ast.Name):
                for resolved in _resolved_literal_strings(candidate, assignments):
                    if Path(resolved).name.lower() == "ncu":
                        result.add("invoke")
                        if _bare_ncu_path(resolved):
                            result.add("discover")
        if any("ncu" in identifier for identifier in identifiers):
            result.add("invoke")
        return result
    return set()


def _resolved_bool(
    node: ast.AST,
    assignments: dict[str, tuple[ast.AST, ...]],
    seen: frozenset[str] = frozenset(),
) -> bool:
    if isinstance(node, ast.Constant):
        return node.value is True
    if isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
        return any(
            _resolved_bool(value, assignments, seen | {node.id})
            for value in assignments[node.id]
        )
    return False


def _keyword_truth(
    call: ast.Call,
    name: str,
    assignments: dict[str, tuple[ast.AST, ...]],
) -> bool:
    for keyword in call.keywords:
        if keyword.arg == name:
            return _resolved_bool(keyword.value, assignments)
    return False


def _call_command_arg(call: ast.Call, qualified: str) -> ast.AST | None:
    if qualified in {
        "os.spawnl",
        "os.spawnle",
        "os.spawnlp",
        "os.spawnlpe",
        "os.spawnv",
        "os.spawnve",
        "os.spawnvp",
        "os.spawnvpe",
    }:
        return call.args[1] if len(call.args) > 1 else None
    if call.args:
        return call.args[0]
    for keyword in call.keywords:
        if keyword.arg in {"path", "cmd"}:
            return keyword.value
    for keyword in call.keywords:
        if keyword.arg in {"args", "command", "program"}:
            return keyword.value
    return None


def _depends_on_parameters(
    node: ast.AST,
    parameters: set[str],
    assignments: dict[str, tuple[ast.AST, ...]],
    seen: frozenset[str] = frozenset(),
) -> bool:
    if isinstance(node, ast.Name):
        if node.id in parameters:
            return True
        if node.id in assignments and node.id not in seen:
            return any(
                _depends_on_parameters(value, parameters, assignments, seen | {node.id})
                for value in assignments[node.id]
            )
        return False
    if isinstance(node, ast.Starred):
        return _depends_on_parameters(node.value, parameters, assignments, seen)
    if isinstance(node, (ast.List, ast.Tuple)):
        if not node.elts:
            return False
        first_values = _resolved_literal_strings(node.elts[0], assignments)
        wrapper = Path(first_values[0]).name if first_values else ""
        if wrapper in SHELLS:
            for index, element in enumerate(node.elts[:-1]):
                if _literal_string(element) == "-c":
                    return _depends_on_parameters(
                        node.elts[index + 1], parameters, assignments, seen
                    )
        if wrapper in {"env", "sudo"}:
            index = 1
            option_arguments = {
                "-u", "-g", "-h", "-p", "-C", "-D", "-R", "-T", "-r", "-t"
            }
            while index < len(node.elts):
                values = _resolved_literal_strings(node.elts[index], assignments)
                if values and all(value.startswith("-") for value in values):
                    if values[0] in option_arguments:
                        index += 1
                    index += 1
                    continue
                if values and all("=" in value for value in values):
                    index += 1
                    continue
                break
            return index < len(node.elts) and _depends_on_parameters(
                node.elts[index], parameters, assignments, seen
            )
        if wrapper == "timeout":
            index = 1
            while index < len(node.elts):
                values = _resolved_literal_strings(node.elts[index], assignments)
                if values and values[0].startswith("-"):
                    index += 1
                    continue
                index += 1
                break
            return index < len(node.elts) and _depends_on_parameters(
                node.elts[index], parameters, assignments, seen
            )
        if wrapper == "nice" and len(node.elts) > 2 and _literal_string(node.elts[1]) == "-n":
            return _depends_on_parameters(node.elts[2], parameters, assignments, seen)
        if wrapper in {"taskset", "chrt"}:
            index = 1
            while index < len(node.elts):
                values = _resolved_literal_strings(node.elts[index], assignments)
                if values and values[0].startswith("-"):
                    index += 2
                    continue
                break
            return index < len(node.elts) and _depends_on_parameters(
                node.elts[index], parameters, assignments, seen
            )
        if wrapper == "find":
            for index, element in enumerate(node.elts):
                if _literal_string(element) in {"-exec", "-execdir", "-ok", "-okdir"}:
                    return index + 1 < len(node.elts) and _depends_on_parameters(
                        node.elts[index + 1], parameters, assignments, seen
                    )
        return _depends_on_parameters(node.elts[0], parameters, assignments, seen)
    if isinstance(node, ast.Call) and _call_name(node) == "str" and node.args:
        return _depends_on_parameters(node.args[0], parameters, assignments, seen)
    return False


def _has_ncu_descriptor_environment(call: ast.Call, aliases: dict[str, str]) -> bool:
    for keyword in call.keywords:
        if keyword.arg != "env" or not isinstance(keyword.value, ast.Call):
            continue
        if _qualified_name(keyword.value.func, aliases).endswith(
            "ncu_action_environment"
        ):
            return True
    return False


def _process_wrapper_names(
    tree: ast.AST, aliases: dict[str, str]
) -> set[str]:
    wrappers: set[str] = set()
    module_assignments = _scope_assignments(tree)
    parents = _parent_map(tree)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    changed = True
    while changed:
        changed = False
        for function in functions:
            parameters = {
                argument.arg
                for argument in (
                    list(function.args.posonlyargs)
                    + list(function.args.args)
                    + list(function.args.kwonlyargs)
                )
            }
            if function.args.vararg:
                parameters.add(function.args.vararg.arg)
            if function.args.kwarg:
                parameters.add(function.args.kwarg.arg)
            assignments = _enclosing_assignments(
                function, parents, module_assignments, include_scope=True
            )
            local_aliases = _callable_aliases(assignments, aliases)
            for call in _scope_calls(function):
                qualified = _api_name(call.func, local_aliases)
                command_arg = _call_command_arg(call, qualified)
                is_process_wrapper = qualified in (PROCESS_APIS | SHELL_STRING_APIS)
                is_nested_wrapper = _call_name(call) in wrappers
                if (
                    command_arg is not None
                    and (is_process_wrapper or is_nested_wrapper)
                    and _depends_on_parameters(command_arg, parameters, assignments)
                    and function.name not in wrappers
                ):
                    wrappers.add(function.name)
                    changed = True
                    break
    return wrappers


def _python_call_boundary_kinds(
    call: ast.Call,
    scope: ast.AST,
    assignments: dict[str, tuple[ast.AST, ...]],
    aliases: dict[str, str],
    process_wrappers: set[str],
) -> set[str]:
    kinds: set[str] = set()
    qualified = _api_name(call.func, aliases)
    name = _call_name(call)
    if name in EXECUTOR_CALLS:
        kinds.add("invoke")
    if qualified in DISCOVERY_APIS or name in {"which", "resolver"}:
        if call.args:
            discovery_values = _resolved_literal_strings(call.args[0], assignments)
            if any(Path(value).name.lower() == "ncu" for value in discovery_values):
                kinds.add("discover")
    command_arg = _call_command_arg(call, qualified)
    if command_arg is not None and qualified in SHELL_STRING_APIS:
        kinds.update(_shell_command_boundary(command_arg, assignments))
    elif command_arg is not None and qualified in PROCESS_APIS:
        if qualified.startswith("subprocess.") and _keyword_truth(
            call, "shell", assignments
        ):
            kinds.update(_shell_command_boundary(command_arg, assignments))
        else:
            kinds.update(_command_boundary(command_arg, assignments))
        if (
            isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
            and scope.name in EXPLICIT_NCU_PROCESS_FUNCTIONS
        ):
            kinds.add("invoke")
        if _has_ncu_descriptor_environment(call, aliases):
            kinds.add("invoke")
    elif (
        name in process_wrappers
        or qualified in process_wrappers
    ) and command_arg is not None:
        kinds.update(_command_boundary(command_arg, assignments))
        kinds.update(_shell_command_boundary(command_arg, assignments))
    return kinds


def _definition_header_calls(tree: ast.AST):
    for definition in ast.walk(tree):
        nodes: list[ast.AST] = []
        if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes.extend(definition.decorator_list)
            nodes.extend(definition.args.defaults)
            nodes.extend(
                default for default in definition.args.kw_defaults if default is not None
            )
            if definition.returns is not None:
                nodes.append(definition.returns)
        elif isinstance(definition, ast.ClassDef):
            nodes.extend(definition.decorator_list)
            nodes.extend(definition.bases)
            nodes.extend(keyword.value for keyword in definition.keywords)
        elif isinstance(definition, ast.Lambda):
            nodes.extend(definition.args.defaults)
            nodes.extend(
                default for default in definition.args.kw_defaults if default is not None
            )
        for node in nodes:
            for candidate in ast.walk(node):
                if isinstance(candidate, ast.Call):
                    yield candidate, definition


def _python_boundary_kinds(text: str, *, path: Path) -> set[str]:
    tree = ast.parse(text, filename=str(path))
    module_aliases = _import_aliases(tree)
    module_assignments = _scope_assignments(tree)
    process_wrappers = _process_wrapper_names(tree, module_aliases)
    parents = _parent_map(tree)
    kinds: set[str] = set()

    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        if function.name in EXECUTOR_CALLS:
            kinds.add("invoke")

    scopes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
    ]
    for scope in scopes:
        assignments = _enclosing_assignments(
            scope, parents, module_assignments, include_scope=True
        )
        aliases = _callable_aliases(assignments, module_aliases)
        for call in _scope_calls(scope):
            kinds.update(
                _python_call_boundary_kinds(
                    call, scope, assignments, aliases, process_wrappers
                )
            )

    for call, definition in _definition_header_calls(tree):
        header_assignments = _enclosing_assignments(
            definition, parents, module_assignments
        )
        kinds.update(
            _python_call_boundary_kinds(
                call, definition, header_assignments, module_aliases, process_wrappers
            )
        )
    return kinds


def _heredoc_runs_shell(line: str) -> bool:
    for segment in _command_segments(_shell_tokens(line)):
        index = 0
        while index < len(segment) and (
            ASSIGNMENT.fullmatch(segment[index]) or segment[index] in CONTROL_WORDS
        ):
            index += 1
        while index < len(segment):
            command = Path(segment[index]).name
            index += 1
            if command in SHELLS:
                return True
            if command in {"env", "exec", "sudo"}:
                while index < len(segment) and (
                    segment[index].startswith("-")
                    or ASSIGNMENT.fullmatch(segment[index])
                ):
                    index += 1
                continue
            if command in {"timeout", "nice", "nohup", "stdbuf", "taskset", "chrt"}:
                option_arguments = {
                    "-n", "-c", "-p", "-f", "-m", "-o", "-i", "-r", "-a"
                }
                while index < len(segment) and segment[index].startswith("-"):
                    option = segment[index]
                    index += 1
                    if option in option_arguments and index < len(segment):
                        index += 1
                if command == "timeout" and index < len(segment):
                    index += 1
                continue
            if command in {"xargs", "parallel"}:
                while index < len(segment) and segment[index].startswith("-"):
                    index += 1
                if index < len(segment) and Path(segment[index]).name in SHELLS:
                    return True
                break
            break
    return False


def _logical_shell_lines(text: str):
    pending = ""
    heredoc_delimiter: str | None = None
    heredoc_executes = False
    heredoc_body: list[str] = []
    for raw_line in text.splitlines():
        if heredoc_delimiter is not None:
            if raw_line.strip() == heredoc_delimiter:
                if heredoc_executes:
                    yield from heredoc_body
                heredoc_delimiter = None
                heredoc_executes = False
                heredoc_body = []
            else:
                heredoc_body.append(raw_line)
            continue
        line = f"{pending}{raw_line}"
        if line.endswith("\\") and not line.endswith("\\\\"):
            pending = f"{line[:-1]} "
            continue
        pending = ""
        match = HEREDOC_OPEN.search(line)
        if match:
            heredoc_delimiter = next(
                group for group in match.groups() if group is not None
            )
            heredoc_executes = _heredoc_runs_shell(
                f"{line[: match.start()]} {line[match.end() :]}"
            )
            line = line[: match.start()]
        yield line
    if pending:
        yield pending


def _shell_tokens(line: str) -> list[str]:
    escaped_semicolon = "__NCU_ESCAPED_SEMICOLON__"
    line = line.replace(r"\;", escaped_semicolon)
    try:
        lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|()")
        lexer.commenters = "#"
        lexer.whitespace_split = True
        return [
            token.replace(escaped_semicolon, r"\;")
            for token in lexer
        ]
    except ValueError:
        return []


def _command_segments(tokens: list[str]):
    segment: list[str] = []
    for token in tokens:
        if token in SEPARATORS:
            if segment:
                yield segment
                segment = []
        else:
            segment.append(token)
    if segment:
        yield segment


def _shell_executable_boundary(
    token: str, variables: dict[str, str]
) -> set[str]:
    match = VARIABLE_TOKEN.fullmatch(token)
    if match:
        variable = match.group(1)
        result: set[str] = set()
        if variable in variables:
            result.update(_shell_executable_boundary(variables[variable], variables))
        if "ncu" in variable.lower():
            result.add("invoke")
        return result
    if Path(token).name.lower() != "ncu":
        return set()
    result = {"invoke"}
    if _bare_ncu_path(token):
        result.add("discover")
    return result


def _shell_variable_values(token: str, variables: dict[str, str]) -> tuple[str, ...]:
    match = VARIABLE_TOKEN.fullmatch(token)
    if match and match.group(1) in variables:
        return (variables[match.group(1)],)
    return (token,)


def _shell_segment_kinds(
    segment: list[str], variables: dict[str, str]
) -> set[str]:
    kinds: set[str] = set()
    index = 0
    while index < len(segment) and segment[index] in CONTROL_WORDS:
        index += 1
    while index < len(segment):
        match = ASSIGNMENT.fullmatch(segment[index])
        if not match:
            break
        variable = segment[index].split("=", 1)[0]
        value = match.group(1)
        variables[variable] = value
        substitutions = _shell_command_substitutions(value)
        for substitution in substitutions:
            nested_kinds = _shell_boundary_kinds(substitution, variables)
            kinds.update(nested_kinds)
            if nested_kinds & {"discover", "invoke"}:
                variables[variable] = "ncu"
        if re.search(r"(?:which|command\s+-[vV])\s+[^;&|]*\bncu\b", value):
            kinds.add("discover")
            variables[variable] = "ncu"
        index += 1
    if index >= len(segment):
        return kinds

    command = segment[index]
    index += 1
    basename = Path(command).name
    if basename == "export":
        return kinds | _shell_segment_kinds(segment[index:], variables)
    if basename in {"local", "readonly", "declare", "typeset"}:
        while index < len(segment):
            token = segment[index]
            index += 1
            if token.startswith("-"):
                continue
            match = ASSIGNMENT.fullmatch(token)
            if not match:
                continue
            variable = token.split("=", 1)[0]
            value = match.group(1)
            variables[variable] = value
            for substitution in _shell_command_substitutions(value):
                nested_kinds = _shell_boundary_kinds(substitution, variables)
                kinds.update(nested_kinds)
                if nested_kinds & {"discover", "invoke"}:
                    variables[variable] = "ncu"
            if Path(value).name.lower() == "ncu":
                variables[variable] = "ncu"
        return kinds
    if basename in {"which", "type"}:
        arguments = [token for token in segment[index:] if not token.startswith("-")]
        if any(
            Path(value).name.lower() == "ncu"
            for token in arguments
            for value in _shell_variable_values(token, variables)
        ):
            kinds.add("discover")
        return kinds
    if basename == "command":
        options: list[str] = []
        while index < len(segment) and segment[index].startswith("-"):
            options.append(segment[index])
            index += 1
        if any(option in {"-v", "-V"} for option in options):
            if index < len(segment) and any(
                Path(value).name.lower() == "ncu"
                for value in _shell_variable_values(segment[index], variables)
            ):
                kinds.add("discover")
            return kinds
        if index >= len(segment):
            return kinds
        command = segment[index]
        basename = Path(command).name
        index += 1
    elif basename in {"exec", "sudo"}:
        while index < len(segment) and segment[index].startswith("-"):
            index += 1
        if index >= len(segment):
            return kinds
        command = segment[index]
        basename = Path(command).name
        index += 1
    elif basename == "env":
        while index < len(segment):
            if segment[index].startswith("-") or ASSIGNMENT.fullmatch(segment[index]):
                index += 1
                continue
            break
        if index >= len(segment):
            return kinds
        command = segment[index]
        basename = Path(command).name
        index += 1

    if basename in {"which", "type", "command"}:
        kinds.update(_shell_segment_kinds([command, *segment[index:]], variables))
        return kinds
    if basename == "eval":
        payload = " ".join(segment[index:])
        for value in _shell_variable_values(payload, variables):
            kinds.update(_shell_boundary_kinds(value, variables))
        return kinds
    if basename == "find":
        for option_index, option in enumerate(segment[index:], start=index):
            if option not in {"-exec", "-execdir", "-ok", "-okdir"}:
                continue
            end = len(segment)
            for terminator in {";", r"\;", "+"}:
                try:
                    end = min(end, segment.index(terminator, option_index + 1))
                except ValueError:
                    pass
            if option_index + 1 < end:
                kinds.update(_shell_segment_kinds(segment[option_index + 1 : end], variables))
        return kinds
    if basename in SHELL_WRAPPERS:
        for child_index in range(index, len(segment)):
            child_kinds = _shell_segment_kinds(segment[child_index:], variables)
            if child_kinds:
                kinds.update(child_kinds)
                break
        return kinds
    if basename in SHELLS and index < len(segment):
        while index < len(segment):
            if segment[index] == "-c" and index + 1 < len(segment):
                return _shell_boundary_kinds(segment[index + 1], variables)
            index += 1
    kinds.update(_shell_executable_boundary(command, variables))
    return kinds


def _shell_command_substitutions(text: str) -> tuple[str, ...]:
    substitutions: list[str] = []
    index = 0
    quote: str | None = None
    while index < len(text):
        character = text[index]
        if quote == "'":
            if character == "'":
                quote = None
            index += 1
            continue
        if quote == '"':
            if character == "\\":
                index += 2
                continue
            if character == '"':
                quote = None
                index += 1
                continue
        else:
            if character == "#":
                break
            if character in {"'", '"'}:
                quote = character
                index += 1
                continue
        if character == "\\":
            index += 2
            continue
        if character == "$" and index + 1 < len(text) and text[index + 1] == "(":
            depth = 1
            inner_index = index + 2
            inner_quote: str | None = None
            while inner_index < len(text) and depth:
                inner = text[inner_index]
                if inner_quote == "'":
                    if inner == "'":
                        inner_quote = None
                elif inner_quote == '"':
                    if inner == "\\":
                        inner_index += 1
                    elif inner == '"':
                        inner_quote = None
                elif inner in {"'", '"'}:
                    inner_quote = inner
                elif inner == "\\":
                    inner_index += 1
                elif inner == "$(":
                    depth += 1
                elif inner == "(":
                    depth += 1
                elif inner == ")":
                    depth -= 1
                inner_index += 1
            if depth == 0:
                substitutions.append(text[index + 2 : inner_index - 1])
                index = inner_index
                continue
        if character == "`" and quote != "'":
            end = index + 1
            while end < len(text):
                if text[end] == "\\":
                    end += 2
                    continue
                if text[end] == "`":
                    substitutions.append(text[index + 1 : end])
                    index = end + 1
                    break
                end += 1
            else:
                index = end
            continue
        index += 1
    return tuple(substitutions)


def _shell_boundary_kinds(
    text: str, variables: dict[str, str] | None = None
) -> set[str]:
    kinds: set[str] = set()
    if variables is None:
        variables = {}
    for line in _logical_shell_lines(text):
        nested_source = "" if line.lstrip().startswith("#") else line
        for segment in _command_segments(_shell_tokens(line)):
            kinds.update(_shell_segment_kinds(segment, variables))
        for nested in _shell_command_substitutions(nested_source):
            kinds.update(_shell_boundary_kinds(nested, variables))
    return kinds


def _is_test_source(path: Path) -> bool:
    relative = path.relative_to(REPO_ROOT)
    return (
        path.name.startswith("test_")
        or path.stem.endswith("_test")
        or "tests" in relative.parts
    )


def _is_excluded(relative: Path) -> bool:
    return any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts) or any(
        relative.parts[: len(prefix.parts)] == prefix.parts
        for prefix in EXCLUDED_PREFIXES
    )


def _source_kind(path: Path) -> str | None:
    if path.suffix == ".py":
        return "python"
    if path.suffix in SOURCE_SUFFIXES - {".py"}:
        return "shell"
    if path.suffix and not os.access(path, os.X_OK):
        return None
    try:
        first_line = path.open("r", encoding="utf-8", errors="replace").readline()
    except OSError:
        return None
    if not first_line.startswith("#!"):
        return None
    if re.search(r"\bpython[0-9.]*\b", first_line):
        return "python"
    if re.search(r"\b(?:ba|da|fi|k|z)?sh\b", first_line):
        return "shell"
    return None


def _source_paths():
    this_file = Path(__file__).resolve()
    for current, directories, files in os.walk(
        REPO_ROOT, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        directories[:] = [
            name
            for name in sorted(directories)
            if not _is_excluded((current_path / name).relative_to(REPO_ROOT))
        ]
        for name in sorted(files):
            path = current_path / name
            relative = path.relative_to(REPO_ROOT)
            if path == this_file or path.is_symlink() or _is_excluded(relative):
                continue
            if path.is_file() and _source_kind(path) is not None:
                yield path


def _discovered_registry() -> dict[str, tuple[str, str, set[str]]]:
    discovered: dict[str, tuple[str, str, set[str]]] = {}
    for path in _source_paths():
        text = path.read_text(encoding="utf-8", errors="replace")
        source_kind = _source_kind(path)
        assert source_kind is not None
        kinds = (
            _python_boundary_kinds(text, path=path)
            if source_kind == "python"
            else _shell_boundary_kinds(text)
        )
        if kinds:
            discovered[path.relative_to(REPO_ROOT).as_posix()] = (
                source_kind,
                "test" if _is_test_source(path) else "production",
                kinds,
            )
    return discovered


def _boundary_set(boundary: str) -> set[str]:
    return {"discover", "invoke"} if boundary == "both" else {boundary}


def test_registry_is_closed_and_matches_direct_ncu_sources():
    malformed = {
        path: entry
        for path, entry in NCU_INVOKER_REGISTRY.items()
        if entry.role not in ROLES
        or entry.source_kind not in SOURCE_KINDS
        or entry.scope not in SCOPES
        or entry.boundary not in BOUNDARIES
    }
    assert not malformed, f"malformed NCU invoker registry entries: {malformed!r}"

    discovered = _discovered_registry()
    registered_paths = set(NCU_INVOKER_REGISTRY)
    discovered_paths = set(discovered)
    unknown = sorted(discovered_paths - registered_paths)
    stale = sorted(registered_paths - discovered_paths)
    mismatched = {
        path: {
            "registered": NCU_INVOKER_REGISTRY[path],
            "discovered_source_kind": discovered[path][0],
            "discovered_scope": discovered[path][1],
            "discovered_boundaries": sorted(discovered[path][2]),
        }
        for path in sorted(registered_paths & discovered_paths)
        if NCU_INVOKER_REGISTRY[path].source_kind != discovered[path][0]
        or NCU_INVOKER_REGISTRY[path].scope != discovered[path][1]
        or _boundary_set(NCU_INVOKER_REGISTRY[path].boundary)
        != discovered[path][2]
    }
    assert not (unknown or stale or mismatched), (
        "direct NCU source registry drifted; registration accounts for a source "
        "but does not authorize campaign evidence:\n"
        f"unknown={unknown!r}\nstale={stale!r}\nmismatched={mismatched!r}"
    )


def test_python_detector_controls():
    production = REPO_ROOT / "tools/fixture.py"
    test_source = REPO_ROOT / "tools/test_fixture.py"
    samples = (
        "import subprocess\nsubprocess.run(['ncu', '--version'])\n",
        "import subprocess as sp\nsp.run(['ncu', '--version'])\n",
        "from subprocess import run as launch\nlaunch(['ncu', '--version'])\n",
        "import os\nos.system('ncu --version')\n",
        "import asyncio\nasyncio.create_subprocess_exec('ncu', '--version')\n",
        "import subprocess\nsubprocess.run('prepare && ncu --version', shell=True)\n",
        "import subprocess\nsubprocess.run(args=['ncu', '--version'])\n",
        "import subprocess\nrun = subprocess.run\nrun(['ncu', '--version'])\n",
        "import os\nos.spawnvp(os.P_WAIT, 'ncu', ['ncu', '--version'])\n",
        "import subprocess\nclass Probe:\n    version = subprocess.run(['ncu', '--version'])\n",
        "import subprocess\ndef probe(version=subprocess.run(['ncu', '--version'])):\n    pass\n",
        "import subprocess\nshell = True\nsubprocess.run('ncu --version', shell=shell)\n",
        "import os\nos.popen(cmd='ncu --version')\n",
        "import asyncio\nasyncio.create_subprocess_shell(cmd='ncu --version')\n",
    )
    for source in samples:
        assert _python_boundary_kinds(source, path=production) == {
            "discover",
            "invoke",
        }
    assert _python_boundary_kinds(
        "import subprocess\nsubprocess.run([f'{cuda_home}/ncu', '--version'])\n",
        path=production,
    ) == {"invoke"}
    assert _python_boundary_kinds(samples[0], path=test_source) == {
        "discover",
        "invoke",
    }
    assert _python_boundary_kinds(
        "import shutil\ntool = 'ncu'\nshutil.which(tool)\n",
        path=production,
    ) == {"discover"}
    assert _python_boundary_kinds(
        "def execute_ncu_action(authority):\n    pass\n",
        path=production,
    ) == {"invoke"}
    assert _python_boundary_kinds(
        "import subprocess\ndef run_driver(driver, ncu):\n"
        "    subprocess.run([driver, '--ncu', ncu])\n",
        path=production,
    ) == set()
    assert _python_boundary_kinds(
        "import subprocess\ndef run(command):\n"
        "    subprocess.run(['echo'])\nrun(['ncu'])\n",
        path=production,
    ) == set()
    assert _python_boundary_kinds(
        "import subprocess\ndef wrapper(cmd):\n"
        "    subprocess.run(cmd)\n"
        "def outer(cmd):\n"
        "    wrapper(command=cmd)\n"
        "outer(['ncu'])\n",
        path=production,
    ) == {"discover", "invoke"}
    assert _python_boundary_kinds(
        "import subprocess\ndef one():\n    cmd = ['ncu']\n"
        "def two():\n    subprocess.run(cmd)\n",
        path=production,
    ) == set()
    assert _python_boundary_kinds(
        "def run(command):\n    return command\nrun(['ncu'])\n",
        path=production,
    ) == set()
    assert _python_boundary_kinds(
        "from subprocess import *\nPopen(['ncu'])\n",
        path=production,
    ) == {"discover", "invoke"}


def test_shell_detector_controls():
    invoke_and_discover = (
        "ncu --version\n",
        "env ncu --version\n",
        "exec ncu --version\n",
        "prepare && ncu --version\n",
        "tool=ncu\n$tool --version\n",
        "sh -c 'ncu --version'\n",
        "timeout 30 ncu --version\n",
        "timeout 1 bash <<EOF\nncu --version\nEOF\n",
        "xargs ncu\n",
        "parallel ncu\n",
        "find . -exec ncu {} \\;\n",
        "version=\"$(ncu --version)\"\n",
    )
    for source in invoke_and_discover:
        assert _shell_boundary_kinds(source) == {"discover", "invoke"}
    assert _shell_boundary_kinds("command -v ncu\n") == {"discover"}
    assert _shell_boundary_kinds('tool="$(command -v ncu)"\n') == {"discover"}
    assert _shell_boundary_kinds("which ncu\n") == {"discover"}
    assert _shell_boundary_kinds('"$NCU_ACTION_BIN" --version\n') == {"invoke"}
    assert _shell_boundary_kinds("printf '%s\\n' ncu\n") == set()
    assert _shell_boundary_kinds("printf '%s\\n' '$(ncu --version)'\n") == set()
    assert _shell_boundary_kinds("echo ok # $(ncu --version)\n") == set()
    assert _shell_boundary_kinds("cat <<'EOF'\nncu --version\nEOF\n") == set()
    assert _shell_boundary_kinds("cat <<1\nncu --version\n1\n") == set()
    assert _shell_boundary_kinds("cat <<'EOF-X'\nncu --version\nEOF-X\n") == set()
    assert _shell_boundary_kinds("cat <<\\EOF\nncu --version\nEOF\n") == set()
    assert _shell_boundary_kinds("bash <<'EOF'\nncu --version\nEOF\n") == {
        "discover",
        "invoke",
    }
    assert _shell_boundary_kinds("find . -name ncu\n") == set()
    assert _shell_boundary_kinds("eval 'ncu --version'\n") == {
        "discover",
        "invoke",
    }
    assert _shell_boundary_kinds('name=ncu\nwhich "$name"\n') == {"discover"}
    assert _shell_boundary_kinds('tool="$(which ncu)"\n"$tool" --version\n') == {
        "discover",
        "invoke",
    }
