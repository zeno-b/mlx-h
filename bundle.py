#!/usr/bin/env python3
"""Regenerate a single-file deploy_bundled.py from the local_llm package.

The split improves readability; this keeps the old "one file you can copy
anywhere and run" deployment. It concatenates the modules in dependency order,
drops the intra-package imports and the generated __all__ blocks, and keeps one
copy of the shared import header.

One file means one namespace, and the package relies on things a namespace
does not give for free, so the bundler also:

  * refuses to build when a package module is missing from ORDER (the bundle
    once shipped without documents/skills/checkpoints/mcp and died on import);
  * moves a module up when ORDER lists it after a module that imports it;
  * turns `from . import X` into a module-like object for X, and rewrites
    relative imports inside functions into reads of the shared globals;
  * renames private names that two modules bind with different meanings
    (config's `import datetime` against everyone's `from datetime import
    datetime`), and refuses when two EXPORTED names collide;
  * embeds files read as text at run time (tool_rpc_shim.py) in
    _BUNDLED_SOURCES, and drops per-module `if __name__ == "__main__":` blocks.

    python3 bundle.py [OUTPUT]   -> writes deploy_bundled.py (or OUTPUT)
    python3 deploy_bundled.py --selftest
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ORDER = [
    "core", "obslog", "codecheck", "database", "sysutil",
    # The settings metadata feeds both the generated Settings markup and the
    # /api/settings/schema route, so it precedes the UI parts.
    "settings_schema",
    # UI parts must precede ui.py, which assembles HTML_PAGE from them.
    "ui_styles", "ui_markup", "ui_script_chat", "ui_script_views",
    "ui_script_tasks", "ui_script_models", "ui_script_auth", "ui_script_panels",
    "ui_script_kit", "ui_script_admin", "ui_script_settings", "ui_settings",
    # Settings panels for client wiring, tokens, MCP, catalogue, approvals,
    # audit and projects (one more script part ui.py assembles).
    "ui_script_integrations",
    # Tool-call parsing: config clamps TOOL_CALL_FORMAT with it and llm reads
    # its reasoning-tag constants at class-definition time.
    "ui", "toolparse", "config", "model_server", "websearch", "training", "calculator",
    # tools star-imports these four and reads `documents` as a module object.
    "documents", "skills", "checkpoints", "mcp",
    # Tool approval policy: tools star-imports it (ToolRegistry.call enforces
    # it); it reads turnctx, which resolve_order hoists above it.
    "policy",
    "tools", "llm", "model_client", "textutil",
    # tasks star-imports admission (the generation admission controller).
    "taskstate", "agent", "admission", "tasks", "auth", "cluster", "claude_import",
    "appctx",
    "turnctx",
    "backends",
    # Per-user preferences: store + /api/settings/schema, /api/me/preferences.
    "prefs",
    # Cluster security + remote access (model_client and cluster call into
    # them only at run time, so they may follow those modules here), then the
    # layer-sharding backend, which registers with backends and cluster.
    "nodesec", "remote_access", "sharding",
    # Training methods + autopilot: training.py star-imports it (resolve_order
    # moves it up), and api registers its routes.
    "train_methods",
    # Model catalogue: imported by sysutil, so the bundler hoists it above that.
    "catalog",
    # The app's own MLX server registers a backend, so it follows backends.
    "mlx_engine",
    # Workspaces/projects: tools star-imports it (resolve_order hoists it and
    # turnctx above tools), and api registers its routes.
    "workspaces",
    # Chat turn registry (resume/cancel); api star-imports it.
    "streams",
    # OpenAI / Anthropic compatible /v1 API: anthropic_compat star-imports
    # openai_compat (shared engine), api registers both.
    "openai_compat", "anthropic_compat",
    # API tokens (auth star-imports it, so resolve_order hoists it above auth)
    # and client wiring (`deploy.py wire`, /api/integrations/snippets).
    "api_tokens", "wire",
    # Multimodal input: vision registers the mlx-vlm backend (after backends and
    # catalog); audio star-imports attachments, which star-imports tools.
    "vision", "attachments", "audio",
    # Named capability tools (tools star-imports it; resolve_order hoists it
    # above tools) and the MCP server endpoint (api registers its routes).
    "tools_extra", "mcp_server",
    # The Settings tool catalogue route (reads config, policy, mcp, tools_extra).
    "tool_catalog",
    # ---- extension modules: feature modules in dependency order, above this line ---- #
    "api",
    "diagnostics", "selftest", "cli",
]

PKG = Path(__file__).resolve().parent / "local_llm"
OUT = Path(__file__).resolve().parent / "deploy_bundled.py"

# Package files that are never concatenated: the package entry points. The
# bundle's own footer is its entry point.
SKIP = {"__init__", "__main__"}

# Package files the app reads as TEXT at run time instead of importing them.
# tool_rpc_shim.py is copied into the execute_code sandbox and imported there as
# `tools`; it used to be located with Path(__file__).with_name(...), and a single
# file has no sibling to find. Their sources are embedded in _BUNDLED_SOURCES,
# keyed by file name, and the readers fall back to that dict.
EMBED = ["tool_rpc_shim"]

# Names the bundle defines for itself. A package module binding one of them
# would silently replace the bundle's copy, so that is a build error.
RESERVED = {"_BUNDLED_SOURCES", "_BUNDLED_MODULES", "_bundled_namespace",
            "_bundled_types"}

# Private names that collide across modules are renamed to this prefix plus the
# module and the original name, e.g. taskstate's _FENCE -> _bundled_taskstate__FENCE.
RENAME_PREFIX = "_bundled_"


class BundleError(SystemExit):
    """A package shape the single file cannot reproduce faithfully.

    Raised instead of writing a file that would only fail later, at the line
    that happens to need the missing name (the bundle's NameError on
    `documents` at import time was exactly that).
    """

    def __init__(self, message: str) -> None:
        super().__init__(f"bundle.py: {message}")


# --------------------------------------------------------------------------- #
# Reading the package
# --------------------------------------------------------------------------- #

def _is_main_guard(node: ast.stmt) -> bool:
    """`if __name__ == "__main__":` at module level."""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    return (isinstance(test.left, ast.Name) and test.left.id == "__name__"
            and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__")


def _is_all(node: ast.stmt) -> bool:
    """An assignment to, or in-place extension of, __all__."""
    if isinstance(node, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
    if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        return isinstance(node.target, ast.Name) and node.target.id == "__all__"
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        func = node.value.func
        return (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id == "__all__")
    return False


_BLOCKS = tuple(t for t in (ast.If, ast.Try, getattr(ast, "TryStar", None),
                            ast.With, ast.For, ast.While) if t is not None)


def _module_level(body: list, fallback: bool = False):
    """Yield (statement, in_except_handler) for code that runs at import time.

    Looks inside top-level if/try/with/for blocks (their bodies run when the
    module loads) but never into functions or classes. Main guards are skipped
    because the bundle drops them.
    """
    for node in body:
        if _is_main_guard(node):
            continue
        if isinstance(node, (ast.For, ast.With)):
            yield node, fallback             # for its loop / `as` targets
        if isinstance(node, _BLOCKS):
            yield from _module_level(node.body, fallback)
            yield from _module_level(getattr(node, "orelse", []), fallback)
            for handler in getattr(node, "handlers", []):
                yield from _module_level(handler.body, True)
            yield from _module_level(getattr(node, "finalbody", []), fallback)
        else:
            yield node, fallback


def _target_names(target: ast.expr):
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _target_names(element)
    elif isinstance(target, ast.Starred):
        yield from _target_names(target.value)


class Module:
    """One package module, parsed once, with what the bundler needs to know."""

    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self.text = text
        try:
            self.tree = ast.parse(text, filename=f"local_llm/{name}.py")
        except SyntaxError as exc:
            raise BundleError(f"local_llm/{name}.py does not parse: {exc}") from None
        self.lines = text.split("\n")
        self.deps: list[str] = []            # modules its import-time code needs
        self.exports: set[str] = set()       # __all__ plus names others import by name
        self.bindings: dict[str, tuple] = {}  # module-level name -> meaning
        self.reexports: set[str] = set()     # `from .x import name`: x's global
        self.relative: list[tuple[ast.ImportFrom, bool]] = []  # (node, module-level?)
        self.loads: set[str] = set()         # every name it reads, at any depth
        self._scan()

    def where(self, node: ast.AST) -> str:
        return f"local_llm/{self.name}.py:{node.lineno}"

    def _relative_targets(self, node: ast.ImportFrom) -> list[str]:
        if node.level != 1 or (node.module and "." in node.module):
            raise BundleError(f"{self.where(node)}: only `from .module import ...` and "
                              "`from . import module` can be bundled")
        if node.module:
            return [node.module]
        return [alias.name for alias in node.names]

    def _scan(self) -> None:
        # Absolute imports of the package itself would be hoisted into the header
        # and fail there: the single file has no package to import.
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            else:
                if isinstance(node, ast.ImportFrom):
                    self.relative.append((node, False))
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    self.loads.add(node.id)
                continue
            if any(n == "local_llm" or n.startswith("local_llm.") for n in names):
                raise BundleError(f"{self.where(node)}: import the package relatively "
                                  "(`from .x import *`); the single file has no "
                                  "local_llm package to import")

        literal_all = False
        meanings: dict[str, list[tuple[str, tuple]]] = {}

        def bind(name: str, kind: str, meaning: tuple) -> None:
            if name != "__all__":
                meanings.setdefault(name, []).append((kind, meaning))

        module_level = set()
        for node, fallback in _module_level(self.tree.body):
            module_level.add(id(node))
            if _is_all(node):
                value = getattr(node, "value", None)
                if isinstance(node, ast.Expr):
                    value = node.value.args[0] if node.value.args else None
                if isinstance(value, (ast.List, ast.Tuple)) and all(
                        isinstance(e, ast.Constant) and isinstance(e.value, str)
                        for e in value.elts):
                    self.exports.update(e.value for e in value.elts)
                    literal_all = True
                continue
            own = ("def", self.name)
            if isinstance(node, ast.ImportFrom) and node.level:
                for target in self._relative_targets(node):
                    if target not in self.deps:
                        self.deps.append(target)
                if node.module is None:
                    for alias in node.names:
                        bind(alias.asname or alias.name, "import", ("ns", alias.name))
                else:
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        if alias.asname and alias.asname != alias.name:
                            bind(alias.asname, "import", ("pkg", node.module, alias.name))
                        else:
                            self.reexports.add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        bind(alias.asname, "import", ("module", alias.name))
                    else:
                        top = alias.name.split(".")[0]
                        bind(top, "import", ("module", top))
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        raise BundleError(f"{self.where(node)}: `from {node.module} import *` "
                                          "cannot be bundled safely; import the names")
                    bind(alias.asname or alias.name, "import",
                         ("from", node.module, alias.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bind(node.name, "fallback" if fallback else "def", own)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    for name in _target_names(target):
                        bind(name, "fallback" if fallback else "def", own)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                for name in _target_names(node.target):
                    bind(name, "fallback" if fallback else "def", own)
            elif isinstance(node, ast.For):
                for name in _target_names(node.target):
                    bind(name, "def", own)
            elif isinstance(node, ast.With):
                for item in node.items:
                    if item.optional_vars is not None:
                        for name in _target_names(item.optional_vars):
                            bind(name, "def", own)

        # One meaning per name and module. A real definition wins over an import
        # (the module's own code sees its definition), and an import wins over
        # an except-branch fallback: `try: from fastapi import Request` /
        # `except ImportError: Request = Any` means the fastapi object wherever
        # fastapi is installed, the same object every other module imports.
        for name, entries in meanings.items():
            kinds = {kind: meaning for kind, meaning in reversed(entries)}
            self.bindings[name] = kinds.get("def") or kinds.get("import") or kinds["fallback"]
        if not literal_all:
            # No literal __all__: every name may be used by a star import, so no
            # name of this module is safe to rename.
            self.exports.update(self.bindings)
        self.relative = [(node, id(node) in module_level) for node, _ in self.relative]


# --------------------------------------------------------------------------- #
# Planning: order and renames
# --------------------------------------------------------------------------- #

def load_modules(order: list[str]) -> dict[str, Module]:
    """Read every module in ORDER, after checking ORDER covers the package."""
    duplicates = sorted({n for n in order if order.count(n) > 1})
    if duplicates:
        raise BundleError("ORDER lists these modules more than once: " + ", ".join(duplicates))
    on_disk = {p.stem for p in PKG.glob("*.py")} - SKIP
    absent = [n for n in order + EMBED if n not in on_disk]
    if absent:
        raise BundleError("ORDER/EMBED name modules that do not exist in local_llm/: "
                          + ", ".join(absent))
    missing = sorted(on_disk - set(order) - set(EMBED))
    if missing:
        raise BundleError(
            "these package modules are not in ORDER, so the single file would be "
            "missing them: " + ", ".join(missing) + ". Add each one to ORDER in "
            "bundle.py after the modules it imports (above the extension-modules "
            "marker unless api/cli/selftest import it).")
    return {name: Module(name, (PKG / f"{name}.py").read_text(encoding="utf-8"))
            for name in order}


def resolve_order(order: list[str], modules: dict[str, Module]) -> tuple[list[str], list[str]]:
    """ORDER, with any module moved up to just before its first importer.

    ORDER stays the authority for everything it already gets right; a module
    listed after a module that imports it at load time (easy to do when several
    features append at the same marker) is hoisted instead of producing a file
    whose top-level code dies on a NameError. Returns (order, notes).
    """
    position = {name: i for i, name in enumerate(order)}
    placed: set[str] = set()
    visiting: list[str] = []
    result: list[str] = []
    notes: list[str] = []

    def visit(name: str) -> None:
        if name in placed:
            return
        if name in visiting:
            cycle = visiting[visiting.index(name):] + [name]
            raise BundleError("import cycle at module load: " + " -> ".join(cycle))
        visiting.append(name)
        for dep in modules[name].deps:
            if dep not in modules:
                kind = "embedded as text" if dep in EMBED else "not in ORDER"
                raise BundleError(f"local_llm/{name}.py imports .{dep}, which is {kind}")
            if dep not in placed and position[dep] > position[name]:
                notes.append(f"moved {dep} before {name}, which imports it; "
                             f"list it earlier in ORDER")
            visit(dep)
        visiting.pop()
        placed.add(name)
        result.append(name)

    for name in order:
        visit(name)
    return result, notes


def plan_renames(order: list[str],
                 modules: dict[str, Module]) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Which module-level names to rename, per module, so one namespace works.

    In the package each module has its own globals; in the single file they all
    share one, so a name bound by two modules with different meanings silently
    resolves to whichever module came last, for both of them. Two that bit:
    config.py's `import datetime` (the module) against everyone else's
    `from datetime import datetime` (the class), and the _FENCE regex that
    documents.py and taskstate.py both define for different fence syntaxes.
    The meaning that is exported (in __all__, or imported by name elsewhere)
    keeps the name, else the one most modules share, else the first; every
    other module's copy is renamed throughout that module. Two different
    exported meanings cannot both survive and are an error. Returns
    (renames per module, warnings).
    """
    warnings: list[str] = []
    for module in modules.values():
        for node, _ in module.relative:
            if node.module:
                target = modules.get(node.module)
                if target is not None:
                    target.exports.update(a.name for a in node.names if a.name != "*")
    import builtins
    builtin_names = set(dir(builtins))
    first = {name: i for i, name in enumerate(order)}
    renames: dict[str, dict[str, str]] = {name: {} for name in order}
    by_name: dict[str, dict[tuple, list[str]]] = {}
    for name in order:
        for bound, meaning in modules[name].bindings.items():
            if bound in RESERVED:
                raise BundleError(f"local_llm/{name}.py binds {bound}, a name the "
                                  "single-file build reserves for itself")
            by_name.setdefault(bound, {}).setdefault(meaning, []).append(name)
    for bound, meanings in by_name.items():
        if bound in builtin_names:
            # Shadowing a builtin at module level shadows it for EVERY module in
            # the single file; treat the builtin as one more meaning to keep.
            readers = [n for n in order if bound in modules[n].loads
                       and bound not in modules[n].bindings]
            if readers:
                meanings = dict(meanings)
                meanings[("builtin", bound)] = readers
        if len(meanings) < 2:
            continue
        exported = [m for m, owners in meanings.items()
                    if m[0] != "builtin" and any(bound in modules[o].exports for o in owners)]
        if len(exported) > 1:
            owners = "; ".join(f"{', '.join(meanings[m])} ({m[0]})" for m in exported)
            raise BundleError(f"{bound!r} is exported with different meanings by "
                              f"{owners}. The single file has one namespace: rename "
                              "one of them in the package")
        if exported:
            keep = exported[0]
            if ("builtin", bound) in meanings:
                warnings.append(f"{', '.join(meanings[keep])} export {bound!r}, which "
                                f"shadows the builtin for every module in the single "
                                f"file, including {', '.join(meanings[('builtin', bound)])}")
        elif ("builtin", bound) in meanings:
            keep = ("builtin", bound)
        else:
            keep = max(meanings, key=lambda m: (len(meanings[m]), -min(first[o] for o in meanings[m])))
        for meaning, owners in meanings.items():
            if meaning == keep or meaning[0] == "builtin":
                continue
            for owner in owners:
                renames[owner][bound] = f"{RENAME_PREFIX}{owner}_{bound}"
    return renames, warnings


# --------------------------------------------------------------------------- #
# Rewriting one module
# --------------------------------------------------------------------------- #

class _Renamer(ast.NodeVisitor):
    """Collect byte-range edits that rename module-level names throughout a module.

    Renames every Name node, `global`/`nonlocal` declaration and def/class
    header that refers to a renamed identifier. Anything the rename cannot make
    consistent (a parameter, an `except ... as` target or a class attribute of
    the same name) is refused rather than half-renamed.
    """

    def __init__(self, module: Module, rename: dict[str, str]) -> None:
        self.module = module
        self.rename = rename
        self.scope = ["module"]
        self.in_fstring = 0
        self.edits: dict[int, list[tuple[int, int, str]]] = {}

    def _refuse(self, node: ast.AST, name: str, what: str) -> None:
        raise BundleError(
            f"{self.module.where(node)}: {name!r} collides with another module's "
            f"{name!r} in the single file and must be renamed, but it is also {what} "
            "here, which a rename cannot follow. Rename it in the package.")

    def _edit(self, lineno: int, start: int, end: int, text: str) -> None:
        if self.in_fstring and sys.version_info < (3, 12):
            raise BundleError(f"local_llm/{self.module.name}.py:{lineno}: renaming "
                              "inside an f-string needs Python 3.12+ to run bundle.py")
        self.edits.setdefault(lineno, []).append((start, end, text))

    def _edit_word(self, node: ast.AST, pattern: str, name: str) -> None:
        line = self.module.lines[node.lineno - 1].encode("utf-8")
        end = node.end_col_offset if node.end_lineno == node.lineno else len(line)
        match = re.compile(pattern.encode()).search(line, node.col_offset, end)
        if match is None:
            raise BundleError(f"{self.module.where(node)}: could not find {name!r} to rename")
        self._edit(node.lineno, match.start(1), match.end(1), self.rename[name])

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.rename:
            if not isinstance(node.ctx, ast.Load) and self.scope[-1] == "class":
                self._refuse(node, node.id, "a class attribute")
            self._edit(node.lineno, node.col_offset, node.end_col_offset, self.rename[node.id])

    def _visit_def(self, node) -> None:
        if node.name in self.rename:
            if self.scope[-1] == "class":
                self._refuse(node, node.name, "a method or nested class")
            keyword = "class" if isinstance(node, ast.ClassDef) else "def"
            self._edit_word(node, rf"\b{keyword}\s+({re.escape(node.name)})\b", node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                self.visit(base)
            for keyword_arg in node.keywords:
                self.visit(keyword_arg.value)
            self.scope.append("class")
        else:
            self.visit(node.args)
            if node.returns is not None:
                self.visit(node.returns)
            self.scope.append("function")
        for statement in node.body:
            self.visit(statement)
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _visit_def

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit(node.args)
        self.scope.append("function")
        self.visit(node.body)
        self.scope.pop()

    def _visit_comprehension(self, node) -> None:
        self.scope.append("function")
        self.generic_visit(node)
        self.scope.pop()

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _visit_comprehension

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self.in_fstring += 1
        self.generic_visit(node)
        self.in_fstring -= 1

    def visit_arg(self, node: ast.arg) -> None:
        if node.arg in self.rename:
            self._refuse(node, node.arg, "a parameter name")
        if node.annotation is not None:
            self.visit(node.annotation)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name in self.rename:
            self._refuse(node, node.name, "an `except ... as` target")
        self.generic_visit(node)

    def _visit_declaration(self, node) -> None:
        for name in node.names:
            if name in self.rename:
                self._edit_word(node, rf"\b({re.escape(name)})\b", name)

    visit_Global = visit_Nonlocal = _visit_declaration

    def _visit_capture(self, node) -> None:
        for attr in ("name", "rest"):
            if getattr(node, attr, None) in self.rename:
                self._refuse(node, getattr(node, attr), "a match-case capture")
        self.generic_visit(node)

    visit_MatchAs = visit_MatchStar = visit_MatchMapping = _visit_capture

    def _visit_import(self, node) -> None:
        bound = [(a.asname or a.name.split(".")[0]) for a in node.names]
        if any(b in self.rename for b in bound) and self.scope[-1] == "class":
            self._refuse(node, next(b for b in bound if b in self.rename),
                         "imported into a class body")

    visit_Import = visit_ImportFrom = _visit_import


def _import_text(module: Module, node, rename: dict[str, str]) -> str:
    """An absolute import statement, regenerated on one line with renames applied."""
    if isinstance(node, ast.Import):
        parts = []
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            new = rename.get(bound, alias.asname)
            if bound in rename and not alias.asname and "." in alias.name:
                raise BundleError(f"{module.where(node)}: cannot rename the package bound "
                                  f"by `import {alias.name}`; write `import "
                                  f"{alias.name.split('.')[0]}` separately")
            parts.append(alias.name + (f" as {new}" if new and new != alias.name else ""))
        return "import " + ", ".join(parts)
    parts = []
    for alias in node.names:
        new = rename.get(alias.asname or alias.name, alias.asname)
        parts.append(alias.name + (f" as {new}" if new and new != alias.name else ""))
    return f"from {node.module} import " + ", ".join(parts)


def _relative_text(node: ast.ImportFrom, module_level: bool,
                   rename: dict[str, str]) -> str:
    """What replaces a relative import: assignments to the names it bound."""
    statements = []
    for alias in node.names:
        if alias.name == "*":
            continue
        target = rename.get(alias.asname or alias.name, alias.asname or alias.name)
        if node.module is None:          # from . import X [as Y]
            statements.append(f'{target} = _BUNDLED_MODULES["{alias.name}"]')
        elif module_level:
            # The name is already a global of the one namespace; only an alias
            # (or a rename) needs a new binding.
            if target != alias.name:
                statements.append(f"{target} = {alias.name}")
        else:
            # Inside a function `x = x` would be an UnboundLocalError: read the
            # global explicitly.
            statements.append(f'{target} = globals()["{alias.name}"]')
    return "; ".join(statements)


def render_module(module: Module, rename: dict[str, str]) -> tuple[str, list[str]]:
    """Return (body, hoisted_import_lines) for one module.

    Drops the module docstring, __all__, `if __name__ == "__main__":` blocks
    (the bundle's footer is its only entry point; cli.py's guard used to start
    the server a second time when the first one exited) and every intra-package
    import, which becomes either nothing or the assignment it implied.
    """
    lines = module.lines
    raw = [line.encode("utf-8") for line in lines]
    drop: set[int] = set()                  # 1-indexed lines to remove
    replace: dict[int, str] = {}            # 1-indexed first line -> new text
    hoisted: list[str] = []

    def whole_lines(node) -> bool:
        head = raw[node.lineno - 1][:node.col_offset]
        tail = raw[node.end_lineno - 1][node.end_col_offset:].strip()
        return not head.strip() and (not tail or tail.startswith(b"#"))

    def substitute(node, text: str) -> None:
        if not whole_lines(node):
            raise BundleError(f"{module.where(node)}: put this import on its own line "
                              "so the bundler can rewrite it")
        span = range(node.lineno, node.end_lineno + 1)
        drop.update(span)
        if text:
            indent = raw[node.lineno - 1][:node.col_offset].decode("utf-8")
            replace[node.lineno] = indent + text
            drop.discard(node.lineno)

    body = list(module.tree.body)
    handled: set[int] = set()
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        drop.update(range(body[0].lineno, body[0].end_lineno + 1))
    for node in body:
        span = range(node.lineno, node.end_lineno + 1)
        if _is_main_guard(node):
            drop.update(span)
            handled.update(id(n) for n in ast.walk(node))
        elif _is_all(node):
            start = node.lineno
            while start > 1 and lines[start - 2].startswith("#"):   # its comment too
                start -= 1
            drop.update(range(start, node.end_lineno + 1))
        elif isinstance(node, ast.ImportFrom) and node.level:
            handled.add(id(node))
            substitute(node, _relative_text(node, True, rename))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            handled.add(id(node))
            drop.update(span)
            if not (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
                hoisted.append(_import_text(module, node, rename))

    # Imports nested in functions or in top-level try/if blocks stay where they
    # are, rewritten: relative ones into assignments, absolute ones only when
    # they bind a renamed name.
    module_level = {id(node) for node, at_top in module.relative if at_top}
    for node in ast.walk(module.tree):
        if id(node) in handled:
            continue
        if isinstance(node, ast.ImportFrom) and node.level:
            substitute(node, _relative_text(node, id(node) in module_level, rename) or "pass")
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and rename and any(
                (a.asname or a.name.split(".")[0]) in rename for a in node.names):
            substitute(node, _import_text(module, node, rename))

    if rename:
        renamer = _Renamer(module, rename)
        renamer.visit(module.tree)
        for lineno, edits in renamer.edits.items():
            if lineno in drop or lineno in replace:
                continue
            line = raw[lineno - 1]
            for start, end, text in sorted(set(edits), reverse=True):
                line = line[:start] + text.encode("utf-8") + line[end:]
            raw[lineno - 1] = line

    out = []
    for i, line in enumerate(raw, 1):
        if i in replace:
            out.append(replace[i])
        elif i not in drop:
            out.append(line.decode("utf-8"))
    return "\n".join(out).strip("\n"), hoisted


def merge_imports(lines: list[str]) -> list[str]:
    """Collapse the collected import lines into one statement per module.

    Modules import only what they use, so the same module arrives here through
    several different lines ("from dataclasses import dataclass" from one file,
    "from dataclasses import dataclass, field" from another). De-duplicating by
    line text alone would emit all of them: correct, but three redundant imports
    where one belongs. Union the names per module instead.
    """
    plain: list[str] = []                       # "import x" / "import x as y"
    froms: dict[str, list[str]] = {}            # module -> imported names
    for line in lines:
        node = ast.parse(line.strip()).body[0]
        if isinstance(node, ast.Import):
            for alias in node.names:
                label = ("import " + alias.name
                         + (f" as {alias.asname}" if alias.asname else ""))
                if label not in plain:
                    plain.append(label)
        else:                                    # ast.ImportFrom
            module = "." * node.level + (node.module or "")
            names = froms.setdefault(module, [])
            for alias in node.names:
                label = alias.name + (f" as {alias.asname}" if alias.asname else "")
                if label not in names:
                    names.append(label)
    merged = sorted(plain)
    merged += [f"from {module} import " + ", ".join(sorted(names))
               for module, names in sorted(froms.items())]
    return merged


def _namespace_shim(module: Module, rename: dict[str, str]) -> str:
    """The module object that `from . import <module>` elsewhere expects."""
    names = sorted((set(module.bindings) | module.exports | module.reexports)
                   - {"__all__"})
    renamed = {n: rename[n] for n in names if n in rename}
    listing = ",\n".join(f"    {n!r}" for n in names)
    return (f"# `from . import {module.name}` reads attributes off a module object, and\n"
            f"# the single file has none: give it one holding the same names.\n"
            f"_BUNDLED_MODULES[{module.name!r}] = _bundled_namespace([\n{listing},\n"
            f"], {renamed!r})")


PREAMBLE = '''\
# Package files read as text at run time (see EMBED in bundle.py).
_BUNDLED_SOURCES = {sources}

# Module objects for `from . import X` (see _namespace_shim in bundle.py).
_BUNDLED_MODULES: dict = {{}}


def _bundled_namespace(names, renamed):
    import types as _bundled_types
    scope = globals()
    return _bundled_types.SimpleNamespace(**{{
        name: scope[renamed.get(name, name)]
        for name in names if renamed.get(name, name) in scope}})'''


def build(order: list[str] | None = None) -> tuple[str, list[str]]:
    """Return (single_file_source, notes) without writing anything."""
    order = list(ORDER if order is None else order)
    modules = load_modules(order)
    order, notes = resolve_order(order, modules)
    renames, warnings = plan_renames(order, modules)
    notes += [f"warning: {w}" for w in warnings]
    for name in order:
        for old, new in sorted(renames[name].items()):
            notes.append(f"renamed {name}.{old} -> {new} (collides with another module)")
    namespaces = {alias.name for m in modules.values() for node, _ in m.relative
                  if node.module is None for alias in node.names}
    seen_imports: list[str] = []
    chunks: list[str] = []
    for name in order:
        body, imports = render_module(modules[name], renames[name])
        for imp in imports:
            if imp not in seen_imports:
                seen_imports.append(imp)
        chunk = f"# {'=' * 70}\n# {name}\n# {'=' * 70}\n\n{body}"
        if name in namespaces:
            chunk += "\n\n" + _namespace_shim(modules[name], renames[name])
        chunks.append(chunk)
    sources = {f"{name}.py": (PKG / f"{name}.py").read_text(encoding="utf-8")
               for name in EMBED}
    header = (
        "#!/usr/bin/env python3\n"
        '"""Local LLM: single-file build.\n\n'
        "Generated by bundle.py from the local_llm package. Edit the package, not\n"
        "this file; re-run `python3 bundle.py` to rebuild.\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
        + "\n".join(merge_imports(seen_imports)) + "\n\n\n"
        # repr() keeps each embedded file on ONE line, so none of its lines
        # (its own __future__ import, its docstrings) appears as a line of the
        # bundle; selftest counts __future__ lines to spot a duplicated file.
        + PREAMBLE.format(sources=repr(sources)) + "\n\n\n"
    )
    footer = '\n\nif __name__ == "__main__":\n    main()\n'
    text = header + "\n\n\n".join(chunks) + footer
    try:
        compile(text, str(OUT), "exec")
    except SyntaxError as exc:
        raise BundleError(f"the generated file does not compile: {exc}") from None
    return text, notes


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0].startswith("-"):
        print(__doc__)
        raise SystemExit(0 if argv[0] in ("-h", "--help") else 2)
    out = Path(argv[0]).resolve() if argv else OUT
    text, notes = build()
    for note in notes:
        print(f"note: {note}")
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({text.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
