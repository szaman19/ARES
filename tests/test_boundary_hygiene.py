"""Strict boundary hygiene tests verifying core ARES v2 dependencies and symbols."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

FORBIDDEN_MODULES = {
    "ares.ext",
    "duckdb",
    "sql",
    "sqlite3",
    "httpx",
    "openai",
    "google.genai",
    "google",
    "fastapi",
    "matplotlib",
    "numpy",
    "pyarrow",
    "uvicorn",
}

FORBIDDEN_SYMBOLS = {
    "InvestigationBoard",
    "SearchQuestion",
    "DuckDBBackend",
    "Domain",
}

CORE_FILES = [
    "__init__.py",
    "types.py",
    "store.py",
    "machine.py",
    "cli.py",
]


def _collect_imports(tree: ast.AST) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
    return imported


def test_core_ares_ast_has_no_forbidden_imports_or_symbols() -> None:
    ares_dir = Path(__file__).resolve().parent.parent / "ares"
    present_core_files = [ares_dir / name for name in CORE_FILES if (ares_dir / name).exists()]
    assert len(present_core_files) == len(CORE_FILES)

    for py_file in present_core_files:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        imports = _collect_imports(tree)

        for mod in imports:
            top = mod.split(".", 1)[0]
            for forbidden in FORBIDDEN_MODULES:
                assert mod != forbidden and not mod.startswith(forbidden + "."), (
                    f"{py_file.name} imports forbidden {mod}"
                )
            assert top not in FORBIDDEN_MODULES, f"{py_file.name} imports forbidden {top}"
            if top != "ares" and top != "pydantic":
                assert top in sys.stdlib_module_names, (
                    f"{py_file.name} imports non-stdlib external dependency {mod}"
                )

        for symbol in FORBIDDEN_SYMBOLS:
            assert symbol not in source, f"{py_file.name} references forbidden symbol {symbol}"


def test_subprocess_import_ares_loads_no_forbidden_modules() -> None:
    script = (
        "import json, sys, ares; "
        "mods = sorted(sys.modules.keys()); "
        "print(json.dumps(mods))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded_modules = set(proc.stdout.strip().strip("[]").replace('"', "").split(", "))
    for forbidden in FORBIDDEN_MODULES:
        assert forbidden not in loaded_modules
        assert not any(m.startswith(forbidden + ".") for m in loaded_modules)
