"""Static input-confinement audit of the TCB (spec section 2.2).

Mechanically enforces the closed-world execution policy over every module in
clean_runtime/: imports must come from a fixed whitelist and no ambient-
authority builtin may be referenced. gateway.py is the sole module allowed to
perform file I/O (it is the mediation point); attest.py may import the
signing library. This audit is cited in the paper as evidence that the
input-closure obligation is auditable by machine, not by trust.
"""
import ast
import pathlib

RUNTIME = pathlib.Path(__file__).resolve().parent.parent / "clean_runtime"

IMPORT_WHITELIST = {"numpy", "hashlib", "hmac", "struct", "json", "types"}
EXTRA_IMPORTS = {
    "attest.py": {"cryptography"},
    "gateway.py": {"pathlib"},  # file access is confined to the gateway
}
FORBIDDEN_BUILTINS = {
    "open", "eval", "exec", "compile", "__import__", "globals", "locals",
    "input", "breakpoint", "vars", "getattr", "setattr", "delattr",
}
EXTRA_BUILTINS = {
    "gateway.py": {"open"},  # sole mediation point: file I/O allowed here
}

# Raw block bytes may be touched only where they are verified. Confining the
# untrusted backend and the untrusted transfer path to store.py (which checks
# every fetch against the commitment taken at admission), device.py (which
# owns the boundary) and gateway.py (which installs both) is what makes the
# verified read path unbypassable rather than merely conventional.
BOUNDARY_ATTRS = {"backend", "_backend", "path", "_path"}
BOUNDARY_MODULES = {"store.py", "gateway.py", "device.py"}

# numpy is whitelisted for arithmetic, but it also carries a filesystem API.
# Without this rule a module could reach the host through np.load/np.fromfile
# while importing nothing outside the whitelist and naming no forbidden
# builtin, which would make the "file access lives in the gateway alone"
# property false while the audit still passed.
NUMPY_FS = {
    "load", "save", "savez", "savez_compressed", "loadtxt", "savetxt",
    "genfromtxt", "fromfile", "tofile", "memmap", "fromregex", "DataSource",
    "lib",  # np.lib.format / np.lib.npyio reach the same functions
}
NUMPY_FS_MODULES = {"gateway.py"}  # sole mediation point


def _numpy_aliases(tree):
    """Names that numpy is bound to in this module (np, numpy, ...)."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "numpy":
                    out.add(alias.asname or alias.name.split(".")[0])
    return out


def _audit(path: pathlib.Path):
    tree = ast.parse(path.read_text())
    allowed_imports = IMPORT_WHITELIST | EXTRA_IMPORTS.get(path.name, set())
    allowed_builtins = EXTRA_BUILTINS.get(path.name, set())
    np_names = _numpy_aliases(tree)
    fs_ok = path.name in NUMPY_FS_MODULES
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in allowed_imports:
                    violations.append(f"{path.name}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:  # absolute import; relative (intra-TCB) is fine
                root = (node.module or "").split(".")[0]
                if root not in allowed_imports:
                    violations.append(f"{path.name}:{node.lineno} from {node.module}")
                elif root == "numpy" and not fs_ok:
                    for alias in node.names:
                        if alias.name in NUMPY_FS:
                            violations.append(
                                f"{path.name}:{node.lineno} imports the numpy "
                                f"filesystem API ({alias.name}) outside the gateway")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_BUILTINS and node.id not in allowed_builtins:
                violations.append(f"{path.name}:{node.lineno} use of {node.id}")
        elif isinstance(node, ast.Attribute):
            if (node.attr in NUMPY_FS and not fs_ok
                    and isinstance(node.value, ast.Name)
                    and node.value.id in np_names):
                violations.append(
                    f"{path.name}:{node.lineno} reaches the numpy filesystem "
                    f"API ({node.value.id}.{node.attr}) outside the gateway")
            if (node.attr in BOUNDARY_ATTRS
                    and path.name not in BOUNDARY_MODULES):
                violations.append(
                    f"{path.name}:{node.lineno} reaches the untrusted backend "
                    f"or transfer path outside the verified read path")
    return violations


def test_runtime_is_input_closed():
    violations = []
    for path in sorted(RUNTIME.glob("*.py")):
        violations += _audit(path)
    assert violations == [], "confinement violations:\n" + "\n".join(violations)


def test_runtime_modules_exist():
    names = {p.name for p in RUNTIME.glob("*.py")}
    assert {"gateway.py", "rng.py", "commit.py", "store.py",
            "device.py"} <= names


def test_audit_rejects_a_planted_violation(tmp_path):
    """The audit is only evidence if it fails on code that breaks the policy."""
    planted = tmp_path / "kernel.py"
    planted.write_text("import numpy as np\n\n\ndef f(p):\n    return np.load(p)\n")
    assert _audit(planted), "audit passed code that reaches the filesystem"
