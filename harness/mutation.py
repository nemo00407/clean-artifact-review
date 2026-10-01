"""E10 - Mutation testing of the confinement audit (non-TCB harness).

The audit in `tests/test_confinement.py` is what makes Assumption 1 checkable
by machine rather than by trust. An audit is only evidence if it fails on
code that breaks the policy, so this measures it: each mutant is a runtime
module carrying exactly one violation, and we report which classes the audit
catches and which it does not.

The misses are reported alongside the catches deliberately. The audit is a
conservative syntactic check over the runtime's own modules; it reasons about
neither dynamic rebinding nor attribute chains that name no forbidden
identifier, and saying so is worth more than a catch rate that hides it.
"""
import json
import pathlib
import sys
import tempfile

from . import common

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tests.test_confinement import _audit  # noqa: E402

BASE = "import numpy as np\n\n\ndef f(h, p):\n    return %s\n"

# (class, module the mutant is planted in, source, expected to be caught)
MUTANTS = [
    ("forbidden import", "kernel.py",
     "import os\n\n\ndef f():\n    return os.getcwd()\n", True),
    ("forbidden from-import", "kernel.py",
     "from socket import socket\n\n\ndef f():\n    return socket()\n", True),
    ("ambient builtin: eval", "kernel.py",
     "def f(s):\n    return eval(s)\n", True),
    ("ambient builtin: getattr", "kernel.py",
     "def f(o, n):\n    return getattr(o, n)\n", True),
    ("file open outside the gateway", "kernel.py",
     "def f(p):\n    return open(p).read()\n", True),
    ("array-library filesystem API", "kernel.py", BASE % "np.load(p)", True),
    ("array-library FS via from-import", "kernel.py",
     "from numpy import load\n\n\ndef f(p):\n    return load(p)\n", True),
    ("untrusted backend outside the read path", "kernel.py",
     BASE % "h._backend.get('k')", True),
    ("storage path outside the boundary", "kernel.py",
     BASE % "h._path.transfer(b'')", True),
    # Known misses. Both name no forbidden identifier and import nothing
    # outside the whitelist, so a syntactic whitelist cannot see them.
    ("attribute-chain escape", "kernel.py",
     "def f():\n    return ().__class__.__bases__[0].__subclasses__()\n",
     False),
    ("alias rebinding of the array library", "kernel.py",
     "import numpy as np\n\n\ndef f(p):\n    g = np\n    return g.load(p)\n",
     False),
]


def run():
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for name, module, src, expected in MUTANTS:
            path = pathlib.Path(td) / module
            path.write_text(src)
            caught = bool(_audit(path))
            rows.append((name, caught, expected))
            flag = "caught" if caught else "MISSED"
            print(f"[{flag:6s}] {name}")
            assert caught == expected, (
                f"{name}: audit behaviour changed (caught={caught}, "
                f"documented={expected}); update the table, do not silence it")
    return rows


def main():
    rows = run()
    # The unmutated runtime must still pass, or a catch rate means nothing.
    runtime_dir = pathlib.Path(__file__).resolve().parents[1] / "clean_runtime"
    clean = sum(len(_audit(p)) for p in sorted(runtime_dir.glob("*.py")))

    caught = [r for r in rows if r[1]]
    missed = [r for r in rows if not r[1]]

    out = common.DATA_DIR.parent / "results"
    out.mkdir(exist_ok=True)
    lines = ["# E10 - Mutation testing of the confinement audit", "",
             f"Each mutant is a runtime module carrying exactly one policy "
             f"violation. The unmutated runtime reports **{clean}** "
             f"violations.", "",
             "| violation class | audit verdict |",
             "|---|---|"]
    for name, was_caught, _ in rows:
        lines.append(f"| {name} | {'caught' if was_caught else 'missed'} |")
    lines += ["",
              f"Caught {len(caught)} of {len(rows)} classes. The {len(missed)} "
              f"misses are recorded rather than removed from the suite: the "
              f"audit is a conservative syntactic check over the runtime's "
              f"own modules and does not reason about dynamic rebinding or "
              f"attribute chains that name no forbidden identifier. What it "
              f"establishes is that a change opening a second channel of the "
              f"caught classes stops at the build.", ""]
    (out / "mutation.md").write_text("\n".join(lines))

    dat = ["i caught label"]
    for i, (name, was_caught, _) in enumerate(rows):
        dat.append(f"{i} {1 if was_caught else 0} {name.replace(' ', '_')}")
    (out / "mutation.dat").write_text("\n".join(dat) + "\n")
    (out / "mutation.meta.json").write_text(json.dumps(
        {"total": len(rows), "caught": len(caught), "missed": len(missed),
         "clean_runtime_violations": clean,
         "missed_classes": [n for n, _, _ in missed]}, indent=1))
    print(f"-> {out / 'mutation.md'} ({len(caught)}/{len(rows)} caught, "
          f"clean runtime violations={clean})")


if __name__ == "__main__":
    main()
