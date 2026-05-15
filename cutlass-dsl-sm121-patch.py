#!/usr/bin/env python3
"""Patch CUTLASS DSL 4.4.2 to accept sm_121a for warp-level MMA operations."""

import importlib.util
import pathlib
import sys

spec = importlib.util.find_spec("cutlass")
if not spec or not spec.origin:
    print("cutlass package not found", file=sys.stderr)
    sys.exit(1)

mma_path = pathlib.Path(spec.origin).parent / "cute" / "nvgpu" / "warp" / "mma.py"
if not mma_path.exists():
    print(f"{mma_path} not found", file=sys.stderr)
    sys.exit(1)

txt = mma_path.read_text()

old_guard = '''    admissible_archs = [
        "sm_120a",
    ]'''

new_guard = '''    admissible_archs = [
        "sm_120a",
        "sm_121a",
    ]'''

old_check = "if not arch == Arch.sm_120a:"
new_check = "if arch not in (Arch.sm_120a, Arch.sm_121a):"

if old_guard not in txt:
    print("mma.py already patched or unexpected content", file=sys.stderr)
    sys.exit(0)

txt = txt.replace(old_guard, new_guard)
txt = txt.replace(old_check, new_check)
mma_path.write_text(txt)
print(f"Patched {mma_path} for SM121a support")
