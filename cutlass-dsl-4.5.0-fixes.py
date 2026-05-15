#!/usr/bin/env python3
"""Fix internal nvidia-cutlass-dsl 4.5.0 bugs in the installed package."""

import os
import sys

BASE_DIR = "/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl"


def find_file(name, subpath):
    """Find a file by name under BASE_DIR containing subpath in its path."""
    for root, dirs, files in os.walk(BASE_DIR):
        rel = os.path.relpath(root, BASE_DIR)
        if subpath in rel.replace(os.sep, "/") and name in files:
            return os.path.join(root, name)
    return None


def patch_core_py(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Fix 1: OpResultList handling in _unpack_x_tuple
    old = (
        "            vals = get_leaves(t, loc=loc, ip=ip)\n"
        "            if not isinstance(vals, list):\n"
        "                vals = [vals]"
    )
    new = (
        "            vals = get_leaves(t, loc=loc, ip=ip)\n"
        "            if isinstance(vals, ir.OpResultList):\n"
        "                vals = list(vals)\n"
        "            elif not isinstance(vals, list):\n"
        "                vals = [vals]"
    )
    if old in content:
        content = content.replace(old, new)
        print("  [core.py] Patched OpResultList handling")
    else:
        print("  [core.py] OpResultList fix not applied (already fixed or different version?)")

    # Fix 2: Remove static_tile=None and static_coord=None from local_tile
    old = (
        "    return _cute_ir.local_tile(\n"
        "        input=input.value,  # type: ignore[attr-defined]\n"
        "        tile=tiler_val,\n"
        "        static_tile=None,\n"
        "        coord=coord_val,\n"
        "        static_coord=None,\n"
        "        proj=proj,\n"
        "        loc=loc,\n"
        "        ip=ip,\n"
        "    )"
    )
    new = (
        "    return _cute_ir.local_tile(\n"
        "        input=input.value,  # type: ignore[attr-defined]\n"
        "        tile=tiler_val,\n"
        "        coord=coord_val,\n"
        "        proj=proj,\n"
        "        loc=loc,\n"
        "        ip=ip,\n"
        "    )"
    )
    if old in content:
        content = content.replace(old, new)
        print("  [core.py] Patched local_tile static args")
    else:
        print("  [core.py] local_tile fix not applied (already fixed or different version?)")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def patch_experimental_init(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old = (
        "raise NotImplementedError(\n"
        '    "CuTe Experimental module is only supported on Cuda toolkit 13.1 and above!"\n'
        ")"
    )
    new = "# Module stubbed for compatibility with nvidia-cutlass-dsl 4.5.0"
    
    if old in content:
        content = content.replace(old, new)
        print("  [experimental/__init__.py] Patched NotImplementedError")
    else:
        print("  [experimental/__init__.py] NotImplementedError fix not applied (already fixed or different version?)")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    print("Applying nvidia-cutlass-dsl 4.5.0 fixes...")
    
    core_py = find_file("core.py", "cute")
    if not core_py:
        print("ERROR: Could not find cute/core.py under", BASE_DIR)
        sys.exit(1)
    print(f"Found {core_py}")
    patch_core_py(core_py)

    init_py = find_file("__init__.py", "cute/experimental")
    if not init_py:
        print("ERROR: Could not find cute/experimental/__init__.py under", BASE_DIR)
        sys.exit(1)
    print(f"Found {init_py}")
    patch_experimental_init(init_py)

    print("Done.")


if __name__ == "__main__":
    main()
