#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Apply the CUTLASS SM121a workaround inside the Docker build.
#
# nvidia-cutlass-dsl 4.4.2 hard-codes sm_120a in its warp/mma.py whitelist
# but rejects sm_121a, causing PTX generation failures on GB10/DGX Spark.
# NVIDIA fixed this in CUTLASS PR #3082 but the fix is not yet in the
# 4.4.2 wheel. This script patches the installed package in-place.

set -euo pipefail

CUTE_DSL_MMA_PY=$(python -c "import nvidia_cutlass_dsl; print(nvidia_cutlass_dsl.__path__[0])")/cute/nvgpu/warp/mma.py

if [[ ! -f "$CUTE_DSL_MMA_PY" ]]; then
    echo "ERROR: Cannot find $CUTE_DSL_MMA_PY" >&2
    exit 1
fi

if grep -q 'sm_121a' "$CUTE_DSL_MMA_PY"; then
    echo "CUTLASS SM121 patch already applied."
    exit 0
fi

# Replace sm_120a with sm_121a (the codegen logic is identical for both)
sed -i 's/sm_120a/sm_121a/g' "$CUTE_DSL_MMA_PY"

# Verify
if grep -q 'sm_121a' "$CUTE_DSL_MMA_PY"; then
    echo "CUTLASS SM121 patch applied successfully."
else
    echo "ERROR: Patch failed." >&2
    exit 1
fi
