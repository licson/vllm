# vLLM SM12x Docker Build

This orphan branch contains the Docker build configuration for vLLM optimized for **SM12x (SM120/SM121)** consumer Blackwell GPUs (RTX Pro 6000, GB10 / DGX Spark).

## What's in this branch

- `Dockerfile` — Multi-stage build that compiles vLLM `pr-ports` with native SM12x kernels.
- `README.md` — This file.

## Architecture

The Dockerfile uses a **multi-stage wheel-based** design modeled after `licson/sglang` and `eugr/spark-vllm-docker`:

1. **`base`** — CUDA 13.0.1 + Ubuntu 24.04 + system deps + ccache + uv + Rust
2. **`torch_deps`** — PyTorch cu130 + build tools
3. **Parallel builders** (each with `repo-cache` and `ccache` mounts):
   - `deepep_builder` — DeepEP with timeout tweaks
   - `flashinfer_builder` — FlashInfer with Blackwell PRs (3174, 3180)
   - `deepgemm_builder` — DeepGEMM `sm120` branch (DeepGEMM#324)
   - `vllm_builder` — vLLM `pr-ports` branch
4. **`framework`** — Assemble & install all wheels, apply CUTLASS SM121 patch, Triton ptxas fix
5. **`runtime`** — Clean image with only runtime libs and installed packages (no source code)

## Key optimizations included

1. **Native SM12x cubins** — `TORCH_CUDA_ARCH_LIST="12.0;12.1"` (no PTX JIT fallback).
2. **CUTLASS SM121 patch** — Fixes `nvidia-cutlass-dsl` `warp/mma.py` to accept `sm_121a`.
3. **DeepGEMM#324** — Builds `deepseek-ai/DeepGEMM@sm120` for native SM120 grouped/dense GEMM and attention kernels.
4. **FlashInfer b12x** — Built from source with Blackwell PRs merged, arch `12.1a`.
5. **DeepEP** — Expert parallelism library with SM12x-compatible timeout configs.
6. **Marlin native** — Native `sm_120`/`sm_121` cubins for MoE Marlin (no JIT gibberish).

## Build

```bash
docker build \
  --build-arg VLLM_BRANCH=pr-ports \
  --build-arg DEEPGEMM_BRANCH=sm120 \
  --build-arg FI_PR_NUMBERS="3174 3180" \
  -t vllm-sm12x:latest \
  -f Dockerfile .
```

### Build args

| Arg | Default | Description |
|-----|---------|-------------|
| `CUDA_VERSION` | `13.0.1` | CUDA base image version |
| `VLLM_BRANCH` | `pr-ports` | vLLM git branch to build |
| `DEEPGEMM_BRANCH` | `sm120` | DeepGEMM git branch to build |
| `FI_PR_NUMBERS` | `3174 3180` | Space-separated FlashInfer PRs to merge |
| `MAX_JOBS` | `8` | Build parallelism |
| `TORCH_CUDA_ARCH_LIST` | `12.0;12.1` | CUDA archs for vLLM/DeepEP/DeepGEMM |

## Run (DeepSeek-V4-Flash example)

```bash
docker run --gpus all --rm -it \
  -v /models:/models \
  -p 8000:8000 \
  vllm-sm12x:latest \
  vllm serve /models/DeepSeek-V4-Flash \
    --tensor-parallel-size 2 \
    --moe-backend deep_gemm \
    --kv-cache-dtype fp8_ds_mla \
    --gpu-memory-utilization 0.78 \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
    --trust-remote-code
```

## Notes

- The `pr-ports` branch already contains merged PRs #40923, #41834, #40082, #41062, #41028, and b12x auto-selection re-enablement.
- This branch is intentionally **orphan** (no shared history with `main` or `pr-ports`) to keep the Docker build context lightweight.
- **ccache** and **repo-cache** BuildKit mounts are used for fast incremental rebuilds. Ensure `DOCKER_BUILDKIT=1` is set.
