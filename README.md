# vLLM SM12x Docker Build

This orphan branch contains the Docker build configuration for vLLM optimized for **SM12x (SM120/SM121)** consumer Blackwell GPUs (RTX Pro 6000, GB10 / DGX Spark).

## What's in this branch

- `Dockerfile` — Multi-stage build that compiles vLLM `pr-ports` with native SM12x kernels.
- `patches/` — Runtime patches applied inside the container build.
- `README.md` — This file.

## Key optimizations included

1. **Native SM12x cubins** — `TORCH_CUDA_ARCH_LIST="12.0;12.1"` (no PTX JIT fallback).
2. **CUTLASS SM121 patch** — Fixes `nvidia-cutlass-dsl` `warp/mma.py` to accept `sm_121a`.
3. **DeepGEMM#324** — Builds `leavelet/DeepGEMM@sm120` for native SM120 grouped/dense GEMM and attention kernels.
4. **FlashInfer b12x** — Pre-installed so `FlashInferB12xExperts` auto-selects on SM12x.
5. **Marlin native** — Native `sm_120`/`sm_121` cubins for MoE Marlin (no JIT gibberish).

## Build

```bash
docker build \
  --build-arg VLLM_BRANCH=pr-ports \
  --build-arg DEEPGEMM_BRANCH=sm120 \
  -t vllm-sm12x:latest \
  -f Dockerfile .
```

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
