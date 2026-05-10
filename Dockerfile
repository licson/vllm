# SPDX-License-Identifier: Apache-2.0
# Multi-stage Dockerfile for vLLM on SM12x (SM120/SM121) consumer Blackwell.

ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
# Stage 1: Base builder image with CUDA dev tools
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG PYTHON_VERSION
ARG VLLM_BRANCH=pr-ports
ARG DEEPGEMM_BRANCH=sm120
ARG TORCH_CUDA_ARCH_LIST="12.0;12.1"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    ninja-build \
    libnccl2 \
    libnccl-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Install Python + torch
RUN uv python install ${PYTHON_VERSION} \
    && uv venv /opt/vllm-venv --python ${PYTHON_VERSION}
ENV PATH="/opt/vllm-venv/bin:${PATH}"

# Install PyTorch with CUDA 12.8 (matches base image)
RUN uv pip install --system torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Install build-time Python deps
RUN uv pip install --system \
    setuptools setuptools-scm wheel packaging \
    ninja numpy psutil ray pyarrow \
    "transformers>=4.48.0" \
    "accelerate>=1.0.0" \
    "flashinfer-python>=0.2.3" \
    sentencepiece protobuf \
    fastapi uvicorn openai

# ---------------------------------------------------------------------------
# Stage 1a: Patch nvidia-cutlass-dsl for SM121a support
# ---------------------------------------------------------------------------
RUN uv pip install --system "nvidia-cutlass-dsl==4.4.2"
# The 4.4.2 wheel hard-codes sm_120a but rejects sm_121a. Patch it.
RUN CUTE_DSL_MMA_PY=$(python -c \
    "import nvidia_cutlass_dsl; print(nvidia_cutlass_dsl.__path__[0])")/cute/nvgpu/warp/mma.py \
    && sed -i 's/sm_120a/sm_121a/g' "$CUTE_DSL_MMA_PY" \
    && grep -q sm_121a "$CUTE_DSL_MMA_PY" \
    && echo "CUTLASS SM121 patch applied OK"

# ---------------------------------------------------------------------------
# Stage 1b: Build DeepGEMM#324 (SM120 native kernels)
# ---------------------------------------------------------------------------
WORKDIR /workspace
RUN git clone --depth 1 --branch ${DEEPGEMM_BRANCH} \
    https://github.com/deepseek-ai/DeepGEMM.git deepgemm

WORKDIR /workspace/deepgemm
RUN python setup.py develop \
    && python tests/test_fp8_fp4.py --quick \
    && echo "DeepGEMM build OK"

# ---------------------------------------------------------------------------
# Stage 1c: Build vLLM pr-ports with native SM12x cubins
# ---------------------------------------------------------------------------
WORKDIR /workspace
RUN git clone --depth 1 --branch ${VLLM_BRANCH} \
    https://github.com/licson/vllm.git vllm

WORKDIR /workspace/vllm
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV MAX_JOBS=8
ENV NVCC_THREADS=2

# Install vLLM in editable mode with C++ extensions compiled for SM12x
RUN uv pip install --system -e . \
    --no-build-isolation \
    -v

# Verify native cubins were emitted for Marlin-MoE
RUN cuobjdump --list-elf /workspace/vllm/vllm/*.so 2>/dev/null | grep -c sm_12 || true
RUN cuobjdump --list-elf /workspace/vllm/vllm/*.so 2>/dev/null | grep -c sm_121 || true

# ---------------------------------------------------------------------------
# Stage 2: Runtime image (smaller, no build tools)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libnccl2 \
    libnccl-dev \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment from builder
COPY --from=builder /opt/vllm-venv /opt/vllm-venv
COPY --from=builder /workspace/vllm /workspace/vllm
COPY --from=builder /workspace/deepgemm /workspace/deepgemm
ENV PATH="/opt/vllm-venv/bin:${PATH}"

# Ensure DeepGEMM Python bindings are on PYTHONPATH
ENV PYTHONPATH="/workspace/deepgemm:${PYTHONPATH}"

# SM12x runtime tuning defaults
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV NCCL_MIN_NCHANNELS=32

WORKDIR /workspace/vllm
EXPOSE 8000

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
