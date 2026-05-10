# syntax=docker/dockerfile:1.6
# SPDX-License-Identifier: Apache-2.0
# Multi-stage Dockerfile for vLLM on SM12x (SM120/SM121) consumer Blackwell.
# Modeled after licson/sglang forked-sglang-docker-build and eugr/spark-vllm-docker.

ARG CUDA_VERSION=13.0.1
ARG PYTHON_VERSION=3.12
ARG TORCH_CUDA_ARCH_LIST="12.0;12.1"
ARG FLASHINFER_CUDA_ARCH_LIST="12.1a"
ARG MAX_JOBS=8

# =============================================================================
# Base Stage: CUDA 13.0 + Ubuntu 24.04 + System Dependencies
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG MAX_JOBS

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

# GKE default paths
ENV PATH="${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Install system dependencies, native Python 3.12, and bootstrap pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    software-properties-common \
    netcat-openbsd \
    kmod \
    unzip \
    openssh-server \
    curl \
    wget \
    lsof \
    locales \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    patchelf \
    git-lfs \
    ccache \
    devscripts \
    debhelper \
    fakeroot \
    check \
    libsubunit0 \
    libsubunit-dev \
    gnupg2 \
    libopenmpi-dev \
    libnuma1 \
    libnuma-dev \
    numactl \
    libibverbs-dev \
    libibverbs1 \
    libibumad3 \
    librdmacm1 \
    libnl-3-200 \
    libnl-route-3-200 \
    libnl-route-3-dev \
    libnl-3-dev \
    ibverbs-providers \
    infiniband-diags \
    perftest \
    libgoogle-glog-dev \
    libgtest-dev \
    libjsoncpp-dev \
    libunwind-dev \
    libboost-all-dev \
    libssl-dev \
    libgrpc-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    pybind11-dev \
    libhiredis-dev \
    libcurl4-openssl-dev \
    libczmq4 \
    libczmq-dev \
    libfabric-dev \
    linux-libc-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && wget -q https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py --break-system-packages \
    && rm get-pip.py \
    && python3 -m pip config set global.break-system-packages true \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Locale setup
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Rust toolchain (required by some Python deps)
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal \
    && rustc --version && cargo --version

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Configure ccache
ENV PATH="/usr/lib/ccache:$PATH" \
    CCACHE_DIR=/root/.ccache \
    CCACHE_MAXSIZE=50G \
    CCACHE_COMPRESS=1 \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# Fix DeepEP IBGDA symlink
RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so


# =============================================================================
# Torch Deps Stage: PyTorch cu130 + build constraints
# =============================================================================
FROM base AS torch_deps

ARG CUDA_VERSION
ARG MAX_JOBS

WORKDIR /workspace

# PyTorch ecosystem (cu130)
RUN uv pip install --system --python python3.12 --break-system-packages \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    torch torchvision torchaudio ninja wheel packaging build setuptools setuptools-scm

# vLLM build deps (shallow clone just to resolve build requirements)
RUN git clone --depth=1 -b pr-ports https://github.com/licson/vllm.git /tmp/vllm \
    && cd /tmp/vllm \
    && if [ -f requirements/build/cuda.txt ]; then \
           uv pip install --system --python python3.12 --break-system-packages \
               -r requirements/build/cuda.txt; \
       fi \
    && rm -rf /tmp/vllm


# =============================================================================
# DeepEP Builder Stage
# =============================================================================
FROM torch_deps AS deepep_builder

ARG CUDA_VERSION
ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG MAX_JOBS

WORKDIR /build

RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    cd /repo-cache && \
    if [ ! -d "DeepEP" ]; then \
        git clone https://github.com/deepseek-ai/DeepEP.git; \
    fi \
    && cd DeepEP \
    && git fetch origin \
    && git checkout ${DEEPEP_COMMIT} \
    && git submodule update --init --recursive \
    && cp -a /repo-cache/DeepEP /build/DeepEP

WORKDIR /build/DeepEP

# Timeout tweaks (same as SGLang reference)
RUN sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh \
    && sed -i 's/#define NUM_TIMEOUT_CYCLES 200000000000ull/#define NUM_TIMEOUT_CYCLES 2000000000000ull/' csrc/kernels/configs.cuh

# CUDA 13 fix: add cccl include dir
RUN if [ "${CUDA_VERSION%%.*}" = "13" ]; then \
        sed -i "/^    include_dirs = \['csrc\/'\]/a\\    include_dirs.append('${CUDA_HOME}/include/cccl')" setup.py; \
    fi

RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    TORCH_CUDA_ARCH_LIST="12.0;12.1" MAX_JOBS=${MAX_JOBS} \
        python3 setup.py bdist_wheel -d /wheels


# =============================================================================
# FlashInfer Builder Stage (with Blackwell PR patches)
# =============================================================================
FROM torch_deps AS flashinfer_builder

ARG MAX_JOBS
ARG FI_PR_NUMBERS="3174 3180"

WORKDIR /build

RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    cd /repo-cache && \
    if [ ! -d "flashinfer" ]; then \
        git clone --recursive https://github.com/flashinfer-ai/flashinfer.git; \
    fi \
    && cd flashinfer \
    && git fetch origin \
    && git checkout main \
    && git submodule update --init --recursive \
    && git clean -fdx \
    && cp -a /repo-cache/flashinfer /build/flashinfer

WORKDIR /build/flashinfer

RUN git config --global user.email "builder@local" \
    && git config --global user.name "Docker Builder"

RUN set -e; \
    for PR in ${FI_PR_NUMBERS}; do \
        echo "Fetching and merging FlashInfer PR #${PR}..." && \
        git fetch origin pull/${PR}/head:pr-${PR} && \
        git merge --no-edit pr-${PR}; \
    done

ENV TORCH_CUDA_ARCH_LIST="12.1"
ENV FLASHINFER_CUDA_ARCH_LIST="12.1a"
ENV MAX_JOBS=${MAX_JOBS}

RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    uv pip install --system --python python3.12 --break-system-packages --no-deps build \
    && python3 -m pip wheel . --no-deps --no-build-isolation -w /wheels

# Build flashinfer-cubin wheel (requires main flashinfer package installed)
RUN uv pip install --system --python python3.12 --break-system-packages --no-deps /wheels/flashinfer*.whl \
    && uv pip install --system --python python3.12 --break-system-packages build \
    && cd /build/flashinfer/flashinfer-cubin \
    && python3 -m build --no-isolation --wheel \
    && cp dist/*.whl /wheels/

# Build flashinfer-jit-cache wheel if present
RUN if [ -d /build/flashinfer/flashinfer-jit-cache ]; then \
        cd /build/flashinfer/flashinfer-jit-cache \
        && python3 -m build --no-isolation --wheel \
        && cp dist/*.whl /wheels/; \
    fi


# =============================================================================
# DeepGEMM Builder Stage
# =============================================================================
FROM torch_deps AS deepgemm_builder

ARG DEEPGEMM_BRANCH=sm120
ARG MAX_JOBS

WORKDIR /build

RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    cd /repo-cache && \
    if [ ! -d "DeepGEMM" ]; then \
        git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git; \
    fi \
    && cd DeepGEMM \
    && git fetch origin \
    && git checkout ${DEEPGEMM_BRANCH} \
    && git submodule update --init --recursive \
    && cp -a /repo-cache/DeepGEMM /build/DeepGEMM

WORKDIR /build/DeepGEMM

RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    TORCH_CUDA_ARCH_LIST="12.0;12.1" MAX_JOBS=${MAX_JOBS} \
    python3 -m pip wheel . --no-deps --no-build-isolation -w /wheels


# =============================================================================
# vLLM Builder Stage
# =============================================================================
FROM torch_deps AS vllm_builder

ARG VLLM_BRANCH=pr-ports
ARG MAX_JOBS

WORKDIR /build

RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    cd /repo-cache && \
    if [ ! -d "vllm" ]; then \
        git clone --recursive https://github.com/licson/vllm.git; \
    fi \
    && cd vllm \
    && git fetch origin \
    && git checkout ${VLLM_BRANCH} \
    && git submodule update --init --recursive \
    && git clean -fdx \
    && cp -a /repo-cache/vllm /build/vllm

WORKDIR /build/vllm

# Remove flashinfer from requirements since we build it separately
RUN sed -i "/flashinfer/d" requirements/cuda.txt 2>/dev/null || true \
    && python3 use_existing_torch.py

RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    TORCH_CUDA_ARCH_LIST="12.0;12.1" MAX_JOBS=${MAX_JOBS} \
    uv build --no-build-isolation --wheel . --out-dir=/wheels -v


# =============================================================================
# Framework Stage: Assemble everything
# =============================================================================
FROM torch_deps AS framework

WORKDIR /workspace

# Copy built wheels from parallel stages
COPY --from=deepep_builder /wheels /tmp/wheels
COPY --from=flashinfer_builder /wheels /tmp/wheels
COPY --from=deepgemm_builder /wheels /tmp/wheels
COPY --from=vllm_builder /wheels /tmp/wheels

# Install all wheels
RUN uv pip install --system --python python3.12 --break-system-packages \
    /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

# Patch nvidia-cutlass-dsl for SM121a support
RUN uv pip install --system --python python3.12 --break-system-packages \
    "nvidia-cutlass-dsl==4.4.2"
RUN CUTE_DSL_MMA_PY=$(python3 -c \
    "import nvidia_cutlass_dsl; print(nvidia_cutlass_dsl.__path__[0])")/cute/nvgpu/warp/mma.py \
    && sed -i 's/sm_120a/sm_121a/g' "$CUTE_DSL_MMA_PY" \
    && grep -q sm_121a "$CUTE_DSL_MMA_PY" \
    && echo "CUTLASS SM121 patch applied OK"

# Fix Triton to use system ptxas for Blackwell (sm_120/sm_121) support (CUDA 13+)
RUN if [ "${CUDA_VERSION%%.*}" = "13" ] && [ -d /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin ]; then \
        rm -f /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas && \
        ln -s /usr/local/cuda/bin/ptxas /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas; \
    fi

# Additional runtime deps
RUN uv pip install --system --python python3.12 --break-system-packages \
    fastsafetensors ray[default]

# Smoke test
RUN python3 -c "import vllm; import flashinfer; import deep_gemm; print('vLLM SM12x framework OK')"


# =============================================================================
# Runtime Stage
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS runtime

ARG CUDA_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

ENV PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Install runtime dependencies only (no build tools, no ccache)
RUN apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    ca-certificates \
    software-properties-common \
    netcat-openbsd \
    curl \
    wget \
    git \
    locales \
    python3.12-full \
    python3.12-dev \
    libopenmpi3 \
    libnuma1 \
    libibverbs1 \
    libibumad3 \
    librdmacm1 \
    libnl-3-200 \
    libnl-route-3-200 \
    ibverbs-providers \
    rdma-core \
    infiniband-diags \
    perftest \
    libgoogle-glog0v6t64 \
    libunwind8 \
    libboost-system1.83.0 \
    libboost-thread1.83.0 \
    libboost-filesystem1.83.0 \
    libgrpc++1.51t64 \
    libprotobuf32t64 \
    libhiredis1.1.0 \
    libcurl4 \
    libczmq4 \
    libfabric1 \
    libssl3 \
    ninja-build \
    libnccl2 \
    libnccl-dev \
    linux-libc-dev \
    gnupg2 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && wget -q https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py --break-system-packages \
    && rm get-pip.py \
    && python3 -m pip config set global.break-system-packages true \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Copy Python packages from framework
COPY --from=framework /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages

# Fix DeepEP IBGDA symlink
RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so

# Fix Triton ptxas for Blackwell
RUN if [ "${CUDA_VERSION%%.*}" = "13" ] && [ -d /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin ]; then \
        rm -f /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas && \
        ln -s /usr/local/cuda/bin/ptxas /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas; \
    fi

# SM12x runtime tuning defaults
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV NCCL_MIN_NCHANNELS=32

WORKDIR /workspace
EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
