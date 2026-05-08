# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_triton_kernels
from vllm.utils.torch_utils import direct_register_custom_op, is_torch_equal_or_newer

logger = init_logger(__name__)

# CK's pre-compiled MXFP4 MoE GEMM kernel instances require the
# intermediate_size (after TP split) to be a multiple of this value.
# This arises from FP4 packing (2 values per byte) combined with CK
# tile size constraints. When violated, AITER raises:
# "device_gemm ... does not support this GEMM problem".
CK_MXFP4_MOE_DIM_ALIGNMENT = 256


def _swizzle_mxfp4(quant_tensor, scale, num_warps=8):
    """weight swizzle for mxfp4 moe, used for OAI mxfp4 kernel"""
    assert has_triton_kernels()
    import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
    from triton_kernels.numerics import InFlexData
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details import layout
    from triton_kernels.tensor_details.layout import StridedLayout

    value_layout_opts: dict[str, Any] = {}
    scale_layout_opts: dict[str, Any] = {}

    if (
        current_platform.is_cuda()
        and current_platform.is_device_capability(90)
        and not is_torch_equal_or_newer("2.8.1")
    ):
        logger.warning_once(
            "Mxfp4 on hopper is running on torch < 2.8.1, "
            "this cause swizling to be disabled, which may "
            "cause performance degradation. Please upgrade to torch nightly"
        )
        value_layout = StridedLayout
        scale_layout = StridedLayout
    elif current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx950

        value_layout = StridedLayout
        if on_gfx950():
            try:
                # triton < 3.6
                from triton_kernels.tensor_details.layout import GFX950MXScaleLayout

                scale_layout = GFX950MXScaleLayout
            except ImportError:
                # triton >= 3.6
                from triton_kernels.tensor_details.layout import CDNA4MXScaleLayout

                scale_layout = CDNA4MXScaleLayout
        else:
            scale_layout = StridedLayout
    else:
        value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(
            mx_axis=1
        )
        scale_layout, scale_layout_opts = (
            layout.make_default_matmul_mxfp4_w_scale_layout(
                mx_axis=1, num_warps=num_warps
            )
        )
    if current_platform.is_cuda():
        if current_platform.is_device_capability(90):
            constraints = {
                "split_k": 1,
            }
            opt_flags.update_opt_flags_constraints(constraints)
        elif current_platform.is_device_capability_family(100):
            constraints = {
                "is_persistent": True,
                "epilogue_subtile": 1,
            }
            opt_flags.update_opt_flags_constraints(constraints)
    # transpose the tensor so that the quantization axis is on dim1
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4), value_layout, **value_layout_opts
    )
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout, **scale_layout_opts)
    return quant_tensor, InFlexData(), scale


def _dequant_mxfp4(
    x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype
) -> torch.Tensor:
    try:
        from quark.torch.kernel import mx
    except ImportError as err:
        raise ImportError(
            "The package `amd-quark` is required to use "
            "MX-FP4 models. Please install it with `pip install "
            "amd-quark`."
        ) from err

    return mx.dq_mxfp4(x, scale, float_dtype)


def _dequant_mxfp4_fake(
    x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype
) -> torch.Tensor:
    return torch.empty(
        (*x.shape[:-1], x.shape[-1] * 2), dtype=float_dtype, device=x.device
    )


def _quant_dequant_mxfp4(
    x: torch.Tensor, scale_calculation_mode: str = "even"
) -> torch.Tensor:
    try:
        from quark.torch.kernel import mx
    except ImportError as err:
        raise ImportError(
            "The package `amd-quark` is required to use "
            "MX-FP4 models. Please install it with `pip install "
            "amd-quark`."
        ) from err

    return mx.qdq_mxfp4(x, scale_calculation_mode)


def _quant_dequant_mxfp4_fake(
    x: torch.Tensor, scale_calculation_mode: str = "even"
) -> torch.Tensor:
    return torch.empty_like(x)


# Protect these operations into a torch custom op to avoid errors as
# torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
# Explanation: Dynamo does not know how to trace the builtin
# `kernel_ext.PyCapsule.dq_uint8_mxfp4_to_half.` This function is either a
# Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python
# extension (perhaps created with pybind).
# TODO: Make sure there is no way to avoid having these functions
# marked as skipped by dynamo.
try:
    direct_register_custom_op(
        op_name="dequant_mxfp4",
        op_func=_dequant_mxfp4,
        fake_impl=_dequant_mxfp4_fake,
    )
    dequant_mxfp4 = torch.ops.vllm.dequant_mxfp4
except AttributeError as error:
    raise error

try:
    direct_register_custom_op(
        op_name="quant_dequant_mxfp4",
        op_func=_quant_dequant_mxfp4,
        fake_impl=_quant_dequant_mxfp4_fake,
    )
    quant_dequant_mxfp4 = torch.ops.vllm.quant_dequant_mxfp4
except AttributeError as error:
    raise error


def xpu_mxfp4_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.xpu_mxfp4_quantize(x)


def _cast_to_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round float values to the nearest E2M1 representable value.

    Matches the thresholds used by the reference MXFP4 implementation.
    """
    sign = torch.sign(x)
    abs_x = x.abs()
    result = torch.where(abs_x > 5.0, 6.0, 0.0)
    result = torch.where((abs_x >= 3.5) & (abs_x <= 5.0), 4.0, result)
    result = torch.where((abs_x > 2.5) & (abs_x < 3.5), 3.0, result)
    result = torch.where((abs_x >= 1.75) & (abs_x <= 2.5), 2.0, result)
    result = torch.where((abs_x > 1.25) & (abs_x < 1.75), 1.5, result)
    result = torch.where((abs_x >= 0.75) & (abs_x <= 1.25), 1.0, result)
    result = torch.where((abs_x > 0.25) & (abs_x < 0.75), 0.5, result)
    return result * sign


def _float_to_e2m1_nibble(x: torch.Tensor) -> torch.Tensor:
    """Convert float values (already rounded to E2M1) to 4-bit codes.

    Returns uint8 tensor with values in [0, 15].
    """
    sign = (x < 0).to(torch.uint8) << 3
    abs_x = x.abs()
    mag = torch.zeros_like(abs_x, dtype=torch.uint8)
    mag = torch.where(abs_x >= 6.0, 7, mag)
    mag = torch.where((abs_x >= 4.0) & (abs_x < 6.0), 6, mag)
    mag = torch.where((abs_x >= 3.0) & (abs_x < 4.0), 5, mag)
    mag = torch.where((abs_x >= 2.0) & (abs_x < 3.0), 4, mag)
    mag = torch.where((abs_x >= 1.5) & (abs_x < 2.0), 3, mag)
    mag = torch.where((abs_x >= 1.0) & (abs_x < 1.5), 2, mag)
    mag = torch.where((abs_x >= 0.5) & (abs_x < 1.0), 1, mag)
    return sign | mag


def mxfp4_e2m1_quantize(
    x: torch.Tensor,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float tensor to MXFP4 (E2M1 weights + E8M0 block scales).

    Args:
        x: Float tensor of shape (..., K). K will be padded to a multiple
           of ``group_size`` (32) if necessary.
        group_size: Block size for quantization. Must be 32 for MXFP4.

    Returns:
        qweight: uint8 tensor of shape (..., K//2) with two E2M1 values
                 packed per byte (lower nibble first).
        scale: uint8 tensor of shape (..., K//group_size) with E8M0
               scale values.
    """
    assert x.dtype in (torch.float16, torch.bfloat16)
    assert group_size == 32, "MXFP4 requires group_size=32"

    orig_k = x.shape[-1]
    pad_k = ((orig_k + group_size - 1) // group_size) * group_size
    if pad_k > orig_k:
        x = torch.nn.functional.pad(x, (0, pad_k - orig_k))

    # Reshape to blocks
    x_blocks = x.reshape(-1, group_size)

    # Per-block max abs value
    block_max = x_blocks.abs().max(dim=-1).values.to(torch.float32)

    # Compute E8M0 scale exponent.
    # E8M0 represents 2^(exp - 127). We choose exp so that
    # scale = 2^(floor(log2(block_max)) - 2), which matches the
    # reference MXFP4 implementation.
    log2_max = torch.floor(torch.log2(block_max.clamp(min=1e-30)))
    scale_exp = (127 + log2_max - 2).to(torch.int32)
    scale_exp = torch.clamp(scale_exp, 0, 254)

    # E8M0 scale bytes
    scale = scale_exp.to(torch.uint8)

    # Convert scale back to float for quantization
    scale_float = (2.0 ** (scale_exp.float() - 127.0)).to(x.dtype)

    # Quantize: divide by scale, clamp, round to E2M1
    x_scaled = x / scale_float.reshape(x.shape[:-1] + (1,))
    x_scaled = x_scaled.clamp(-6.0, 6.0)
    x_fp4 = _cast_to_fp4(x_scaled)

    # Convert to nibbles and pack
    nibbles = _float_to_e2m1_nibble(x_fp4)
    nibbles = nibbles.reshape(*x.shape[:-1], -1, 2)
    packed = (nibbles[..., 1] << 4) | nibbles[..., 0]

    # Reshape outputs
    scale = scale.reshape(x.shape[:-1] + (pad_k // group_size,))

    return packed, scale


def get_padding_alignment(input_size: int, alignment: int = 32) -> int:
    """Return the padded size for the given input size."""
    return ((input_size + alignment - 1) // alignment) * alignment
