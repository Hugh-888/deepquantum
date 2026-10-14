"""Validated input domain and cancellation diagnostics for DD Kensingtonians."""

from contextlib import nullcontext
from math import comb, isfinite, prod
from numbers import Integral

import torch


def _check_result(values: torch.Tensor, sum_abs: torch.Tensor, modes: int, rtol: float) -> None:
    """Reject nonpositive results and severe cancellation using a heuristic indicator.

    This measures loss from the alternating sum at roughly 106-bit precision. It
    is not a rigorous error bound and does not certify determinant conditioning or
    recover precision already lost when constructing the input matrix.
    """
    error_scale = 64 * (2 * modes) ** 3 * 2.0**-106 * sum_abs
    invalid = (~torch.isfinite(values)) | (~torch.isfinite(sum_abs)) | (sum_abs < 0) | (values <= 0)
    invalid |= error_scale > rtol * values.abs()
    if invalid.any().item():
        raise ArithmeticError(
            'double_double encountered a nonpositive/nonfinite result or excessive cancellation. '
            'Use an independent higher-precision calculation for this input; no automatic CPU fallback is performed. '
            'dd_rtol controls a heuristic diagnostic, not a guaranteed relative error bound.'
        )


def kensingtonian_dd(
    matrix: torch.Tensor,
    clicks: torch.Tensor | list[int],
    num_detectors: int,
    gamma: torch.Tensor | None,
    chunk_size: int | None,
    rtol: float,
) -> torch.Tensor:
    """Evaluate the bounded, undisplaced CPU/CUDA DD backend with explicit validation."""
    if matrix.dtype != torch.float64:
        raise TypeError('double_double requires a real torch.float64 input matrix')
    if matrix.ndim not in (2, 3) or matrix.shape[-2] != matrix.shape[-1] or matrix.shape[-1] % 2:
        raise ValueError('Input matrix must be square, even-dimensional, and 2D or 3D')
    if gamma is not None:
        raise NotImplementedError('double_double does not support gamma (displaced states)')
    if chunk_size is not None:
        raise NotImplementedError('double_double does not support term chunk_size; batch input matrices instead')
    if torch.is_grad_enabled() and matrix.requires_grad:
        raise NotImplementedError('double_double does not support autograd; use torch.no_grad() for inference')
    if (
        isinstance(num_detectors, bool)
        or not isinstance(num_detectors, Integral)
        or not 1 <= num_detectors <= 2**31 - 1
    ):
        raise ValueError('num_detectors must be a positive 32-bit integer')
    if not isfinite(rtol) or rtol <= 0:
        raise ValueError('dd_rtol must be finite and positive')
    if matrix.device.type not in ('cpu', 'cuda'):
        raise ValueError('double_double supports CPU and CUDA tensors only')
    context = torch.cuda.device(matrix.device) if matrix.is_cuda else nullcontext()
    with context:
        if matrix.is_cuda and torch.cuda.is_current_stream_capturing():
            raise NotImplementedError('double_double validation does not support CUDA graph capture')
        nmode = matrix.shape[-1] // 2
        counts = torch.as_tensor(clicks).detach().cpu().reshape(-1)
        if counts.is_complex() or counts.numel() != nmode:
            raise ValueError(f'clicks must contain {nmode} integer counts')
        if not ((counts >= 0) & (counts <= num_detectors) & (counts == counts.to(torch.long))).all().item():
            raise ValueError('clicks must be integers between zero and num_detectors')
        pattern = counts.to(torch.long).tolist()
        if sum(pattern) > 4:
            raise NotImplementedError('double_double currently supports at most four total clicks')
        active = [i for i, count in enumerate(pattern) if count]
        reduced = [pattern[i] for i in active]
        max_coefficient = prod(comb(num_detectors, count) * comb(count, count // 2) for count in reduced)
        if max_coefficient > 2**53:
            raise ValueError('double_double requires all integer coefficients to fit exactly in float64 (<= 2**53)')
        is_single = matrix.ndim == 2
        matrices = matrix.unsqueeze(0) if is_single else matrix
        if matrices.shape[0] > 2**31 - 1:
            raise ValueError('double_double batch size must fit a 32-bit integer')
        if not active or matrices.shape[0] == 0:
            result = matrix.new_ones(matrices.shape[0])
        else:
            indices = torch.tensor(active + [i + nmode for i in active], device=matrix.device)
            matrices = matrices[:, indices[:, None], indices].contiguous()
            if not torch.isfinite(matrices).all().item():
                raise ValueError('double_double requires finite entries in the clicked submatrix')
            if not torch.allclose(matrices, matrices.mT, rtol=1e-12, atol=1e-15):
                raise ValueError('double_double requires a symmetric clicked submatrix')
            identity = torch.eye(matrices.shape[-1], dtype=matrix.dtype, device=matrix.device)
            _, info = torch.linalg.cholesky_ex(identity - matrices)
            if info.any().item():
                raise ValueError('double_double requires positive definite I - O on the clicked modes')
            # A clicked mode with identically zero rows AND columns is independent vacuum,
            # so the exact answer is zero, even if other modes contain photons.
            modes = len(active)
            zero_rows = (matrices == 0).all(dim=-1) & (matrices == 0).all(dim=-2)
            exact_zero = (zero_rows[:, :modes] & zero_rows[:, modes:]).any(dim=-1)
            result = matrices.new_zeros(matrices.shape[0])
            nonzero = ~exact_zero
            if nonzero.any().item():
                if matrix.is_cuda:
                    from ._kensingtonian_nvrtc import launch
                else:
                    from ._kensingtonian_cpu import launch

                device_clicks = torch.tensor(reduced, dtype=torch.int32, device=matrix.device)
                values, sum_abs = launch(matrices[nonzero].contiguous(), device_clicks, int(num_detectors))
                _check_result(values, sum_abs, modes, rtol)
                result[nonzero] = values
        return result[0] if is_single else result
