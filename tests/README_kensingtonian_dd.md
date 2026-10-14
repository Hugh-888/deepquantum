# Double-double Kensingtonian

Enable the optional backend explicitly:

```python
import torch
from deepquantum.photonic import kensingtonian

device = 'cpu'  # use 'cuda' or 'cuda:3' for an NVIDIA GPU
matrix = torch.eye(2, dtype=torch.float64, device=device) * 1e-4
value = kensingtonian(matrix, [4], 4, precision='double_double')
```

The default `precision='standard'` retains the existing PyTorch implementation.
`QumodeCircuit` continues to use that default; this change exposes the DD backend
through the `kensingtonian` function, without switching circuit calculations automatically.

The backend follows `matrix.device` automatically and returns results on that device:

- CPU: macOS/Linux, with a C++17 compiler (`clang++` or `g++`, or set `CXX`). The
  shared DD core compiles on first use into a private temporary directory and is
  cached for the process lifetime. The CPU batch loop runs serially in compiled
  C++; it does not initialize CUDA or need any CUDA libraries.
- CUDA: Linux, NVIDIA CUDA-enabled PyTorch, and NVIDIA NVRTC (normally included
  in the CUDA runtime dependencies of PyTorch). No nvcc or host C++ compiler is
  needed. Compilation is cached per process/device. Computation uses that device's
  current PyTorch stream; input/result validation synchronizes the stream.

Both paths use the same DD arithmetic and determinant core. Neither path requires
mpmath or CuPy. The first call includes compilation overhead. MPS, Windows,
CUDA graph capture and differentiable execution are currently unsupported.

Supported inputs are real `torch.float64` matrices in `xxpp` order, a shared click
pattern across a batch, no displacement (`gamma=None`), and at most four total
clicks. Zero-click modes are removed before computing. Empty batches, all-zero
patterns and independent vacuum modes are handled explicitly. `chunk_size` is
unsupported; split a large matrix batch at the call site if needed. Detector
counts must be positive 32-bit integers and the maximum integer coefficient must
fit exactly in float64. Unsupported inputs raise errors rather than silently
using another precision/backend. Autograd inputs require an explicit
`torch.no_grad()` context for inference.

All finite-sum terms are evaluated: integer coefficients, construction of
`I - diag(d/N, d/N) O`, pivoted elimination, determinant, square root, division,
and signed accumulation use double-double arithmetic. Input and output remain
float64. The arithmetic primitives are adapted from GBSKernels; provenance and
its Apache-2.0 license are packaged alongside the kernel.

Double-double still has finite precision. Nonfinite/nonpositive results and a
heuristic indicator of severe cancellation raise `ArithmeticError`. The default
`dd_rtol=1e-6` controls that indicator:

```
64 * (2 * active_modes)**3 * 2**-106 * sum(abs(term)) / abs(result)
```

This conservative indicator is **not a certified relative-error bound**. It can
reject accurate results, does not account fully for ill-conditioned determinants,
and cannot recover precision lost in input construction. Raising `dd_rtol` relaxes
the check but does not improve the calculation. For rejected inputs use an
independent higher-precision reference; there is no automatic CPU fallback.

## Running tests

From the repository root, with development test dependencies installed:

```bash
PYTHONPATH=src python -m pytest tests/test_photonic_kensingtonian.py tests/test_photonic_kensingtonian_dd.py -q
```

`test_photonic_kensingtonian_dd.py` covers exact rational thermal references,
frozen 120-digit correlated-state references, mode permutations, nonbinary
`d/N`, squeezing, batches, unchanged standard autograd, unsupported options and
extreme cancellation. It uses relative error with zero absolute tolerance for
small nonzero reference values.

The host cross-check compiles the **same packaged kernel source** with a synthetic
thread adapter using clang++ or g++. Numerical cases also run through the public
CPU API. Both device backends test input validation, zero-click reduction, vacuum
and empty inputs. CUDA tests additionally exercise NVRTC, non-default streams,
input-device selection and agreement with the public CPU backend.
Host checks cannot replace those CUDA integration tests. CUDA tests skip only
when CUDA is unavailable (the multi-device test additionally needs two GPUs).

Local validation on 2026-10-14: 114 passed and 52 skipped across the two modules on
a CPU-only macOS Python/PyTorch environment; CUDA and MPS execution was unavailable.
An additional host cross-check of 13,326 previously saved physical input matrices
against independent 80-digit references had maximum relative error 1.94e-12,
with no cancellation diagnostic rejection. The large external audit bank is not
required by this portable test module. No new GPU speed measurement was made for
this integrated version.
