"""DD numerical regression tests, with the same kernel exercised on host and CUDA.

The independent host harness is a test adapter alongside the public CPU backend. It executes
our packaged CUDA kernel with one synthetic thread at a time. CUDA cases test the
public API and NVRTC/stream integration, and skip only when CUDA is unavailable.
"""

import ctypes
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from math import comb, factorial, prod
from pathlib import Path

import pytest
import torch

from deepquantum.photonic import kensingtonian
from deepquantum.photonic._kensingtonian_dd import _check_result
from deepquantum.photonic._kensingtonian_nvrtc import _SOURCE_DIR

_REFERENCE = json.loads((Path(__file__).parent / 'data/kensingtonian_dd_reference.json').read_text())['cases']
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='NVIDIA CUDA is not available')


@pytest.fixture(scope='module')
def host_kernel(tmp_path_factory):
    compiler = shutil.which('clang++') or shutil.which('g++')
    if compiler is None or sys.platform == 'win32':
        pytest.skip('Host DD cross-check requires clang++/g++ on macOS or Linux')
    directory = tmp_path_factory.mktemp('kensingtonian_dd_host')
    source = directory / 'host.cpp'
    source.write_text("""
#define __host__
#define __device__
#define __global__
struct Index { int x; } blockIdx, blockDim, threadIdx;
#include "kensingtonian_dd.cu"
extern "C" void host_batch(const double* matrices, const int* clicks, int modes,
                           int detectors, int batch, double* output, double* sum_abs) {
    blockDim.x = 1; threadIdx.x = 0;
    for (int i = 0; i < batch; ++i) {
        blockIdx.x = i;
        kensingtonian_dd_batch(matrices, clicks, modes, detectors, batch, output, sum_abs);
    }
}
""")
    library = directory / ('host.dylib' if sys.platform == 'darwin' else 'host.so')
    subprocess.run(
        [
            compiler,
            '-std=c++17',
            '-O2',
            '-fno-fast-math',
            '-ffp-contract=off',
            '-shared',
            '-fPIC',
            '-I',
            str(_SOURCE_DIR),
            str(source),
            '-o',
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    handle = ctypes.CDLL(str(library))
    function = handle.host_batch
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    function.restype = None

    def calculate(matrix, clicks, detectors, *, diagnostic=True):
        matrix = matrix.cpu().double().contiguous()
        is_single = matrix.ndim == 2
        matrix = matrix.unsqueeze(0) if is_single else matrix
        pattern = torch.tensor(clicks, dtype=torch.int32)
        output = torch.empty(len(matrix), dtype=torch.double)
        sum_abs = torch.empty_like(output)
        function(
            matrix.data_ptr(),
            pattern.data_ptr(),
            len(clicks),
            detectors,
            len(matrix),
            output.data_ptr(),
            sum_abs.data_ptr(),
        )
        if diagnostic:
            _check_result(output, sum_abs, len(clicks), 1e-6)
        return output[0] if is_single else output

    return calculate


@pytest.fixture(params=['host', 'cpu', 'cuda'])
def evaluate(request):
    if request.param == 'host':
        return request.getfixturevalue('host_kernel')
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('NVIDIA CUDA is not available')

    def calculate(matrix, clicks, detectors):
        return kensingtonian(matrix.to(request.param), clicks, detectors, precision='double_double').cpu()

    return calculate


def thermal_reference(diagonal, clicks, detectors):
    # Exact rational closed form for the input float; no alternating-sum reference.
    occupation = Fraction.from_float(float(diagonal)) / (1 - Fraction.from_float(float(diagonal)))
    probability = comb(detectors, clicks) * factorial(clicks) * (occupation / detectors) ** clicks
    probability /= prod(1 + occupation - j * occupation / detectors for j in range(clicks + 1))
    return probability / (1 - Fraction.from_float(float(diagonal)))


@pytest.mark.parametrize('clicks', [[1], [2], [3], [4], [1, 1], [2, 1], [1, 1, 2], [1, 1, 1, 1]])
@pytest.mark.parametrize('diagonal', [0.2, 1e-4, 5e-5])
def test_thermal_exact_rational(evaluate, clicks, diagonal):
    modes = len(clicks)
    matrix = torch.eye(2 * modes, dtype=torch.double) * diagonal
    expected = float(prod(thermal_reference(diagonal, count, 4) for count in clicks))
    actual = evaluate(matrix, clicks, 4).item()
    assert actual == pytest.approx(expected, rel=2e-8, abs=0)


@pytest.mark.parametrize('detectors,clicks', [(1, [1, 1]), (3, [1, 2]), (7, [4]), (17, [2, 2])])
def test_nonbinary_detector_ratios(evaluate, detectors, clicks):
    diagonal = 0.013
    matrix = torch.eye(2 * len(clicks), dtype=torch.double) * diagonal
    expected = float(prod(thermal_reference(diagonal, count, detectors) for count in clicks))
    assert evaluate(matrix, clicks, detectors).item() == pytest.approx(expected, rel=1e-12, abs=0)


@pytest.mark.parametrize('case', _REFERENCE, ids=lambda case: case['model'])
def test_frozen_120_digit_reference_and_permutation(evaluate, case):
    matrix = torch.tensor(case['matrix'], dtype=torch.double)
    expected = float(case['reference'])
    actual = evaluate(matrix, case['clicks'], 4)
    assert actual.item() == pytest.approx(expected, rel=1e-9, abs=0)
    modes = len(case['clicks'])
    reverse = list(reversed(range(modes)))
    indices = reverse + [i + modes for i in reverse]
    permuted = matrix[indices][:, indices]
    assert evaluate(permuted, list(reversed(case['clicks'])), 4).item() == pytest.approx(expected, rel=1e-9, abs=0)


def test_batch_and_noncontiguous(evaluate):
    matrices = torch.stack([torch.eye(4, dtype=torch.double) * value for value in (1e-4, 0.01, 0.2)])
    expected = torch.tensor(
        [float(thermal_reference(value, 2, 4) ** 2) for value in (1e-4, 0.01, 0.2)], dtype=torch.double
    )
    torch.testing.assert_close(evaluate(matrices.mT, [2, 2], 4), expected, rtol=1e-8, atol=0)


def test_standard_default_and_autograd_unchanged():
    matrix = (torch.eye(2, dtype=torch.double) * 0.1).requires_grad_()
    standard = kensingtonian(matrix, [1], 4, precision='standard')
    assert torch.equal(standard, kensingtonian(matrix, [1], 4))
    assert torch.autograd.gradcheck(lambda value: kensingtonian(value, [1], 4), (matrix,))
    with pytest.raises(ValueError, match='precision'):
        kensingtonian(matrix, [1], 4, precision='unknown')


@pytest.mark.parametrize(
    'options,error,match',
    [
        ({'gamma': torch.zeros(2)}, NotImplementedError, 'gamma'),
        ({'chunk_size': 2}, NotImplementedError, 'chunk_size'),
        ({'dd_rtol': 0}, ValueError, 'dd_rtol'),
        ({'dd_rtol': float('nan')}, ValueError, 'dd_rtol'),
        ({'num_detectors': 2.5}, ValueError, 'num_detectors'),
        ({'num_detectors': True}, ValueError, 'num_detectors'),
    ],
)
def test_unsupported_options_without_cuda(options, error, match):
    arguments = {'num_detectors': 4, 'precision': 'double_double'} | options
    with pytest.raises(error, match=match):
        kensingtonian(torch.eye(2, dtype=torch.double), [1], **arguments)


def test_device_dtype_and_grad_contract():
    with pytest.raises(ValueError, match='CPU and CUDA'):
        kensingtonian(torch.eye(2, dtype=torch.double, device='meta'), [1], 4, precision='double_double')
    with pytest.raises(TypeError, match='float64'):
        kensingtonian(torch.eye(2), [1], 4, precision='double_double')
    with pytest.raises(NotImplementedError, match='autograd'):
        kensingtonian(torch.eye(2, dtype=torch.double, requires_grad=True), [1], 4, precision='double_double')


@pytest.mark.parametrize('diagonal', [1e-7, 1e-8, 1e-9, 1e-10])
def test_extreme_cancellation_is_rejected(host_kernel, diagonal):
    matrix = torch.eye(2, dtype=torch.double) * diagonal
    # At 1e-7 the raw result remains positive despite losing percent-level accuracy.
    if diagonal == 1e-7:
        raw = host_kernel(matrix, [4], 4, diagnostic=False).item()
        assert raw > 0
        assert abs(raw / float(thermal_reference(diagonal, 4, 4)) - 1) > 1e-3
    with pytest.raises(ArithmeticError, match='cancellation'):
        host_kernel(matrix, [4], 4)


@pytest.mark.parametrize('value,terms', [(float('nan'), 1), (float('inf'), 1), (-1, 1), (0, 1), (1, -1)])
def test_bad_numerical_results_raise(value, terms):
    with pytest.raises(ArithmeticError):
        _check_result(torch.tensor([value], dtype=torch.double), torch.tensor([terms], dtype=torch.double), 1, 1e-6)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=_CUDA)])
def test_device_zero_modes_empty_batch_and_reduction(device):
    options = {'precision': 'double_double'}
    matrix = torch.eye(6, dtype=torch.double, device=device) * 0.01
    assert kensingtonian(matrix, [0, 0, 0], 4, **options).item() == 1
    assert kensingtonian(matrix.new_empty(0, 6, 6), [1, 1, 1], 4, **options).shape == (0,)
    assert kensingtonian(matrix.new_empty(0, 0), [], 4, **options).item() == 1
    expected = float(thermal_reference(0.01, 2, 4) ** 2)
    assert kensingtonian(matrix, [2, 0, 2], 4, **options).item() == pytest.approx(expected, rel=1e-10, abs=0)
    vacuum = matrix.clone()
    vacuum[0, 0] = vacuum[3, 3] = 0
    batch = torch.stack([matrix, vacuum, torch.zeros_like(matrix)])
    values = kensingtonian(batch, [1, 1, 1], 4, **options)
    assert values[0] > 0
    assert torch.equal(values[1:], values.new_zeros(2))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=_CUDA)])
@pytest.mark.parametrize(
    'clicks,detectors,error',
    [
        ([1.5], 4, ValueError),
        ([-1], 4, ValueError),
        ([5], 4, ValueError),
        ([0, 1], 4, ValueError),
        ([5], 5, NotImplementedError),
        ([4], 100000, ValueError),
    ],
)
def test_device_click_validation(device, clicks, detectors, error):
    with pytest.raises(error):
        kensingtonian(
            torch.eye(2, dtype=torch.double, device=device) * 0.1, clicks, detectors, precision='double_double'
        )


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=_CUDA)])
@pytest.mark.parametrize('matrix', [[[float('nan'), 0], [0, 0]], [[2, 0], [0, 2]], [[0.1, 0.1], [0, 0.1]]])
def test_device_invalid_matrix(device, matrix):
    with pytest.raises(ValueError):
        kensingtonian(torch.tensor(matrix, dtype=torch.double, device=device), [1], 4, precision='double_double')


@_CUDA
def test_cuda_stream_inference_and_cancellation():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.no_grad():
        matrix = (torch.eye(2, dtype=torch.double, device='cuda') * 1e-4).requires_grad_()
        value = kensingtonian(matrix, [4], 4, precision='double_double')
        assert not value.requires_grad
        assert value.item() == pytest.approx(float(thermal_reference(1e-4, 4, 4)), rel=1e-9, abs=0)
        with pytest.raises(ArithmeticError, match='cancellation'):
            kensingtonian(matrix.detach() * 1e-3, [4], 4, precision='double_double')


@_CUDA
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Multiple CUDA devices are required')
def test_cuda_input_device_differs_from_current_device():
    with torch.cuda.device(0):
        matrix = torch.eye(2, dtype=torch.double, device='cuda:1') * 1e-4
        value = kensingtonian(matrix, [4], 4, precision='double_double')
        assert value.device == matrix.device
        assert torch.cuda.current_device() == 0
        assert value.item() == pytest.approx(float(thermal_reference(1e-4, 4, 4)), rel=1e-9, abs=0)


@pytest.mark.parametrize('modes', [1, 2, 3, 4])
def test_correlated_moderate_inputs_match_standard(evaluate, modes):
    generator = torch.Generator().manual_seed(173 + modes)
    raw = torch.randn(2 * modes, 2 * modes, generator=generator, dtype=torch.double) * 0.2
    identity = torch.eye(2 * modes, dtype=torch.double)
    matrix = identity - torch.linalg.inv(identity + raw @ raw.mT)
    clicks = [1] * modes
    expected = kensingtonian(matrix, clicks, 4)
    torch.testing.assert_close(evaluate(matrix, clicks, 4), expected, rtol=1e-8, atol=0)


@pytest.mark.parametrize('clicks', [1, 2, 3, 4])
def test_squeezed_state_matches_standard(evaluate, clicks):
    variances = torch.exp(torch.tensor([-0.8, 0.8], dtype=torch.double))
    matrix = torch.diag(1 - 2 / (1 + variances))
    expected = kensingtonian(matrix, [clicks], 4)
    torch.testing.assert_close(evaluate(matrix, [clicks], 4), expected, rtol=1e-9, atol=0)


def test_cancellation_diagnostic_is_conservative(host_kernel):
    # The guard may reject accurate results. Its threshold is intentionally not
    # advertised as a calibrated or rigorous forward-error estimate.
    matrix = torch.eye(8, dtype=torch.double) * 2e-5
    raw = host_kernel(matrix, [1, 1, 1, 1], 4, diagnostic=False).item()
    expected = float(thermal_reference(2e-5, 1, 4) ** 4)
    assert raw == pytest.approx(expected, rel=1e-8, abs=0)
    with pytest.raises(ArithmeticError, match='cancellation'):
        host_kernel(matrix, [1, 1, 1, 1], 4)


def test_cpu_public_inference_and_cancellation():
    matrix = (torch.eye(2, dtype=torch.double) * 1e-4).requires_grad_()
    with torch.no_grad():
        value = kensingtonian(matrix, [4], 4, precision='double_double')
    assert value.device.type == 'cpu'
    assert value.dtype == torch.float64
    assert not value.requires_grad
    assert value.item() == pytest.approx(float(thermal_reference(1e-4, 4, 4)), rel=1e-9, abs=0)
    with pytest.raises(ArithmeticError, match='cancellation'):
        kensingtonian(matrix.detach() * 1e-3, [4], 4, precision='double_double')


def test_cpu_missing_compiler_is_actionable(monkeypatch):
    from deepquantum.photonic._kensingtonian_cpu import _Kernel

    monkeypatch.delenv('CXX', raising=False)
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    with pytest.raises(RuntimeError, match=r'C\+\+17 compiler'):
        _Kernel()


@_CUDA
def test_cpu_cuda_agree():
    case = _REFERENCE[0]
    matrix = torch.tensor(case['matrix'], dtype=torch.double)
    cpu = kensingtonian(matrix, case['clicks'], 4, precision='double_double')
    gpu = kensingtonian(matrix.cuda(), case['clicks'], 4, precision='double_double')
    torch.testing.assert_close(cpu, gpu.cpu(), rtol=1e-10, atol=0)


def test_cpu_does_not_initialize_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU computation must not touch CUDA')

    monkeypatch.setattr(torch.cuda, 'init', forbidden)
    monkeypatch.setattr(torch.cuda, 'device', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', forbidden)
    matrix = torch.eye(2, dtype=torch.double) * 0.01
    value = kensingtonian(matrix, [2], 4, precision='double_double')
    assert value.item() == pytest.approx(float(thermal_reference(0.01, 2, 4)), rel=1e-10, abs=0)
