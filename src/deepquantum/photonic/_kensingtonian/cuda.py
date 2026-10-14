"""Lazy NVRTC compilation and CUDA driver launch for the optional DD backend."""

import ctypes
import os
import sys
from ctypes.util import find_library
from functools import cache
from pathlib import Path
from threading import Lock

import torch

_SOURCE_DIR = Path(__file__).with_name('native')
_COMPILE_LOCK = Lock()


def _load_nvrtc() -> ctypes.CDLL:
    """Find NVRTC from a CUDA-enabled PyTorch installation or CUDA toolkit."""
    candidates = []
    for entry in sys.path:
        if entry:
            candidates.extend(sorted((Path(entry) / 'nvidia/cuda_nvrtc/lib').glob('libnvrtc.so*')))
    roots = [Path(torch.__file__).parent / 'lib', Path(sys.prefix) / 'lib', Path('/usr/local/cuda/lib64')]
    for variable in ('CUDA_HOME', 'CUDA_PATH'):
        if os.environ.get(variable):
            roots.append(Path(os.environ[variable]) / 'lib64')
    for root in roots:
        candidates.extend(sorted(root.glob('libnvrtc.so*')))
    found = find_library('nvrtc')
    candidates.extend([found] if found else [])
    candidates.append('libnvrtc.so')
    errors = []
    for candidate in dict.fromkeys(map(str, candidates)):
        try:
            return ctypes.CDLL(candidate)
        except OSError as error:
            errors.append(str(error))
    raise RuntimeError(
        'double_double requires NVIDIA NVRTC on Linux. Install the CUDA runtime dependencies matching '
        'your PyTorch installation, or set CUDA_HOME to a CUDA toolkit. Last loader error: ' + errors[-1]
    )


def _bind(library: ctypes.CDLL, name: str, arguments: list) -> object:
    """Declare a CUDA C function signature (all used functions return int)."""
    function = getattr(library, name)
    function.argtypes = arguments
    function.restype = ctypes.c_int
    return function


def _check(code: int, operation: str) -> None:
    """Raise with the failing CUDA/NVRTC operation and status code."""
    if code:
        raise RuntimeError(f'{operation} failed with CUDA/NVRTC status {code}')


class _Kernel:
    """A compiled module kept alive with its driver library for the process lifetime."""

    def __init__(self, device: int):
        void = ctypes.c_void_p
        pointer = ctypes.POINTER
        integer = ctypes.c_int
        char = ctypes.c_char_p
        size = ctypes.c_size_t
        self.nvrtc = lib = _load_nvrtc()
        self.driver = driver = ctypes.CDLL('libcuda.so.1')
        create = _bind(lib, 'nvrtcCreateProgram', [pointer(void), char, char, integer, pointer(char), pointer(char)])
        compile_program = _bind(lib, 'nvrtcCompileProgram', [void, integer, pointer(char)])
        log_size = _bind(lib, 'nvrtcGetProgramLogSize', [void, pointer(size)])
        get_log = _bind(lib, 'nvrtcGetProgramLog', [void, void])
        cubin_size = _bind(lib, 'nvrtcGetCUBINSize', [void, pointer(size)])
        get_cubin = _bind(lib, 'nvrtcGetCUBIN', [void, void])
        destroy = _bind(lib, 'nvrtcDestroyProgram', [pointer(void)])
        load = _bind(driver, 'cuModuleLoadData', [pointer(void), void])
        get_function = _bind(driver, 'cuModuleGetFunction', [pointer(void), void, char])
        self.launch = _bind(
            driver, 'cuLaunchKernel', [void] + [ctypes.c_uint] * 7 + [void, pointer(void), pointer(void)]
        )
        header_names = (b'dd.cuh', b'kensingtonian_dd.cuh')
        headers = [(_SOURCE_DIR / name.decode()).read_bytes() for name in header_names]
        source = (_SOURCE_DIR / 'kensingtonian_dd.cu').read_bytes()
        program = void()
        _check(
            create(
                ctypes.byref(program),
                source,
                b'kensingtonian_dd.cu',
                2,
                (char * 2)(*headers),
                (char * 2)(*header_names),
            ),
            'nvrtcCreateProgram',
        )
        try:
            major, minor = torch.cuda.get_device_capability(device)
            # Native cubin avoids requiring a driver that understands NVRTC's newer PTX version.
            # Never enable fast math: implicit contraction/reassociation breaks the DD transforms.
            options = [f'--gpu-architecture=sm_{major}{minor}'.encode(), b'--std=c++17', b'--fmad=false']
            status = compile_program(program, len(options), (char * len(options))(*options))
            if status:
                length = size()
                _check(log_size(program, ctypes.byref(length)), 'nvrtcGetProgramLogSize')
                log = ctypes.create_string_buffer(length.value)
                _check(get_log(program, log), 'nvrtcGetProgramLog')
                raise RuntimeError(f'DD CUDA compilation failed ({status}): {log.value.decode(errors="replace")}')
            length = size()
            _check(cubin_size(program, ctypes.byref(length)), 'nvrtcGetCUBINSize')
            binary = ctypes.create_string_buffer(length.value)
            _check(get_cubin(program, binary), 'nvrtcGetCUBIN')
        finally:
            destroy(ctypes.byref(program))
        self.module = void()
        _check(load(ctypes.byref(self.module), binary), 'cuModuleLoadData')
        self.function = void()
        _check(get_function(ctypes.byref(self.function), self.module, b'kensingtonian_dd_batch'), 'cuModuleGetFunction')

    def __call__(
        self, matrices: torch.Tensor, clicks: torch.Tensor, detectors: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = matrices.new_empty(matrices.shape[0])
        sum_abs = torch.empty_like(output)
        void = ctypes.c_void_p
        values = [
            void(matrices.data_ptr()),
            void(clicks.data_ptr()),
            ctypes.c_int(matrices.shape[-1] // 2),
            ctypes.c_int(detectors),
            ctypes.c_int(matrices.shape[0]),
            void(output.data_ptr()),
            void(sum_abs.data_ptr()),
        ]
        arguments = (void * len(values))(*(ctypes.cast(ctypes.pointer(value), void) for value in values))
        stream = torch.cuda.current_stream(matrices.device)
        # The driver launch is invisible to PyTorch's allocator. Retain temporary inputs on this stream.
        matrices.record_stream(stream)
        clicks.record_stream(stream)
        _check(
            self.launch(
                self.function, (len(output) + 63) // 64, 1, 1, 64, 1, 1, 0, void(stream.cuda_stream), arguments, None
            ),
            'cuLaunchKernel',
        )
        return output, sum_abs


@cache
def _cached_kernel(process_id: int, device: int) -> _Kernel:
    """Cache per process/device; never reuse inherited CUDA handles after fork."""
    return _Kernel(device)


def launch(matrices: torch.Tensor, clicks: torch.Tensor, detectors: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch on the input device and its current PyTorch CUDA stream."""
    if sys.platform != 'linux':
        raise RuntimeError('The double_double CUDA backend currently supports Linux only')
    with torch.cuda.device(matrices.device):
        torch.cuda.init()
        # Initialize the primary context before using the CUDA driver directly.
        torch.cuda.current_stream(matrices.device)
        with _COMPILE_LOCK:
            kernel = _cached_kernel(os.getpid(), matrices.device.index)
        return kernel(matrices, clicks, detectors)
