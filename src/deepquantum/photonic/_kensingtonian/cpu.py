"""Lazy host compilation of the shared double-double Kensingtonian core."""

import ctypes
import os
import shlex
import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import torch

_SOURCE_DIR = Path(__file__).with_name('native')
_COMPILE_LOCK = Lock()


class _Kernel:
    """Keep a compiled host library and its private temporary directory alive."""

    def __init__(self):
        if sys.platform not in ('darwin', 'linux'):
            raise RuntimeError('The double_double CPU backend currently supports macOS and Linux')
        compiler = shlex.split(os.environ.get('CXX', ''))
        if not compiler:
            executable = shutil.which('clang++') or shutil.which('g++')
            compiler = [executable] if executable else []
        if not compiler or shutil.which(compiler[0]) is None:
            raise RuntimeError('double_double on CPU requires a C++17 compiler: install clang++/g++ or set CXX')
        self.directory = TemporaryDirectory(prefix='deepquantum_dd_')
        library = Path(self.directory.name) / ('kernel.dylib' if sys.platform == 'darwin' else 'kernel.so')
        command = compiler + [
            '-std=c++17',
            '-O2',
            '-fno-fast-math',
            '-ffp-contract=off',
            '-shared',
            '-fPIC',
            '-I',
            str(_SOURCE_DIR),
            str(_SOURCE_DIR / 'kensingtonian_dd_cpu.cpp'),
            '-o',
            str(library),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
            if result.returncode:
                raise RuntimeError(f'DD CPU compilation failed:\n{result.stdout}\n{result.stderr}')
            self.library = ctypes.CDLL(str(library))
            self.function = self.library.kensingtonian_dd_cpu
            self.function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self.function.restype = None
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
            self.directory.cleanup()
            raise RuntimeError(f'Unable to initialize the double_double CPU backend: {error}') from error

    def __call__(
        self, matrices: torch.Tensor, clicks: torch.Tensor, detectors: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = matrices.new_empty(matrices.shape[0])
        sum_abs = torch.empty_like(output)
        self.function(
            matrices.data_ptr(),
            clicks.data_ptr(),
            matrices.shape[-1] // 2,
            detectors,
            len(matrices),
            output.data_ptr(),
            sum_abs.data_ptr(),
        )
        return output, sum_abs


@cache
def _cached_kernel(process_id: int) -> _Kernel:
    """Compile once per process, keeping build files outside the installed package."""
    return _Kernel()


def launch(matrices: torch.Tensor, clicks: torch.Tensor, detectors: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a contiguous CPU float64 batch with shared int32 click counts."""
    with _COMPILE_LOCK:
        kernel = _cached_kernel(os.getpid())
    return kernel(matrices, clicks, detectors)
