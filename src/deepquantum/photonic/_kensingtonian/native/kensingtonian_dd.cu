#include "kensingtonian_dd.cuh"

extern "C" __global__ void kensingtonian_dd_batch(
    const double* matrices, const int* clicks, int modes, int detectors,
    int batch_size, double* output, double* sum_abs
) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batch_size) return;
    auto value = deepquantum::kensingtonian(
        matrices + static_cast<long long>(batch) * 4 * modes * modes,
        clicks, modes, detectors, sum_abs + batch
    );
    output[batch] = value.hi + value.lo;
}
