// CPU batch entry point using exactly the same DD arithmetic as the CUDA kernel.
#define __host__
#define __device__
#include "kensingtonian_dd.cuh"

extern "C" void kensingtonian_dd_cpu(
    const double* matrices, const int* clicks, int modes, int detectors,
    int batch_size, double* output, double* sum_abs
) {
    for (int batch = 0; batch < batch_size; ++batch) {
        auto value = deepquantum::kensingtonian(
            matrices + static_cast<long long>(batch) * 4 * modes * modes,
            clicks, modes, detectors, sum_abs + batch
        );
        output[batch] = value.hi + value.lo;
    }
}
