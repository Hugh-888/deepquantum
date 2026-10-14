// Full finite Kensingtonian sum in double-double arithmetic, without truncation.
#pragma once
#include "dd.cuh"

namespace deepquantum {
using namespace gbs;

__host__ __device__ inline long long choose(int n, int k) {
    long long result = 1;
    for (int j = 1; j <= k; ++j) result = result * (n - j + 1) / j;
    return result;
}

__host__ __device__ inline dd determinant(dd* matrix, int n) {
    dd result = dd_from(1.0);
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double best = fabs(matrix[col * n + col].hi);
        for (int row = col + 1; row < n; ++row) {
            double value = fabs(matrix[row * n + col].hi);
            if (value > best) { best = value; pivot = row; }
        }
        if (best == 0.0) return dd_from(0.0);
        if (pivot != col) {
            for (int j = col; j < n; ++j) {
                dd temp = matrix[col * n + j];
                matrix[col * n + j] = matrix[pivot * n + j];
                matrix[pivot * n + j] = temp;
            }
            result = dd_neg(result);
        }
        dd diagonal = matrix[col * n + col];
        result = dd_mul(result, diagonal);
        for (int row = col + 1; row < n; ++row) {
            dd factor = dd_div(matrix[row * n + col], diagonal);
            for (int j = col + 1; j < n; ++j) {
                matrix[row * n + j] = dd_add(matrix[row * n + j],
                                            dd_neg(dd_mul(factor, matrix[col * n + j])));
            }
        }
    }
    return result;
}

// Caller checks m <= 4, sum(clicks) <= 4 and exact representability of coefficients.
// Sum |term| is a cancellation diagnostic, not a certified error bound.
__host__ __device__ inline dd kensingtonian(
    const double* matrix, const int* clicks, int m, int detectors, double* sum_abs
) {
    const int n = 2 * m;
    int num_terms = 1, total_clicks = 0;
    for (int i = 0; i < m; ++i) {
        num_terms *= clicks[i] + 1;
        total_clicks += clicks[i];
    }
    dd total = dd_from(0.0);
    *sum_abs = 0.0;
    for (int term = 0; term < num_terms; ++term) {
        int index = term, d[4], parity = total_clicks;
        long long coefficient = 1;
        for (int i = 0; i < m; ++i) {
            d[i] = index % (clicks[i] + 1);
            index /= clicks[i] + 1;
            parity -= d[i];
            coefficient *= choose(detectors, clicks[i]) * choose(clicks[i], d[i]);
        }
        if (parity % 2) coefficient = -coefficient;
        dd scaled[64];
        for (int i = 0; i < n; ++i) {
            // Form both d/N and I - diag(d/N, d/N) O in DD, not in FP64.
            dd ratio = dd_div_d(dd_from(static_cast<double>(d[i % m])), detectors);
            for (int j = 0; j < n; ++j) {
                scaled[i * n + j] = dd_add(dd_from(i == j ? 1.0 : 0.0),
                                           dd_neg(dd_mul_d(ratio, matrix[i * n + j])));
            }
        }
        dd det = determinant(scaled, n);
        if (!(det.hi > 0.0)) {
            // Nonfinite diagnostic makes the Python wrapper reject the result.
            *sum_abs = -1.0;
            return dd_from(0.0);
        }
        dd value = dd_div(dd_from(static_cast<double>(coefficient)), dd_sqrt(det));
        total = dd_add(total, value);
        *sum_abs += fabs(value.hi) + fabs(value.lo);
    }
    return total;
}
}  // namespace deepquantum
