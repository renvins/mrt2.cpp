// quant.h - int8 weights for the transformer's matrix products.
//
// Why: generating one frame reads every transformer weight (~1.5 GB in
// float32), and that read, not the arithmetic, is what limits the speed
// (bench: the big matvecs run at 82-87% of the memory bandwidth). Stored
// as int8, the same weights are 4x fewer bytes. Google's own Mac engine
// does the same: magenta_rt/mlx/system.py quantizes to 8 bits by default.
//
// The format ("Q8", like llama.cpp's Q8_0): every row of the matrix is cut
// into blocks of 32 numbers; a block stores 32 int8 values and one float
// scale d = (largest |w| in the block) / 127, so that w ~= d * q.
//
// The input vector x is quantized the same way, just before the product.
// Then one block's contribution to y is
//     sum_k (dw*qw[k]) * (dx*qx[k]) = dw*dx * sum_k qw[k]*qx[k]
// and the sum on the right is integer math: exact, and on ARM a single
// instruction (sdot) does 16 of those multiply-adds.
//
// Layout: [rows][in], one row per output (the "[out][in]" layout), so each
// output is one contiguous row read front to back. `in` must be a multiple
// of 128 (4 blocks; every size in this model is).
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mrt2 {

constexpr int kQBlock = 32;      // numbers per block
constexpr int kQMaxIn = 4096;    // largest `in` (the FFN hidden size)

struct QMatrix {
    int rows = 0, in = 0;
    std::vector<int8_t> q;       // [rows][in]
    std::vector<float> scale;    // [rows][in / 32]
};

// Bytes an int8 product reads per weight count: 1 + a 4-byte scale per 32
// (for the profiler).
inline double q8_bytes(double weights) { return weights * (1.0 + 4.0 / kQBlock); }

// Quantize n numbers (n a multiple of 32): q[i] = round(x[i] / d[i/32]).
// An all-zero block gets d = 0 and q = 0.
void quantize_q8(const float *x, size_t n, int8_t *q, float *d);

// Append `cols` rows to m, taken from columns [c0, c0 + cols) of a Flax
// kernel W [in][ld] (the "[in][out]" layout; the transpose happens here).
// Appending several kernels with the same `in` fuses them into one matrix,
// e.g. the q, k and v projections: one product instead of three.
void q8_append_kernel(QMatrix &m, const float *W, int in, int ld, int c0, int cols);

// Append `rows` rows of an [out][in] matrix W.
void q8_append_rows(QMatrix &m, const float *W, int rows, int in);

// y[r] = b[r] + sum over the row's blocks of dw*dx * (integer dot product),
// for r < m.rows, with x already quantized (qx [m.in], dx [m.in / 32]).
// `b` may be nullptr. Multithreaded by rows: each output is computed whole
// by one thread, so the result is the same for any thread count.
void q8_matvec(const QMatrix &m, const int8_t *qx, const float *dx, const float *b, float *y);

// The same, quantizing x first (x has m.in numbers).
void q8_matvec(const QMatrix &m, const float *x, const float *b, float *y);

// Plain C++ version of one output, for tests. The SIMD kernel must give
// exactly the same bits: both sum blocks into 4 lanes (block i into lane
// i % 4) and add the lanes as (l0 + l1) + (l2 + l3).
float q8_dot_ref(const int8_t *qw, const float *dw, const int8_t *qx, const float *dx, int in);

} // namespace mrt2
