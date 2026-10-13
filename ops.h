// ops.h - the math every model stage is built from.
//
// These are the plain, scalar float32 versions: the "reference" that is
// easy to read and to check. In the speed phase their bodies get replaced
// by SIMD / multithreaded code, but the signatures and the tests stay, so
// the fast versions must give the same answers.
//
// Rules for the whole file:
//   * nothing allocates memory (the per-frame loop must never call malloc),
//   * the caller owns every buffer; sizes are passed explicitly,
//   * an output buffer may be the same as the input only where it says so.
#pragma once
#include <cstddef>
#include <cstdint>

namespace mrt2 {

// ---- matrix x vector ------------------------------------------------------

// y[j] = b[j] + sum_i x[i] * W[i*ld + j]   for j < cols.
//
// W is stored [in][ld], row-major: the layout of a Flax Dense "kernel"
// [in, out]. Usually ld == cols. A larger ld selects a column slice: e.g.
// one RVQ level's 1024 logits out of the 12294 columns of to_logits, by
// passing W + first_column, ld = 12294, cols = 1024.
// `b` may be nullptr (no bias). `y` must not overlap `x`.
void vecmat(const float *x, size_t in, const float *W, size_t ld, size_t cols,
            const float *b, float *y);

// sum_i a[i] * b[i]. Uses 16 independent partial sums: float additions
// may not be reordered, so a single running sum is one long serial chain
// that the compiler can't turn into SIMD; 16 separate chains it can.
float dot(const float *a, const float *b, size_t n);

// y[r] = sum_k W[r*in + k] * x[k]   for r < rows.
//
// W is stored [rows][in]: the "[out, in]" layout, used by the attention
// output projection [d_model, heads, head_dim]. `y` must not overlap `x`.
void matvec(const float *W, size_t rows, size_t in, const float *x, float *y);

// ---- normalization --------------------------------------------------------

// out = x / sqrt(mean(x^2) + eps) * gamma.    out may be x.
void rmsnorm(const float *x, const float *gamma, size_t n, float eps, float *out);

// out = (x - mean) / sqrt(var + eps) * gamma + beta.    out may be x.
void layernorm(const float *x, const float *gamma, const float *beta, size_t n,
               float eps, float *out);

// ---- element-wise, in place -------------------------------------------------

// GELU, tanh approximation (jax.nn.gelu default):
//   0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
void gelu_tanh(float *x, size_t n);

// ELU: x > 0 ? x : exp(x) - 1.
void elu(float *x, size_t n);

// x = cap * tanh(x / cap): squashes values into (-cap, cap), keeps order.
void softcap(float *x, size_t n, float cap);

// x = exp(x - max) / sum(exp(x - max)).  Subtracting the max first keeps
// exp() from overflowing; it does not change the result.
void softmax(float *x, size_t n);

// log(1 + exp(x)), without overflow for large x.
float softplus(float x);

// ---- sampling -----------------------------------------------------------------

// xoshiro256** random number generator (public domain algorithm).
// Our own instead of <random>: std::uniform_real_distribution gives
// different numbers on libc++ (macOS), libstdc++ (Linux) and MSVC
// (Windows), so the same seed would make different music per platform.
struct Rng {
    uint64_t s[4];
    explicit Rng(uint64_t seed);
    uint64_t next();
    // Uniform in the OPEN interval (0, 1): never exactly 0 or 1, so
    // log() of it is always finite.
    double uniform();
};

// Index of the largest value; the first one if several are equal
// (same rule as jnp.argmax).
size_t argmax(const float *x, size_t n);

// Pick an index from `logits` the way Google's sampler does:
//   1. keep only values >= the k-th largest (ties at the k-th value are
//      kept too), drop the rest;
//   2. return argmax(logit + temperature * gumbel_noise).
// That is the "Gumbel-max trick": it draws from softmax(logits / T)
// restricted to the top k. temperature == 0 gives plain argmax (greedy),
// which uses no random numbers at all. k must be 1..kMaxTopK.
constexpr size_t kMaxTopK = 256;
size_t sample_topk(const float *logits, size_t n, size_t k, float temperature, Rng &rng);

} // namespace mrt2
