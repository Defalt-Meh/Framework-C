/*====================================================================
 * ELAS (Embedded Linear Algebra Subsystem) – Minimal BLAS-1/2/3 subset
 * ------------------------------------------------------------------
 * Purpose  : Provide a *zero-dependency*, drop-in replacement for the handful
 *            of CBLAS routines typically used inside small NN kernels.
 * Audience : Performance-conscious embedded / HPC practitioners who would
 *            rather own their math than drag in a heavyweight BLAS distro.
 * License  : MIT (see corresponding .c file for full text)
 * Author   : Defalt – Just a gooner, 2025
 *====================================================================*/
#ifndef ELAS_H
#define ELAS_H 1

/*──────────────────────────────────────────────────────────────────────
 * 0 ▸ Portable compiler intrinsics & attributes
 *    (Change these once and every kernel downstream benefits.)
 *────────────────────────────────────────────────────────────────────*/
#if defined(__clang__) || defined(__GNUC__)
#   define ELAS_RESTRICT  __restrict__
#   define ELAS_INLINE    static inline __attribute__((always_inline))
#   define ELAS_ALIGNED16 __attribute__((aligned(16)))
#   define ELAS_PREFETCH(addr, rw, locality) __builtin_prefetch((addr),(rw),(locality))
#else
#   define ELAS_RESTRICT
#   define ELAS_INLINE    static inline
#   define ELAS_ALIGNED16
#   define ELAS_PREFETCH(addr, rw, locality)
#endif

/*──────────────────────────────────────────────────────────────────────
 * 1 ▸ Enumerations mirroring CBLAS (so we can #define-redirect safely)
 *────────────────────────────────────────────────────────────────────*/
enum {
    ELAS_ROW_MAJOR = 101,
    ELAS_COL_MAJOR = 102
};

enum {
    ELAS_NO_TRANS      = 111,
    ELAS_TRANS         = 112,
    ELAS_CONJ_TRANS    = 113
};

/*──────────────────────────────────────────────────────────────────────
 * 2 ▸ Optional global tuning context (block sizes, threads, etc.)
 *     Feel free to tweak/extend; keeping it here means every TU sees
 *     exactly the same constants without a bazillion #defines.
 *────────────────────────────────────────────────────────────────────*/
typedef struct {
    int block_m;   /* M-dimension tile (e.g., 64) */
    int block_n;   /* N-dimension tile (e.g., 64) */
    int block_k;   /* K-dimension tile (e.g., 64) */
    int n_threads; /* #logical threads to spawn – 0 → auto */
} elas_ctx_t;

/* Externally visible, defined in elas_blas.c. Initialise *once* via
 * elas_init() before the first kernel call if you want non-defaults. */
extern elas_ctx_t g_elas_ctx;

/** Initialise / reconfigure global ELAS context.
 *  Passing 0 for a field keeps the previous setting.
 */
void elas_init(int block_m, int block_n, int block_k, int n_threads);

/*====================================================================
 * 3 ▸ Public API – single-precision only (sgemm, sgemv, sger)
 *     Arg-lists *mirror* Netlib CBLAS so you can temporarily do:
 *     #define cblas_sgemm elas_sgemm
 *====================================================================*/

#ifdef __cplusplus
extern "C" {
#endif

void elas_sgemm(const int Order,  const int TransA, const int TransB,
                const int M,      const int N,      const int K,
                const float alpha,
                const float * ELAS_RESTRICT A, const int lda,
                const float * ELAS_RESTRICT B, const int ldb,
                const float beta,
                float * ELAS_RESTRICT C,       const int ldc);

void elas_sgemv(const int Order,  const int TransA,
                const int M,      const int N,
                const float alpha,
                const float * ELAS_RESTRICT A, const int lda,
                const float * ELAS_RESTRICT X, const int incX,
                const float beta,
                float * ELAS_RESTRICT Y,       const int incY);

void elas_sger (const int Order,
                const int M,      const int N,
                const float alpha,
                const float * ELAS_RESTRICT X, const int incX,
                const float * ELAS_RESTRICT Y, const int incY,
                float * ELAS_RESTRICT A,       const int lda);

#ifdef __cplusplus
} /* extern "C" */
#endif

/*──────────────────────────────────────────────────────────────────────
 * 4 ▸ Temporary compatibility shims – remove once migrated
 *────────────────────────────────────────────────────────────────────*/
#define cblas_sgemm  elas_sgemm
#define cblas_sgemv  elas_sgemv
#define cblas_sger   elas_sger

/* Align frequently-used stack buffers to 16 B so the compiler is free
 * to vectorise with 128-bit SIMD (SSE/NEON). Adjust to 32/64 if you rely
 * on AVX-512 or SVE.
 */
#define ELAS_STACK_ALIGN 16

/* Utility: ceiling-division helper (a ÷ b rounded up) – used in tilers. */
ELAS_INLINE int elas_div_ceil(const int a, const int b) {
    return (a + b - 1) / b;
}

#endif /* ELAS_H */
