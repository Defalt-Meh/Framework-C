/*====================================================================
 * ELAS (Embedded Linear Algebra Subsystem) – Minimal BLAS-1/2/3 subset
 * ------------------------------------------------------------------
 * Purpose  : Provide a zero-dependency, drop-in replacement for the few
 *            CBLAS routines used in small NN kernels (sgemm, sgemv, sger).
 * Audience : Performance-minded embedded/HPC devs.
 * License  : MIT
 *====================================================================*/
#ifndef ELAS_H
#define ELAS_H 1
#include <stddef.h>

/*──────────────────────────────────────────────────────────────────────
 * 0 ▸ Export / intrinsics / alignment
 *────────────────────────────────────────────────────────────────────*/
#if defined(_WIN32)
#  ifdef ELAS_BUILD
#    define ELAS_EXPORT __declspec(dllexport)
#  else
#    define ELAS_EXPORT __declspec(dllimport)
#  endif
#else
#  define ELAS_EXPORT __attribute__((visibility("default")))
#endif

#if defined(__clang__) || defined(__GNUC__)
#   define ELAS_RESTRICT  __restrict__
#   define ELAS_INLINE    static inline __attribute__((always_inline))
#   define ELAS_ALIGNED16 __attribute__((aligned(16)))
#   define ELAS_PREFETCH(addr, rw, locality) __builtin_prefetch((addr),(rw),(locality))
#   define ELAS_ASSUME_ALIGNED(p, n) (__builtin_assume_aligned((p),(n)))
#else
#   define ELAS_RESTRICT
#   define ELAS_INLINE    static inline
#   define ELAS_ALIGNED16
#   define ELAS_PREFETCH(addr, rw, locality)
#   define ELAS_ASSUME_ALIGNED(p, n) (p)
#endif

/* Stack alignment hint for small buffers (adjust to 32/64 for AVX-512/SVE) */
#define ELAS_STACK_ALIGN 16

/* Version & feature reporting (runtime) */
#define ELAS_VERSION_MAJOR 0
#define ELAS_VERSION_MINOR 2
#define ELAS_VERSION_PATCH 0
ELAS_INLINE int elas_version(void){
    return (ELAS_VERSION_MAJOR<<16)|(ELAS_VERSION_MINOR<<8)|ELAS_VERSION_PATCH;
}
enum {
    ELAS_FEAT_SCALAR = 1<<0,
    ELAS_FEAT_SSE    = 1<<1,
    ELAS_FEAT_AVX2   = 1<<2,
    ELAS_FEAT_AVX512 = 1<<3,
    ELAS_FEAT_NEON   = 1<<4,
    ELAS_FEAT_THREADS= 1<<5
};

/*──────────────────────────────────────────────────────────────────────
 * 1 ▸ Enumerations mirroring CBLAS (numeric parity maintained)
 *────────────────────────────────────────────────────────────────────*/
enum { ELAS_ROW_MAJOR = 101, ELAS_COL_MAJOR = 102 };
enum { ELAS_NO_TRANS  = 111, ELAS_TRANS     = 112, ELAS_CONJ_TRANS = 113 };

/*──────────────────────────────────────────────────────────────────────
 * 2 ▸ Global tuning context (tile sizes, threading policy, packing)
 *────────────────────────────────────────────────────────────────────*/
typedef struct {
    int block_m;         /* default 64  */
    int block_n;         /* default 64  */
    int block_k;         /* default 64  */
    int n_threads;       /* 0 -> auto   */
    int prefer_affinity; /* hint only   */
    int steal_chunks;    /* hint only   */
    int force_packing;   /* 0/1         */
} elas_ctx_t;

#ifdef __cplusplus
extern "C" {
#endif

/* Context control */
ELAS_EXPORT void              elas_init(int block_m, int block_n, int block_k, int n_threads);
ELAS_EXPORT const elas_ctx_t* elas_get_ctx(void);
ELAS_EXPORT void              elas_set_force_packing(int onoff);
ELAS_EXPORT void              elas_set_workspace(void *buf, size_t bytes); /* optional */
ELAS_EXPORT unsigned          elas_features(void); /* runtime CPU/library feature bits */

/*──────────────────────────────────────────────────────────────────────
 * 3 ▸ Public BLAS subset (single-precision)
 *    Signatures mirror Netlib CBLAS to allow:
 *      #define cblas_sgemm elas_sgemm
 *────────────────────────────────────────────────────────────────────*/
ELAS_EXPORT void elas_sgemm(const int Order,  const int TransA, const int TransB,
                            const int M,      const int N,      const int K,
                            const float alpha,
                            const float * ELAS_RESTRICT A, const int lda,
                            const float * ELAS_RESTRICT B, const int ldb,
                            const float beta,
                            float * ELAS_RESTRICT C,       const int ldc);

ELAS_EXPORT void elas_sgemv(const int Order,  const int TransA,
                            const int M,      const int N,
                            const float alpha,
                            const float * ELAS_RESTRICT A, const int lda,
                            const float * ELAS_RESTRICT X, const int incX,
                            const float beta,
                            float * ELAS_RESTRICT Y,       const int incY);

ELAS_EXPORT void elas_sger (const int Order,
                            const int M,      const int N,
                            const float alpha,
                            const float * ELAS_RESTRICT X, const int incX,
                            const float * ELAS_RESTRICT Y, const int incY,
                            float * ELAS_RESTRICT A,       const int lda);

/* Optional: last-error (ELAS never aborts; caller decides policy) */
typedef enum { ELAS_OK=0, ELAS_BAD_ARG=1, ELAS_UNSUPPORTED=2 } elas_status_t;
ELAS_EXPORT elas_status_t elas_last_error(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

/*──────────────────────────────────────────────────────────────────────
 * 4 ▸ Temporary compatibility shims – remove once migrated
 *────────────────────────────────────────────────────────────────────*/
#define cblas_sgemm  elas_sgemm
#define cblas_sgemv  elas_sgemv
#define cblas_sger   elas_sger

/* Behavioral contract (for speed):
 *  - ELAS_CONJ_TRANS == ELAS_TRANS for float
 *  - beta∈{0,1} uses true fastpaths
 *  - ROW_MAJOR hot path; COL_MAJOR handled via logical transpose/strides
 *  - M==1 or N==1 in sgemm may tail-call specialized kernels
 */

#endif /* ELAS_H */
