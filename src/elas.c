/*====================================================================
 * ELAS (Embedded Linear Algebra Subsystem) – Implementation
 * ------------------------------------------------------------------
 *  Single-precision, CPU-optimised replacements for the tiny BLAS
 *  subset used by the author’s NN framework.
 *  Kernels are deliberately *portable* C with optional OpenMP hooks.
 *  For deeper optimisation (AVX2/NEON/SVE micro-kernels) you can
 *  drop them in behind the same public prototypes.
 *====================================================================*/
#include "elas.h"
#include <stdlib.h>   /* malloc, free    */
#include <string.h>   /* memset, memcpy  */
#include <math.h>     /* fmaf, fabsf ... */

#ifdef _OPENMP
#   include <omp.h>
#endif

/*──────────────────────────────────────────────────────────────────────
 * 0 ▸ Global context – overridable at runtime via elas_init()
 *────────────────────────────────────────────────────────────────────*/
/* ★ Default tile sizes give respectable L1/L2 fit on Apple M-series &
 *   x86-64 with 32-KB L1-D, 512-KB L2. Tune for your cache.        */
elas_ctx_t g_elas_ctx = {
    .block_m   = 64,
    .block_n   = 64,
    .block_k   = 64,
    .n_threads = 0   /* 0 → let OpenMP decide */
};

void elas_init(int block_m, int block_n, int block_k, int n_threads)
{
    if (block_m > 0)   g_elas_ctx.block_m   = block_m;
    if (block_n > 0)   g_elas_ctx.block_n   = block_n;
    if (block_k > 0)   g_elas_ctx.block_k   = block_k;
    if (n_threads >=0) g_elas_ctx.n_threads = n_threads;
}

/*──────────────────────────────────────────────────────────────────────
 * 1 ▸ Helper utilities
 *────────────────────────────────────────────────────────────────────*/
ELAS_INLINE void elas_scale_matrix(float * ELAS_RESTRICT C,
                                   const int rows, const int cols, const int ldc,
                                   const float beta)
{
    if (beta == 1.0f) return;

#ifdef _OPENMP
#   pragma omp parallel for if(rows*cols > 4096)
#endif
    for (int j = 0; j < cols; ++j) {
        float *col = C + j*ldc;
        for (int i = 0; i < rows; ++i)
            col[i] *= beta;
    }
}

ELAS_INLINE void elas_zero_matrix(float * ELAS_RESTRICT C,
                                  const int rows, const int cols, const int ldc)
{
#ifdef _OPENMP
#   pragma omp parallel for if(rows*cols > 4096)
#endif
    for (int j = 0; j < cols; ++j) {
        memset(C + j*ldc, 0, sizeof(float)*rows);
    }
}

/*──────────────────────────────────────────────────────────────────────
 * 2 ▸ Core micro-kernel (MR×NR = 4×4). Keeps registers happy without
 *     assuming SIMD; modern compilers auto-vectorise the inner loop.
 *────────────────────────────────────────────────────────────────────*/
#define MR 4
#define NR 4

ELAS_INLINE void micro_kernel_4x4(int k,
                                  const float * ELAS_RESTRICT A, const int lda,
                                  const float * ELAS_RESTRICT B, const int ldb,
                                  float * ELAS_RESTRICT C,       const int ldc)
{
    float c00=0,c01=0,c02=0,c03=0;
    float c10=0,c11=0,c12=0,c13=0;
    float c20=0,c21=0,c22=0,c23=0;
    float c30=0,c31=0,c32=0,c33=0;

    for (int p=0; p<k; ++p) {
        const float a0 = A[0 + p*lda];
        const float a1 = A[1 + p*lda];
        const float a2 = A[2 + p*lda];
        const float a3 = A[3 + p*lda];

        const float *bptr = B + p;
        c00 += a0 * bptr[0*ldb];
        c01 += a0 * bptr[1*ldb];
        c02 += a0 * bptr[2*ldb];
        c03 += a0 * bptr[3*ldb];

        c10 += a1 * bptr[0*ldb];
        c11 += a1 * bptr[1*ldb];
        c12 += a1 * bptr[2*ldb];
        c13 += a1 * bptr[3*ldb];

        c20 += a2 * bptr[0*ldb];
        c21 += a2 * bptr[1*ldb];
        c22 += a2 * bptr[2*ldb];
        c23 += a2 * bptr[3*ldb];

        c30 += a3 * bptr[0*ldb];
        c31 += a3 * bptr[1*ldb];
        c32 += a3 * bptr[2*ldb];
        c33 += a3 * bptr[3*ldb];
    }

    C[0 + 0*ldc] += c00;  C[0 + 1*ldc] += c01;  C[0 + 2*ldc] += c02;  C[0 + 3*ldc] += c03;
    C[1 + 0*ldc] += c10;  C[1 + 1*ldc] += c11;  C[1 + 2*ldc] += c12;  C[1 + 3*ldc] += c13;
    C[2 + 0*ldc] += c20;  C[2 + 1*ldc] += c21;  C[2 + 2*ldc] += c22;  C[2 + 3*ldc] += c23;
    C[3 + 0*ldc] += c30;  C[3 + 1*ldc] += c31;  C[3 + 2*ldc] += c32;  C[3 + 3*ldc] += c33;
}

/*──────────────────────────────────────────────────────────────────────
 * 3 ▸ SGEMM – Column-major + optional tiling + OpenMP outer-loop split
 *────────────────────────────────────────────────────────────────────*/
void elas_sgemm(const int Order,  const int TransA, const int TransB,
                const int M,      const int N,      const int K,
                const float alpha,
                const float * ELAS_RESTRICT A, const int lda,
                const float * ELAS_RESTRICT B, const int ldb,
                const float beta,
                float * ELAS_RESTRICT C,       const int ldc)
{
    /* Only Column-major path is heavily optimised. Fallback converts
     * row-major to column-major indices on the fly. */
    const int colMajor = (Order == ELAS_COL_MAJOR);
    if (!colMajor) {
        /* Simple, row-major reference: Cᵣ = α Aᵣ·Bᵣ + β Cᵣ */
#ifdef _OPENMP
#       pragma omp parallel for schedule(static)
#endif
        for (int i=0;i<M;++i) {
            for (int j=0;j<N;++j) {
                float acc = 0.0f;
                for (int p=0;p<K;++p)
                    acc += A[i*lda + p] * B[p*ldb + j];
                C[i*ldc + j] = alpha*acc + beta*C[i*ldc + j];
            }
        }
        return;
    }

    /* Column-major fast path */
    if (beta == 0.0f)
        elas_zero_matrix(C, M, N, ldc);
    else if (beta != 1.0f)
        elas_scale_matrix(C, M, N, ldc, beta);

    const int bm = g_elas_ctx.block_m;
    const int bn = g_elas_ctx.block_n;
    const int bk = g_elas_ctx.block_k;

#ifdef _OPENMP
    const int req_threads = g_elas_ctx.n_threads;
    if (req_threads > 0) omp_set_num_threads(req_threads);
#   pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int jc=0; jc<N; jc+=bn) {
        for (int ic=0; ic<M; ic+=bm) {
            const int jb = (jc+bn > N) ? (N-jc) : bn;
            const int ib = (ic+bm > M) ? (M-ic) : bm;
            for (int pc=0; pc<K; pc+=bk) {
                const int kb = (pc+bk > K) ? (K-pc) : bk;

                /* Micro-tile loop */
                for (int j=0; j<jb; j+=NR) {
                    for (int i=0; i<ib; i+=MR) {
                        const int jr = (j+NR>jb)?(jb-j):NR;
                        const int ir = (i+MR>ib)?(ib-i):MR;
                        if (ir==MR && jr==NR) {
                            micro_kernel_4x4(kb,
                                A + (ic+i) + (pc)*lda,           lda,
                                B + (pc)   + (jc+j)*ldb,         ldb,
                                C + (ic+i) + (jc+j)*ldc,         ldc);
                        } else {
                            /* edge cases – fall back to scalar loop */
                            for (int jj=0; jj<jr; ++jj) {
                                for (int ii=0; ii<ir; ++ii) {
                                    float acc = 0.0f;
                                    for (int p=0; p<kb; ++p)
                                        acc += A[(ic+i+ii) + (pc+p)*lda] * B[(pc+p) + (jc+j+jj)*ldb];
                                    C[(ic+i+ii) + (jc+j+jj)*ldc] += acc;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/*──────────────────────────────────────────────────────────────────────
 * 4 ▸ SGEMV – y ← α·op(A)·x + β·y  (Column-major)   O(M·N)
 *────────────────────────────────────────────────────────────────────*/
void elas_sgemv(const int Order,  const int TransA,
                const int M,      const int N,
                const float alpha,
                const float * ELAS_RESTRICT A, const int lda,
                const float * ELAS_RESTRICT X, const int incX,
                const float beta,
                float * ELAS_RESTRICT Y,       const int incY)
{
    const int colMajor = (Order == ELAS_COL_MAJOR);
    if (!colMajor) {
        /* Row-major, non-transposed */
#ifdef _OPENMP
#       pragma omp parallel for schedule(static)
#endif
        for (int i=0;i<M;++i) {
            float acc = 0.0f;
            for (int j=0;j<N;++j)
                acc += A[i*lda + j] * X[j*incX];
            Y[i*incY] = alpha*acc + beta*Y[i*incY];
        }
        return;
    }

    if (TransA == ELAS_NO_TRANS) {
#ifdef _OPENMP
#       pragma omp parallel for schedule(static)
#endif
        for (int i=0;i<M;++i) {
            float acc = 0.0f;
            const float *Ai = A + i;
            for (int j=0;j<N;++j)
                acc += Ai[j*lda] * X[j*incX];
            Y[i*incY] = alpha*acc + beta*Y[i*incY];
        }
    } else { /* Aᵗ · x */
#ifdef _OPENMP
#       pragma omp parallel for schedule(static)
#endif
        for (int j=0;j<N;++j) {
            float acc = 0.0f;
            const float *Aj = A + j*lda;
            for (int i=0;i<M;++i)
                acc += Aj[i] * X[i*incX];
            Y[j*incY] = alpha*acc + beta*Y[j*incY];
        }
    }
}

/*──────────────────────────────────────────────────────────────────────
 * 5 ▸ SGER – A ← α·x·yᵗ + A   (outer-product)       O(M·N)
 *────────────────────────────────────────────────────────────────────*/
void elas_sger(const int Order,
               const int M,      const int N,
               const float alpha,
               const float * ELAS_RESTRICT X, const int incX,
               const float * ELAS_RESTRICT Y, const int incY,
               float * ELAS_RESTRICT A,       const int lda)
{
    const int colMajor = (Order == ELAS_COL_MAJOR);
    if (!colMajor) {
        /* Row-major path */
#ifdef _OPENMP
#       pragma omp parallel for schedule(static)
#endif
        for (int i=0;i<M;++i) {
            const float xi = alpha * X[i*incX];
            float *Ai = A + i*lda;
            for (int j=0;j<N;++j)
                Ai[j] += xi * Y[j*incY];
        }
        return;
    }

#ifdef _OPENMP
#   pragma omp parallel for schedule(static)
#endif
    for (int j=0;j<N;++j) {
        const float yj = alpha * Y[j*incY];
        float *Aj = A + j*lda;
        for (int i=0;i<M;++i)
            Aj[i] += X[i*incX] * yj;
    }
}
