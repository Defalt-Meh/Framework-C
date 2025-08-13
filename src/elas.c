#define ELAS_BUILD
#include "elas.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#if defined(_OPENMP)
  #include <omp.h>
#endif

/*──────────────────────────────────────────────────────────────────────
 * Globals & error state
 *────────────────────────────────────────────────────────────────────*/
static elas_ctx_t g_ctx = {
    /* block_m,n,k */ 64, 64, 64,
    /* n_threads   */ 0,
    /* affinity    */ 1,
    /* steal       */ 1,
    /* packing     */ 0
};

#if defined(_MSC_VER)
  #define ELAS_THREAD_LOCAL __declspec(thread)
#else
  #define ELAS_THREAD_LOCAL __thread
#endif
static ELAS_THREAD_LOCAL elas_status_t g_last = ELAS_OK;

/* Optional shared workspace (for pack buffers); single global for simplicity */
static void*  g_ws_ptr   = NULL;
static size_t g_ws_bytes = 0;

/*──────────────────────────────────────────────────────────────────────
 * Utility
 *────────────────────────────────────────────────────────────────────*/
ELAS_INLINE int elas_div_ceil(const int a, const int b) { return (a + b - 1) / b; }

static ELAS_INLINE int clamp_pos(const int v) { return v < 0 ? 0 : v; }

ELAS_EXPORT elas_status_t elas_last_error(void) { return g_last; }

ELAS_EXPORT void elas_init(int bm, int bn, int bk, int nthr) {
    if (bm > 0) g_ctx.block_m = bm;
    if (bn > 0) g_ctx.block_n = bn;
    if (bk > 0) g_ctx.block_k = bk;
    if (nthr >= 0) g_ctx.n_threads = nthr;
}

ELAS_EXPORT const elas_ctx_t* elas_get_ctx(void) { return &g_ctx; }

ELAS_EXPORT void elas_set_force_packing(int onoff) { g_ctx.force_packing = onoff ? 1 : 0; }

ELAS_EXPORT void elas_set_workspace(void *buf, size_t bytes) {
    g_ws_ptr   = buf;
    g_ws_bytes = bytes;
}

/* Basic runtime feature detection (compile-time + coarse CPUID) */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
  #if defined(__GNUC__) || defined(__clang__)
    #include <cpuid.h>
  #endif
#endif

ELAS_EXPORT unsigned elas_features(void) {
    unsigned f = 0;
    f |= ELAS_FEAT_SCALAR;
#if defined(_OPENMP)
    f |= ELAS_FEAT_THREADS;
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    f |= ELAS_FEAT_NEON;
#endif
#if defined(__AVX2__) || defined(_MSC_VER) && defined(__AVX2__)
    f |= ELAS_FEAT_AVX2;
#elif defined(__SSE2__) || (defined(_M_IX86_FP) && _M_IX86_FP>=2)
    f |= ELAS_FEAT_SSE;
#endif
#if defined(__AVX512F__)
    f |= ELAS_FEAT_AVX512;
#endif
    return f;
}

/*──────────────────────────────────────────────────────────────────────
 * Indexing helpers for ROW_MAJOR (hot path)
 *  - lda/ldb/ldc are leading dimensions (>= N for row-major)
 *  - For COL_MAJOR, we swap notions via logical transpose
 *────────────────────────────────────────────────────────────────────*/
#define A_ROWMAJOR(i,j,lda) ((i)*(lda) + (j))
#define B_ROWMAJOR(i,j,ldb) ((i)*(ldb) + (j))
#define C_ROWMAJOR(i,j,ldc) ((i)*(ldc) + (j))

/*──────────────────────────────────────────────────────────────────────
 * SGEMV (row-major)
 *  y := alpha * A * x + beta * y        (NoTrans)
 *  y := alpha * A^T * x + beta * y      (Trans)
 *────────────────────────────────────────────────────────────────────*/
static void sgemv_rowmajor_nt(int M, int N, float alpha,
                              const float * ELAS_RESTRICT A, int lda,
                              const float * ELAS_RESTRICT X, int incX,
                              float beta,
                              float * ELAS_RESTRICT Y, int incY)
{
    if (beta == 0.0f) {
        for (int i=0; i<M; ++i) {
            float acc = 0.0f;
            const float* a = A + (size_t)i*lda;
            const float* x = X;
            for (int j=0; j<N; ++j) {
                acc += a[j] * x[ (size_t)j*incX ];
            }
            Y[(size_t)i*incY] = alpha * acc;
        }
    } else if (beta == 1.0f) {
        for (int i=0; i<M; ++i) {
            float acc = 0.0f;
            const float* a = A + (size_t)i*lda;
            const float* x = X;
            for (int j=0; j<N; ++j) {
                acc += a[j] * x[(size_t)j*incX];
            }
            Y[(size_t)i*incY] += alpha * acc;
        }
    } else {
        for (int i=0; i<M; ++i) {
            float acc = 0.0f;
            const float* a = A + (size_t)i*lda;
            const float* x = X;
            for (int j=0; j<N; ++j) {
                acc += a[j] * x[(size_t)j*incX];
            }
            Y[(size_t)i*incY] = beta * Y[(size_t)i*incY] + alpha * acc;
        }
    }
}

static void sgemv_rowmajor_t(int M, int N, float alpha,
                             const float * ELAS_RESTRICT A, int lda,
                             const float * ELAS_RESTRICT X, int incX,
                             float beta,
                             float * ELAS_RESTRICT Y, int incY)
{
    /* y:N, x:M, A^T: N×M */
    if (beta == 0.0f) {
        for (int j=0; j<N; ++j) {
            float acc = 0.0f;
            const float* x = X;
            for (int i=0; i<M; ++i) {
                acc += A[(size_t)i*lda + j] * x[(size_t)i*incX];
            }
            Y[(size_t)j*incY] = alpha * acc;
        }
    } else if (beta == 1.0f) {
        for (int j=0; j<N; ++j) {
            float acc = 0.0f;
            const float* x = X;
            for (int i=0; i<M; ++i) {
                acc += A[(size_t)i*lda + j] * x[(size_t)i*incX];
            }
            Y[(size_t)j*incY] += alpha * acc;
        }
    } else {
        for (int j=0; j<N; ++j) {
            float acc = 0.0f;
            const float* x = X;
            for (int i=0; i<M; ++i) {
                acc += A[(size_t)i*lda + j] * x[(size_t)i*incX];
            }
            Y[(size_t)j*incY] = beta * Y[(size_t)j*incY] + alpha * acc;
        }
    }
}

ELAS_EXPORT void elas_sgemv(const int Order,  const int TransA,
                            const int M,      const int N,
                            const float alpha,
                            const float * ELAS_RESTRICT A, const int lda,
                            const float * ELAS_RESTRICT X, const int incX,
                            const float beta,
                            float * ELAS_RESTRICT Y,       const int incY)
{
    g_last = ELAS_OK;
    if (M<0 || N<0 || lda < (TransA==ELAS_NO_TRANS ? clamp_pos(N) : clamp_pos(N)) ||
        incX==0 || incY==0 || !A || !X || !Y) { g_last = ELAS_BAD_ARG; return; }

    /* Normalize to ROW_MAJOR hot path by swapping M/N for COL_MAJOR */
    if (Order == ELAS_COL_MAJOR) {
        /* A is MxN column-major → treat as row-major with swapped roles */
        if (TransA == ELAS_NO_TRANS) {
            /* y := alpha * A * x  → row-major Trans */
            sgemv_rowmajor_t(N, M, alpha, A, lda, X, incX, beta, Y, incY);
        } else {
            sgemv_rowmajor_nt(N, M, alpha, A, lda, X, incX, beta, Y, incY);
        }
        return;
    }

    const int t = (TransA == ELAS_TRANS || TransA == ELAS_CONJ_TRANS);
    if (!t) sgemv_rowmajor_nt(M,N,alpha,A,lda,X,incX,beta,Y,incY);
    else    sgemv_rowmajor_t(M,N,alpha,A,lda,X,incX,beta,Y,incY);
}

/*──────────────────────────────────────────────────────────────────────
 * SGER (row-major): A := A + alpha * x * y^T
 *────────────────────────────────────────────────────────────────────*/
ELAS_EXPORT void elas_sger(const int Order,
                           const int M, const int N,
                           const float alpha,
                           const float * ELAS_RESTRICT X, const int incX,
                           const float * ELAS_RESTRICT Y, const int incY,
                           float * ELAS_RESTRICT A, const int lda)
{
    g_last = ELAS_OK;
    if (M<0 || N<0 || lda < clamp_pos(N) || incX==0 || incY==0 || !A || !X || !Y) {
        g_last = ELAS_BAD_ARG; return;
    }

    if (Order == ELAS_COL_MAJOR) {
        /* Column-major outer: A_ij += alpha * x_i * y_j
           In row-major view, same indexing if we interpret lda accordingly. */
        for (int i=0; i<M; ++i) {
            const float xi = X[(size_t)i*incX];
            float* arow = A + (size_t)i*lda;
            for (int j=0; j<N; ++j) {
                arow[j] += alpha * xi * Y[(size_t)j*incY];
            }
        }
        return;
    }

    /* ROW_MAJOR (hot path) */
    for (int i=0; i<M; ++i) {
        const float xi = X[(size_t)i*incX];
        float* arow = A + (size_t)i*lda;
        for (int j=0; j<N; ++j) {
            arow[j] += alpha * xi * Y[(size_t)j*incY];
        }
    }
}

/*──────────────────────────────────────────────────────────────────────
 * SGEMM (row-major)
 *  C := alpha * op(A) * op(B) + beta * C
 *  - Supports NoTrans/Trans (ConjTrans == Trans for float)
 *  - Fastpaths: beta==0 or beta==1; M==1 or N==1 routed to GEMV/GER-like
 *  - Blocked triple loop; OpenMP parallel over N-tiles (cache-friendly)
 *────────────────────────────────────────────────────────────────────*/
static ELAS_INLINE void scale_C_block(float *C, int ldc, int M, int N, float beta)
{
    if (beta == 1.0f) return;
    if (beta == 0.0f) {
        for (int i=0;i<M;++i) {
            memset(C + (size_t)i*ldc, 0, (size_t)N*sizeof(float));
        }
        return;
    }
    for (int i=0;i<M;++i) {
        float* c = C + (size_t)i*ldc;
        for (int j=0;j<N;++j) c[j] *= beta;
    }
}

/* Micro-kernel: C(ib:ib+mblk, jb:jb+nblk) += alpha * A_block * B_block
 * Plain C with strong hints; compilers will vectorize nicely.
 */
static ELAS_INLINE void gemm_kernel_nn(int M, int N, int K,
                                       float alpha,
                                       const float * ELAS_RESTRICT A, int lda,
                                       const float * ELAS_RESTRICT B, int ldb,
                                       float * ELAS_RESTRICT C, int ldc)
{
    for (int i=0; i<M; ++i) {
        float* c = C + (size_t)i*ldc;
        const float* a = A + (size_t)i*lda;
        for (int k=0; k<K; ++k) {
            const float aik = alpha * a[k];
            const float* b = B + (size_t)k*ldb;
            /* Unroll N by 4 for a light boost */
            int j=0;
            for (; j<=N-4; j+=4) {
                c[j+0] += aik * b[j+0];
                c[j+1] += aik * b[j+1];
                c[j+2] += aik * b[j+2];
                c[j+3] += aik * b[j+3];
            }
            for (; j<N; ++j) c[j] += aik * b[j];
        }
    }
}

static ELAS_INLINE void gemm_kernel_nt(int M, int N, int K,
                                       float alpha,
                                       const float * ELAS_RESTRICT A, int lda,
                                       const float * ELAS_RESTRICT B, int ldb,
                                       float * ELAS_RESTRICT C, int ldc)
{
    /* op(B) = B^T : so B is N×K laid out as (row-major) with stride ldb */
    for (int i=0; i<M; ++i) {
        float* c = C + (size_t)i*ldc;
        const float* a = A + (size_t)i*lda;
        for (int j=0; j<N; ++j) {
            const float* bcol = B + (size_t)j*ldb;
            float acc = 0.0f;
            for (int k=0; k<K; ++k) acc += a[k] * bcol[k];
            c[j] += alpha * acc;
        }
    }
}

/* Core blocked GEMM for ROW_MAJOR with flags tA,tB */
static void sgemm_blocked_rowmajor(int tA, int tB,
                                   int M, int N, int K,
                                   float alpha,
                                   const float * ELAS_RESTRICT A, int lda,
                                   const float * ELAS_RESTRICT B, int ldb,
                                   float beta,
                                   float * ELAS_RESTRICT C, int ldc)
{
    const int BM = g_ctx.block_m, BN = g_ctx.block_n, BK = g_ctx.block_k;

    scale_C_block(C, ldc, M, N, beta);

    /* Choose a parallel strategy: chunk over N tiles (good for row-major) */
    int nthr = g_ctx.n_threads;
#if defined(_OPENMP)
    if (nthr == 0) nthr = omp_get_max_threads();
#else
    (void)nthr;
#endif

    /* Parallelize outer loops if OpenMP is available */
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic) num_threads(nthr) collapse(2)
#endif
    for (int jb = 0; jb < N; jb += BN) {
        for (int ib = 0; ib < M; ib += BM) {
            const int nb = (jb + BN <= N) ? BN : (N - jb);
            const int mb = (ib + BM <= M) ? BM : (M - ib);

            for (int kb = 0; kb < K; kb += BK) {
                const int kbw = (kb + BK <= K) ? BK : (K - kb);

                /* Views of blocks; handle transposes by swapping strides
                   We stay row-major logically. */
                const float* Ab = A + (size_t)(tA ? kb : ib) * (size_t)lda + (tA ? ib : kb);
                const float* Bb = B + (size_t)(tB ? jb : kb) * (size_t)ldb + (tB ? kb : jb);
                float*       Cb = C + (size_t)ib * (size_t)ldc + jb;

                if (!tA && !tB) {
                    /* A: mb×kbw (row-major stride lda), B: kbw×nb (row-major stride ldb) */
                    gemm_kernel_nn(mb, nb, kbw, alpha, Ab, lda, Bb, ldb, Cb, ldc);
                } else if (!tA && tB) {
                    /* B is effectively transposed: treat as N×K with stride ldb by column j */
                    gemm_kernel_nt(mb, nb, kbw, alpha, Ab, lda, Bb, ldb, Cb, ldc);
                } else if (tA && !tB) {
                    /* op(A)=A^T: swap roles—iterate C by rows, compute dot of A^T row with B row */
                    for (int i=0; i<mb; ++i) {
                        float* c = Cb + (size_t)i*ldc;
                        const float* arowT = Ab + (size_t)i; /* step by lda */
                        for (int k=0; k<kbw; ++k) {
                            const float aik = alpha * arowT[(size_t)k*lda];
                            const float* b = Bb + (size_t)k*ldb;
                            int j=0;
                            for (; j<=nb-4; j+=4) {
                                c[j+0] += aik * b[j+0];
                                c[j+1] += aik * b[j+1];
                                c[j+2] += aik * b[j+2];
                                c[j+3] += aik * b[j+3];
                            }
                            for (; j<nb; ++j) c[j] += aik * b[j];
                        }
                    }
                } else { /* tA && tB */
                    for (int i=0; i<mb; ++i) {
                        float* c = Cb + (size_t)i*ldc;
                        const float* arowT = Ab + (size_t)i;
                        for (int j=0; j<nb; ++j) {
                            const float* bcolT = Bb + (size_t)j;
                            float acc = 0.0f;
                            for (int k=0; k<kbw; ++k) {
                                acc += arowT[(size_t)k*lda] * bcolT[(size_t)k*ldb];
                            }
                            c[j] += alpha * acc;
                        }
                    }
                }
            } /* kb */
        } /* ib */
    } /* jb */
}

ELAS_EXPORT void elas_sgemm(const int Order,  const int TransA, const int TransB,
                            const int M,      const int N,      const int K,
                            const float alpha,
                            const float * ELAS_RESTRICT A, const int lda,
                            const float * ELAS_RESTRICT B, const int ldb,
                            const float beta,
                            float * ELAS_RESTRICT C,       const int ldc)
{
    g_last = ELAS_OK;
    if (M<0 || N<0 || K<0 || !A || !B || !C) { g_last = ELAS_BAD_ARG; return; }
    if (Order != ELAS_ROW_MAJOR && Order != ELAS_COL_MAJOR) { g_last = ELAS_BAD_ARG; return; }

    /* Normalize Trans flags */
    const int tA = (TransA == ELAS_TRANS || TransA == ELAS_CONJ_TRANS);
    const int tB = (TransB == ELAS_TRANS || TransB == ELAS_CONJ_TRANS);

    /* Quick shape/lda checks (conservative) */
    if (Order == ELAS_ROW_MAJOR) {
        const int a_cols = tA ? M : K;
        const int b_cols = tB ? K : N;
        if (lda < clamp_pos(a_cols) || ldb < clamp_pos(b_cols) || ldc < clamp_pos(N)) {
            g_last = ELAS_BAD_ARG; return;
        }
    } else { /* COL_MAJOR: check against swapped meanings */
        const int a_rows = tA ? K : M;
        const int b_rows = tB ? N : K;
        if (lda < clamp_pos(a_rows) || ldb < clamp_pos(b_rows) || ldc < clamp_pos(M)) {
            g_last = ELAS_BAD_ARG; return;
        }
    }

    /* Special-case skinny shapes common in NN:
       - M==1 → row-vector x op(B)   (treat as GEMV over columns)
       - N==1 → op(A) x column-vector (GEMV)
    */
    if (Order == ELAS_ROW_MAJOR) {
        if (M == 1) {
            /* C[0,:] = alpha * a1*op(B) + beta*C[0,:]  */
            /* Build a temporary y over N with SGEMV */
            /* For op(A) with M==1: a is 1xK (NoTrans) or Kx1 (Trans) */
            if (!tA) {
                /* y := alpha * (1×K) * op(B) + beta*y → SGEMV over op(B)^T */
                if (!tB) {
                    /* y := alpha * a * B + beta*y  <=> y_j = sum_k a_k * B_kj */
                    sgemv_rowmajor_t(K, N, alpha, B, ldb, A, 1, beta, C, 1);
                } else {
                    /* y := alpha * a * B^T → treat B^T as (N×K), NoTrans */
                    sgemv_rowmajor_nt(N, K, alpha, B, ldb, A, 1, beta, C, 1);
                }
            } else {
                /* a is K×1: same as above with X swapped */
                if (!tB) {
                    sgemv_rowmajor_t(K, N, alpha, B, ldb, A, lda, beta, C, 1);
                } else {
                    sgemv_rowmajor_nt(N, K, alpha, B, ldb, A, lda, beta, C, 1);
                }
            }
            return;
        }
        if (N == 1) {
            /* single column: treat as SGEMV NoTrans on op(A) with x = op(B)[:,0] */
            /* Build x of length K from B’s first column/row depending on tB */
            /* We just perform the mat-vec directly to avoid temp x. */
            if (!tA && !tB) { /* C := alpha*A*Bcol + beta*C */
                const float* x = B;
                for (int i=0; i<M; ++i) {
                    const float* a = A + (size_t)i*lda;
                    float acc = 0.0f;
                    for (int k=0; k<K; ++k) acc += a[k] * x[(size_t)k*ldb];
                    C[i*ldc] = (beta==0.0f?0.0f:beta*C[i*ldc]) + alpha*acc;
                }
                return;
            }
            /* Other tA/tB combos fall through to general path for clarity */
        }
    } else {
        /* COL_MAJOR: swap roles and forward to ROW_MAJOR by transposing the op triple.
           We map:  C := a*Aop * Bop + b*C   (col-major)
           to:     C^T := a*(Bop^T * Aop^T) + b*C^T   (row-major) */
        /* Swap M <-> N, A <-> B, tA <-> tB, and use row-major core on the transposed views. */
        const int M2 = N, N2 = M, K2 = K;
        const int tA2 = tB, tB2 = tA;
        /* For leading dims in row-major view, we use the original ldb/lda/ldc. */
        /* Operate on buffers directly without physical transpose by adjusted indexing in the core. */
        /* Simpler route: call core with Order==ROW_MAJOR but provide shapes swapped and strides as given.
           The core functions assume row-major contiguous by rows, which aligns with the logical transpose mapping. */
        sgemm_blocked_rowmajor(tA2, tB2, M2, N2, K2, alpha, B, ldb, A, lda, beta, C, ldc);
        return;
    }

    /* ROW_MAJOR general path */
    sgemm_blocked_rowmajor(tA, tB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
