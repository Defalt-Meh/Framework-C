/* ============================================================================
 * Neural network forward utilities (single-hidden-layer version)
 * ----------------------------------------------------------------------------
 * This translation unit provides:
 *   - Lightweight helpers for bias-pointer arithmetic
 *   - A fast, branch-light sigmoid approximation and its derivative
 *   - Numerically stable softmax + cross-entropy utilities
 *   - A portable forward pass (with optional BLAS acceleration)
 *
 * Notation (row-major storage throughout):
 *   - in      ∈ R^{nips}                : input row-vector (single sample)
 *   - w       ∈ R^{nhid × nips}         : input→hidden weights (row-major)
 *   - x       ∈ R^{nops × nhid}         : hidden→output weights (row-major)
 *   - b       : contiguous bias buffer; hidden biases then output biases
 *   - h       ∈ R^{nhid}                : hidden activations
 *   - o       ∈ R^{nops}                : output activations (post-epilogue)
 *
 * Complexity:
 *   - Hidden matvec  : Θ(nhid · nips)
 *   - Output matvec  : Θ(nops · nhid)
 *   - Softmax        : Θ(nops)
 *
 * Numerical notes:
 *   - Softmax uses max-subtraction for stability (avoids overflow in exp).
 *   - Cross-entropy uses a small guard to avoid log(0).
 *   - The "fast_sigmoid" trades accuracy for speed (one fabsf, one divide).
 *     It is monotone, smooth, with bounded output in (0,1).
 * ==========================================================================*/

#include "nn.h"
#include <stdlib.h>
#include <math.h>          /* fabsf */
/* The following bias-pointer macros express the assumed bias layout.
 * They are pure pointer arithmetic—no allocation or bounds checks.    */
#define BH_PTR(nn)  ((nn).b)                             /* hidden biases: nhid  */
#define BO_PTR(nn)  ((nn).b + (nn).nhid)                 /* output biases: nops  */

/* For two-hidden-layer variants, the same contiguous bias buffer is sliced.
 * This file uses only one hidden layer, but the macros generalize.      */
#define BH1_PTR(nn) ((nn).b)                             /* nhid   */
#define BH2_PTR(nn) ((nn).b + (nn).nhid)                 /* nhid2  */
#define BO2_PTR(nn) ((nn).b + (nn).nhid + (nn).nhid2)    /* nops   */

/* Compile-time guard for activation choice on the hidden layer.
 * If FWC_RELU_HID is NOT defined, we raise an error at compile-time,
 * effectively preventing accidental mismatch between training configs.  */
#ifndef FWC_RELU_HID
#error "FWC_RELU_HID not defined: you're training with sigmoid!"
#endif



/* ───────────────────── OPENMP selection ─────────────────── */
/* Optional inclusion: if compiled with -fopenmp, expose omp timing/
 * threading utilities. There is no direct OpenMP use in this snippet,
 * but other translation units may rely on the header being present.    */
#ifdef _OPENMP
    #include <omp.h>
#endif
/* ───────────────────────────────────────────────────────────────── */

/* ───────────────────── BLAS backend selection ─────────────────── */
/* Matrix-matrix/vector kernels:
 *  - If ELAS_LOCAL is defined, include a header-only fallback (ELAS).
 *  - Else, prefer platform BLAS: Accelerate on macOS, OpenBLAS/system on others.
 * The forward below optionally calls cblas_sgemm for 1×K by K×N products
 * (still effective; BLAS handles small M dimension).                    */
#ifdef ELAS_LOCAL
    #include "elas.h"          /* header-only fallback */
#else
    #ifdef __APPLE__
        #include <Accelerate/Accelerate.h>   /* CBLAS via Accelerate */
    #else
        #include <cblas.h>                   /* OpenBLAS / system BLAS */
    #endif
#endif
/* ───────────────────────────────────────────────────────────────── */

/* ─────────── Static forward declarations (unchanged) ──────────── */
/* toterr:  typically used to summarize per-sample losses (not shown here)
 * pderr :  placeholder for a derivative helper (not shown here)
 * frand :  uniform RNG helper on [0,1)                                   */
static float toterr(const float *tg, const float *o, int size);
static float frand (void);

/* ─────────────—— Helper implementations (unchanged) —──────────── */
/* frand: uniform(0,1) with a cached reciprocal to avoid a division
 * in the hot path. Note RAND_MAX is implementation-defined; the
 * resulting distribution is adequate for initialization/jittering.   */
static inline float frand(void)
{
    /* And lo, the reciprocal of RAND_MAX was preserved,
     * that no costly division plague the hosts of code. */ 
    static const float inv_rand_max = 1.0f / RAND_MAX;
    return rand() * inv_rand_max;
}

/* ───────────────────────────────────────────────────────────────── */
/* ────────────────────── OPTIMIZED HELPERS ──────────────────────---*/
/* ───────────────────────────────────────────────────────────────── */

/* fast_sigmoid: σ_fast(x) = 0.5 * (x / (1 + |x|)) + 0.5
 *  - Smooth, strictly increasing, bounded in (0,1).
 *  - Approximates logistic σ(x) = 1/(1+e^{-x}) but cheaper:
 *      • one fabsf, one add, one div, one FMA.
 *  - The FMA improves precision by evaluating 0.5*(x*inv)+0.5 in
 *    a single rounding step when hardware FMA is available.
 *  - Good fit when exact probabilities are not critical, e.g., as an
 *    output squashing function under L2 loss or for quick baselines.     */
static inline float fast_sigmoid(float x)
{
    const float a   = fabsf(x);
    const float den = 1.0f + a;
    const float inv = 1.0f / den;                  // ok with -ffast-math
    return fmaf(x, 0.5f * inv, 0.5f);              // 0.5*(x*inv) + 0.5
}

/* softmax_ce_fused_row:
 *  - Computes softmax probabilities in a numerically stable way and
 *    the per-row cross-entropy loss against a one-hot target y.
 *  - Also writes the "delta" = (softmax - y) used by backprop into 'delta'.
 *  - Inputs:
 *      logits[0..n-1] : pre-softmax scores (will be read; can be overwritten)
 *      y[0..n-1]      : one-hot target vector (0/1, sum=1)
 *      bias           : scalar bias added to all logits (common in some designs)
 *      write_probs    : if nonzero, logits is overwritten by probabilities
 *  - Returns:
 *      CE(y, p) = -∑_i y_i log p_i  for the row.
 *
 * Numerical stability:
 *  - Subtract m = max_i (logits_i + bias) before exponentiation.
 *  - logZ = log ∑_i exp(logits_i + bias - m) + m ensures stable log-normalizer.
 *
 * Memory / aliasing:
 *  - delta may alias neither 'y' nor 'logits' (unless caller intends so).
 *  - 'restrict' hints to the compiler that pointers do not alias.            */
static inline float softmax_ce_fused_row(float *restrict logits,
                                         const float *restrict y,
                                         float *restrict delta,
                                         const int n,
                                         const float bias,
                                         const int write_probs)
{
    /* Find the max for stability: avoids exp overflow and improves conditioning. */
    float m = logits[0] + bias;
    for (int i = 1; i < n; ++i) {
        const float z = logits[i] + bias;
        if (z > m) m = z;
    }
    /* Compute the log-partition function: logZ = log ∑ exp(z - m) + m. */
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += expf((logits[i] + bias) - m);
    const float logZ = logf(s) + m;

    /* Cross-entropy and deltas (softmax - y).
     * We add a tiny guard to log to keep CE finite when p_i == 0 numerically. */
    float loss = 0.0f;
    const float inv_guard = 1e-20f;  /* avoid log(0) */
    for (int i = 0; i < n; ++i) {
        const float p = expf((logits[i] + bias) - logZ);  /* softmax component */
        delta[i] = p - y[i];                               /* backprop signal  */
        if (write_probs) logits[i] = p;                    /* optional writeback */
        loss -= y[i] * logf(p + inv_guard);                /* CE contribution  */
    }
    return loss;
}

/* ───────────────────────────────────────────────────────────────── */

/* fprop: Single-sample forward pass.
 *  - Computes hidden pre-activations via a matrix-vector product, then applies
 *    an activation (ReLU if FWC_RELU_HID is defined; otherwise fast sigmoid).
 *  - Computes output pre-activations via a second matrix-vector product.
 *  - Epilogue:
 *      • If nops>1: apply numerically stable softmax (multiclass).
 *      • Else     : apply fast sigmoid (binary/regression-like).
 *
 * Data layout (row-major):
 *  - w:  nhid rows × nips cols. Row j holds incoming weights for hidden unit j.
 *  - x:  nops rows × nhid cols. Row i holds incoming weights for output unit i.
 *  - b:  [bh(0..nhid-1), bo(0..nops-1)] contiguous biases.
 *
 * BLAS path:
 *  - Uses cblas_sgemm with M=1 to compute row-vector × matrix^T efficiently.
 *    While sgemv would be the semantic match, sgemm often maps better to
 *    vendor-tuned kernels and amortizes overhead.
 *  - If FWC_OAT is enabled and calibrated, we pick GEMV vs GEMM empirically
 *    for the hidden matvec to minimize single-sample latency.               */
static void fprop(const NeuralNetwork_Type nn, const float * const in)
{
    const int nips = nn.nips;
    const int nhid = nn.nhid;
    const int nops = nn.nops;

    const float *w = nn.w;  /* input→hidden  (nhid × nips), row-major */
    const float *x = nn.x;  /* hidden→output (nops × nhid), row-major */
    const float *b = nn.b;  /* per-unit biases (hidden then output)   */
    const float *bh = b;               /* hidden biases: nhid */
    const float *bo = b + nhid;        /* output biases: nops */

    float       *h = nn.h;  /* hidden activations */
    float       *o = nn.o;  /* output activations */

    /* -------- Hidden: h = in · w^T --------
     * For each hidden unit j, compute ⟨in, w_j⟩ (row j of w).
     * Fallback loop is straightforward and amenable to auto-vectorization.   */
#if defined(FWC_USE_BLAS) && (FWC_USE_BLAS+0)==1
    /* Choose GEMV vs GEMM based on OAT (if available), else default GEMM. */
    int use_gemv = 0;
    #ifdef FWC_OAT
    if (nn.oat.enabled && nn.oat.use_gemv_single) use_gemv = 1;
    #endif
    if (use_gemv) {
        /* GEMV path: h = W · in */
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    nhid, nips, 1.0f, w, nips, in, 1, 0.0f, h, 1);
    } else {
        /* GEMM path: (1×nips) · (nips×nhid) → (1×nhid) */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    1, nhid, nips,
                    1.0f, in, nips,
                          w,  nips,
                    0.0f, h,  nhid);
    }
#else
    for (int j = 0; j < nhid; ++j) {
        const float *wj = w + (size_t)j * nips;
        float acc = 0.0f;
        for (int k = 0; k < nips; ++k) acc += in[k] * wj[k];
        h[j] = acc;
    }
#endif

    /* -------- Hidden epilogue: per-unit bias + activation --------
     * Pre-activation z_j = h[j] + b_h[j].                              */
    for (int j = 0; j < nhid; ++j) {
        const float z = h[j] + bh[j];
#if defined(FWC_CACHE_Z)
        nn.hz[j] = z;                      /* keep pre-activation if needed */
#endif
#if defined(FWC_RELU_HID)
        h[j] = (z > 0.0f) ? z : 0.0f;      /* ReLU: max(0,z) */
#else
        h[j] = fast_sigmoid(z);            /* Sigmoid approximation */
#endif
    }

    /* -------- Output: o = h · x^T --------
     * Each output i computes ⟨h, x_i⟩ where x_i is row i of the output weight
     * matrix. Equivalent BLAS call mirrors the hidden-layer computation.     */
#if defined(FWC_USE_BLAS) && (FWC_USE_BLAS+0)==1
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                1, nops, nhid,
                1.0f, h,  nhid,
                      x,  nhid,
                0.0f, o,  nops);
#else
    for (int i = 0; i < nops; ++i) {
        const float *xi = x + (size_t)i * nhid;
        float acc = 0.0f;
        for (int j = 0; j < nhid; ++j) acc += h[j] * xi[j];
        o[i] = acc;
    }
#endif

    /* -------- Output epilogue --------
     * Multiclass (nops>1): apply stable softmax.
     * Binary/regression-like (nops==1): apply fast sigmoid.                 */
    if (nops > 1) {  /* stable softmax */
        float m = -FLT_MAX;
        for (int i = 0; i < nops; ++i) { o[i] += bo[i]; if (o[i] > m) m = o[i]; }
        float s = 0.0f;
        for (int i = 0; i < nops; ++i) { o[i] = expf(o[i] - m); s += o[i]; }
        const float invs = s > 0.0f ? (1.0f / s) : 1.0f;
        for (int i = 0; i < nops; ++i) o[i] *= invs;
        return;
    }

    /* binary / regression-like path: sigmoid output */
    for (int i = 0; i < nops; ++i) {
        const float z = o[i] + bo[i];
        o[i] = fast_sigmoid(z);
    }
}


/* ============================================================================
 * Backpropagation and Initialization Routines (single- and two-hidden-layer)
 * ----------------------------------------------------------------------------
 * This section implements:
 *   • bprop : one-sample stochastic gradient step for a 1-hidden-layer MLP
 *             with either {softmax + cross-entropy} (multiclass) or
 *             {sigmoid + BCE/MSE-style} (binary) output heads.
 *   • wbrand / wbrand2 : principled random initialization (He/Xavier) for
 *             1- and 2-hidden-layer architectures, plus zero biases.
 *   • toterr : half-squared ℓ₂ loss accumulator (portable, vectorizable).
 *
 * Mathematical summary for bprop (single hidden layer):
 *   Let
 *      h = φ(W in + b_h)     with φ = ReLU or σ_fast
 *      o = head(X h + b_o)   head = softmax (nops>1) or σ_fast (nops=1)
 *   Targets tg are one-hot (multiclass) or scalar/binary (nops=1).
 *
 *   Output-layer delta:
 *      If softmax+CE:        δ_o = o − tg
 *      If sigmoid+BCE/MSE:   δ_o = (o − tg) ⊙ head'(·)
 *
 *   Hidden-layer delta:
 *      δ_h = (Xᵀ δ_o) ⊙ φ'(·)
 *
 *   Rank-1 SGD updates (outer products):
 *      X ← X − η (δ_o hᵀ),     W ← W − η (δ_h inᵀ)
 *    Biases are unit-wise:     b_o ← b_o − η δ_o,   b_h ← b_h − η δ_h
 *
 * Storage/layout conventions (row-major):
 *   W ∈ ℝ^{nhid×nips} ; row j holds incoming weights of hidden unit j.
 *   X ∈ ℝ^{nops×nhid} ; row i holds incoming weights of output unit i.
 *   b = [b_h(0..nhid−1) | b_o(0..nops−1)] as a single contiguous buffer.
 *
 * Performance notes:
 *   • GEMV/Ger calls (BLAS) are used for δ backpropagation and rank-1 updates.
 *   • 'restrict' annotations clarify no-aliasing for better vectorization.
 *   • VLA-free path for MSVC allocates temporaries on the heap.
 * ==========================================================================*/

static void bprop(const NeuralNetwork_Type nn,
                  const float *in,
                  const float *tg,
                  float rate)
{
    const int nips = nn.nips, nhid = nn.nhid, nops = nn.nops;

    float *restrict W = nn.w;
    float *restrict X = nn.x;
    float *restrict b = nn.b;          /* layout assumed: [bh(0..nhid-1) | bo(0..nops-1)] */
    float *restrict h = nn.h;
    float *restrict o = nn.o;

#if defined(FWC_CACHE_Z)
    const float *restrict hz = nn.hz;
    /* const float *restrict oz = nn.oz; */  /* If available: pre-activations for outputs */
#endif

    /* Portable allocation for delta buffers (avoid VLAs on MSVC)
     * δ_o ∈ ℝ^{nops}, δ_h ∈ ℝ^{nhid}. Each holds per-unit error signals. */
#if defined(_MSC_VER) || defined(FWC_NO_VLA)
    float *delta_o = (float*)malloc((size_t)nops * sizeof(float));
    float *delta_h = (float*)malloc((size_t)nhid * sizeof(float));
    if (!delta_o || !delta_h) { free(delta_o); free(delta_h); return; }
#else
    float delta_o[nops];
    float delta_h[nhid];
#endif

    /* 1) Output deltas
     * Multiclass: softmax + cross-entropy gives the classic δ_o = p − y.
     * Binary/regression-like: chain rule applies the sigmoid derivative. */
    if (nops > 1) {
        /* softmax + cross-entropy: grad = p - y */
        for (int j = 0; j < nops; ++j)
            delta_o[j] = o[j] - tg[j];
    } else {
        /* sigmoid + BCE (or MSE-style): grad = (σ - y) * σ'(z)  ; here σ stored in o
         * If oz (pre-activation) is cached, one may use fast_sigmoid_grad(oz[j]). */
        for (int j = 0; j < nops; ++j) {
            const float oj  = o[j];
            const float err = oj - tg[j];
#if defined(FWC_CACHE_Z)
            /* If oz cached and using fast_sigmoid at output, prefer:
               delta_o[j] = err * fast_sigmoid_grad(oz[j]); */
            delta_o[j] = err * oj * (1.0f - oj);
#else
            delta_o[j] = err * oj * (1.0f - oj);
#endif
        }
    }

    /* 2) Hidden deltas: δ_h = X^T δ_o
     * Backpropagate through the linear map h ↦ X h. BLAS GEMV does
     * the transposed multiply efficiently. */
    cblas_sgemv(CblasRowMajor, CblasTrans,
                nops, nhid, 1.0f,
                X, nhid, delta_o, 1,
                0.0f, delta_h, 1);

    /* Multiply by activation derivative at hidden:
     * ReLU: indicator(z>0) ;  Sigmoid-approx: fast_sigmoid_grad(z) or h⊙(1−h). */
#if defined(FWC_RELU_HID)
    for (int i = 0; i < nhid; ++i) {
    #if defined(FWC_CACHE_Z)
        delta_h[i] *= (hz[i] > 0.0f) ? 1.0f : 0.0f;    /* gate by pre-activation */
    #else
        delta_h[i] *= (h[i] > 0.0f) ? 1.0f : 0.0f;     /* fallback gate by post-act */
    #endif
    }
#else
    for (int i = 0; i < nhid; ++i) {
    #if defined(FWC_CACHE_Z)
        delta_h[i] *= fast_sigmoid_grad(hz[i]);   /* exact grad for fast_sigmoid */
    #else
        const float hi = h[i];                    /* fallback: logistic' using h */
        delta_h[i] *= hi * (1.0f - hi);
    #endif
    }
#endif

    /* 3) Parameter updates
     * Rank-1 outer-product SGD steps:
     *   X ← X − η (δ_o hᵀ),  W ← W − η (δ_h inᵀ).
     * nrate is −η so we can use BLAS AXPY/GER in additive form. */
    const float nrate = -rate;

    /* X ← X + nrate * (delta_o · h^T)  (nops×nhid) */
    cblas_sger(CblasRowMajor, nops, nhid, nrate,
               delta_o, 1, h, 1, X, nhid);

    /* W ← W + nrate * (delta_h · in^T) (nhid×nips) */
    cblas_sger(CblasRowMajor, nhid, nips, nrate,
               delta_h, 1, in, 1, W, nips);

    /* 4) Per-unit bias updates (instead of scalar layer biases)
     * Biases shift the affine pre-activations; gradients are simply the deltas. */
    float *bh = b;            /* hidden biases: nhid elements */
    float *bo = b + nhid;     /* output biases: nops elements */

    for (int i = 0; i < nops; ++i)  bo[i] += nrate * delta_o[i];
    for (int j = 0; j < nhid; ++j)  bh[j] += nrate * delta_h[j];

#if defined(_MSC_VER) || defined(FWC_NO_VLA)
    free(delta_h);
    free(delta_o);
#endif
}



/* ============================================================================
 * wbrand: Random initialization for 1-hidden-layer MLP
 * ----------------------------------------------------------------------------
 * Hidden layer:
 *   • ReLU: He/Kaiming uniform ~ U[−√(6/f_in), √(6/f_in)]
 *   • Sigmoid/Tanh: Xavier/Glorot uniform ~ U[−√(6/(f_in+f_out)), √(6/(f_in+f_out))]
 * Output layer:
 *   • Xavier uniform: balances variance into softmax/sigmoid heads.
 * Biases:
 *   • Initialized to zero (common, keeps symmetry broken by weights).
 * ==========================================================================*/
static inline void wbrand(const NeuralNetwork_Type nn) {
    const int nips = nn.nips;
    const int nhid = nn.nhid;
    const int nops = nn.nops;

    float *w  = nn.w;   /* input→hidden (nhid×nips), row-major */
    float *x  = nn.x;   /* hidden→output (nops×nhid), row-major */
    float *bh = nn.b;               /* hidden biases start here */
    float *bo = nn.b + nhid;        /* output biases follow */

    /* -------- Hidden layer init --------
       - If FWC_RELU_HID: He (Kaiming) uniform for ReLU
       - Else: Xavier (Glorot) uniform for sigmoid/tanh
       The constants match common practice; they preserve activation variance. */
#if defined(FWC_RELU_HID)
    const float limit_h = sqrtf(6.0f / (float)nips);                 /* He uniform */
#else
    const float limit_h = sqrtf(6.0f / ((float)nips + (float)nhid)); /* Xavier uniform */
#endif

    for (int j = 0; j < nhid; ++j) {
        for (int i = 0; i < nips; ++i) {
            w[(size_t)j * nips + i] = (2.0f * frand() - 1.0f) * limit_h; /* U[-lim,lim] */
        }
    }

    /* -------- Output layer init --------
       Use Xavier uniform for softmax/sigmoid outputs to keep logits well-scaled. */
    const float limit_o = sqrtf(6.0f / ((float)nhid + (float)nops));
    for (int k = 0; k < nops; ++k) {
        for (int j = 0; j < nhid; ++j) {
            x[(size_t)k * nhid + j] = (2.0f * frand() - 1.0f) * limit_o;  /* U[-lim,lim] */
        }
    }

    /* -------- Biases -------- */
    for (int j = 0; j < nhid; ++j) bh[j] = 0.0f;  /* hidden biases */
    for (int i = 0; i < nops; ++i) bo[i] = 0.0f;  /* output biases */
}


/* ─────────────────────────────────────────────────────────── */
/* 2-hidden-layer builder                                     */
/* in → h1(nhid) → h2(nhid2) → out(nops)                      */
/* Notes:
 *   • Applies He/Xavier rules layer-wise using respective fan-in/out.
 *   • Bias storage here is illustrated as three scalars; in practice,
 *     multi-unit biases occupy contiguous blocks per layer.                */
/* ─────────────────────────────────────────────────────────── */
static inline void wbrand2(const NeuralNetwork_Type nn)
{
    const int nips  = nn.nips;
    const int nhid  = nn.nhid;
    const int nhid2 = nn.nhid2;
    const int nops  = nn.nops;

    float *w = nn.w;   /* input→h1 (nhid×nips) */
    float *u = nn.u;   /* h1→h2   (nhid2×nhid) */
    float *x = nn.x;   /* h2→out  (nops×nhid2) */
    float *b = nn.b;   /* b[0]=h1, b[1]=h2, b[2]=out */

    /* He/Xavier limits per layer.
     * ReLU case uses He on the first layer; second uses Xavier by default here. */
#if defined(FWC_RELU_HID)
    const float lim_h1 = sqrtf(6.0f / (float)nips);
#else
    const float lim_h1 = sqrtf(6.0f / ((float)nips + (float)nhid));
#endif
    const float lim_h2 = sqrtf(6.0f / ((float)nhid + (float)nhid2));
    const float lim_o  = sqrtf(6.0f / ((float)nhid2 + (float)nops));

    /* W: input→h1 */
    for (int j = 0; j < nhid; ++j)
        for (int i = 0; i < nips; ++i)
            w[(size_t)j*nips + i] = (2.0f*frand()-1.0f)*lim_h1;

    /* U: h1→h2 */
    for (int j = 0; j < nhid2; ++j)
        for (int i = 0; i < nhid; ++i)
            u[(size_t)j*nhid + i] = (2.0f*frand()-1.0f)*lim_h2;

    /* X: h2→out */
    for (int k = 0; k < nops; ++k)
        for (int j = 0; j < nhid2; ++j)
            x[(size_t)k*nhid2 + j] = (2.0f*frand()-1.0f)*lim_o;

    /* Bias placeholders; in a full implementation, these would zero
     * out the contiguous bias vectors per layer. */
    b[0] = 0.0f;  /* h1 bias */
    b[1] = 0.0f;  /* h2 bias */
    b[2] = 0.0f;  /* out bias */
}



/* ============================================================================
 * toterr: Cross-entropy loss (stable) for classification
 * ----------------------------------------------------------------------------
 * Drop-in replacement that *keeps the same prototype* but switches from
 * half-squared error to cross-entropy, which typically yields higher accuracy.
 *
 * Interprets 'o' as either:
 *   • probabilities (post-softmax)  → uses -∑ tg_i * log(max(ε, o_i))
 *   • logits (pre-softmax)          → uses log-sum-exp: L = lse - ∑ tg_i * o_i
 *
 * Heuristic to choose mode:
 *   - If all o_i ∈ [−1e−6, 1+1e−6] and |∑ o_i − 1| ≤ 1e−3 → treat as probabilities
 *   - Otherwise treat as logits (recommended for stability/perf)
 *
 * Returns: scalar CE loss (sum over classes for this sample)
 * ==========================================================================*/
static inline float toterr(const float *restrict tg,
                           const float *restrict o,
                           const int size)
{
    /* ---------- quick probe: are 'o' probabilities? ---------- */
    float s = 0.0f, omin = o[0], omax = o[0];
    #if defined(_OPENMP)
    #pragma omp simd reduction(+:s) reduction(min:omin) reduction(max:omax)
    #endif
    for (int i = 0; i < size; ++i) {
        const float oi = o[i];
        s    += oi;
        if (oi < omin) omin = oi;
        if (oi > omax) omax = oi;
    }
    const float sum_close_to_one = fabsf(s - 1.0f) <= 1e-3f;
    const int   looks_prob       = (omin >= -1e-6f) && (omax <= 1.0f + 1e-6f) && sum_close_to_one;

    /* ---------- cross-entropy in prob-space (safe log) ---------- */
    if (looks_prob) {
        const float eps = 1e-7f;  /* clamp to avoid log(0) */
        float loss = 0.0f;
        #if defined(_OPENMP)
        #pragma omp simd reduction(+:loss)
        #endif
        for (int i = 0; i < size; ++i) {
            const float pi = (o[i] < eps) ? eps : o[i];
            loss -= tg[i] * logf(pi);
        }
        return loss;
    }

    /* ---------- cross-entropy from logits via log-sum-exp ---------- */
    /* 1) max logit for numerical stability */
    float m = o[0];
    for (int i = 1; i < size; ++i) if (o[i] > m) m = o[i];

    /* 2) sum exp(logits - m) */
    float sum_exp = 0.0f;
    #if defined(_OPENMP)
    #pragma omp simd reduction(+:sum_exp)
    #endif
    for (int i = 0; i < size; ++i) {
        sum_exp += expf(o[i] - m);
    }
    const float lse = m + logf(sum_exp);  /* log-sum-exp */

    /* 3) dot(tg, logits) */
    float dot = 0.0f;
    #if defined(_OPENMP)
    #pragma omp simd reduction(+:dot)
    #endif
    for (int i = 0; i < size; ++i) {
        dot += tg[i] * o[i];
    }

    /* CE = lse - ∑ tg_i * o_i */
    return lse - dot;
}

/* ============================================================================
 * Public Prediction & Builder Routines
 * ----------------------------------------------------------------------------
 * This section exposes:
 *   • NNpredict        : single-sample forward evaluation, returns nn.o
 *   • NNpredict_batch  : batched forward for B samples (BLAS-accelerated)
 *   • NNbuild / NNbuild2: memory allocation + randomized initialization
 *   • NNbuild_auto     : heuristic choice of depth based on dataset size
 *
 * Conventions (row-major):
 *   - in           ∈ R^{nips}
 *   - batch_in     ∈ R^{B×nips}
 *   - nn.w         ∈ R^{nhid×nips}   (input→hidden)
 *   - nn.x         ∈ R^{nops×nhid}   (hidden→output)
 *   - nn.b         : biases; for 1-hidden-layer build: [bh(0..nhid−1) | bo(0..nops−1)]
 *   - nn.h, nn.o   : intermediate/output buffers for single-sample path
 *
 * Numerical notes:
 *   - Softmax is applied with a max-subtraction trick for stability.
 *   - Sigmoid is the fast approximation defined elsewhere (fast_sigmoid).
 * Threading notes:
 *   - NNpredict_batch uses a static scratch H; not thread-safe across calls.
 * Memory notes:
 *   - Builders use calloc and goto-based unwinding on failure; returned
 *     structures are zero-initialized on failure via compound literal.
 * ==========================================================================*/

/* Exposed Functions in Header File */

float * NNpredict(const NeuralNetwork_Type nn,
                  const float * in)
{
    /* Hoist the output pointer into a register once
     * (purely a clarity/micro-optimization gesture). */
    float * const out = nn.o;

    /* Forward-propagate (fprop is already static inline, so 
       the compiler will inline it here and eliminate the call)
     * Semantics: fills nn.h and nn.o from 'in' and current parameters. */
    fprop(nn, in);

    /* Return the pre-loaded output buffer
     * Caller owns no memory here; nn.o is part of 'nn'. */
    return out;
}


/* ---------- NEW public function ----------------------------------- */
void NNpredict_batch(const NeuralNetwork_Type nn,
                     const float *batch_in,
                     int B,
                     float *batch_out)
{
    const int nips = nn.nips;
    const int nhid = nn.nhid;
    const int nops = nn.nops;

    // 1) scratch H[B×nhid] (NOTE: static => not thread-safe)
    /* We keep a persistent scratch to avoid repeated malloc/free on hot path.
     * WARNING: Not re-entrant and not safe for concurrent calls with different B.
     * Capacity grows monotonically; no shrink is attempted. */
    static float *H = NULL;
    static size_t Hcap = 0;
    const size_t needH = (size_t)B * nhid;
    if (Hcap < needH) {
        free(H);
        H = (float*)malloc(needH * sizeof *H);
        Hcap = needH;
    }

    // 2) input→hidden
    /* Compute H = batch_in · W^T
     * Shapes: (B×nips)·(nhid×nips)^T = (B×nhid).
     * We use sgemm with RowMajor, NoTrans×Trans as in the single-sample path. */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, nhid, nips,
                1.0f,
                batch_in, nips,
                nn.w,     nips,
                0.0f,
                H,        nhid);

// 3) hidden bias + activation   (FIX: respect FWC_RELU_HID here, too)
/* Apply the hidden-layer epilogue elementwise:
 *   z = H + b_h
 *   h = ReLU(z) or fast_sigmoid(z)
 * NOTE: This implementation adds nn.b[0] uniformly (shared bias),
 *       which differs from the per-unit hidden bias layout used elsewhere
 *       (bh[0..nhid−1]). Keeping as-is per user instruction; adjust if
 *       per-unit biases are intended. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
for (size_t i = 0; i < needH; ++i) {
    const float z = H[i] + nn.b[0];
#if defined(FWC_RELU_HID)
    H[i] = (z > 0.0f) ? z : 0.0f;   /* ReLU (matches NNtrain_batch) */
#else
    H[i] = fast_sigmoid(z);         /* Sigmoid (old path) */
#endif
}


    // 4) hidden→output
    /* Compute batch_out = H · X^T
     * Shapes: (B×nhid)·(nops×nhid)^T = (B×nops). */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, nops, nhid,
                1.0f,
                H,    nhid,
                nn.x, nhid,
                0.0f,
                batch_out, nops);

    // 5) output epilogue: softmax (nops>1) or sigmoid
    /* For multiclass: row-wise stable softmax.
     * For binary/regression-like: elementwise fast sigmoid.
     * NOTE: Uses nn.b[1] uniformly as an output bias; if per-class biases
     *       bo[0..nops−1] are desired, the epilogue would differ. */
    const size_t needO = (size_t)B * nops;
    if (nops > 1) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int r = 0; r < B; ++r) {
            float *row = batch_out + (size_t)r * nops;
            float m = -FLT_MAX;  /* requires <float.h> somewhere in TU */
            for (int i = 0; i < nops; ++i) { row[i] += nn.b[1]; if (row[i] > m) m = row[i]; }
            float s = 0.f;
            for (int i = 0; i < nops; ++i) { row[i] = expf(row[i] - m); s += row[i]; }
            const float invs = s > 0.f ? (1.0f / s) : 1.0f;
            for (int i = 0; i < nops; ++i) row[i] *= invs;
        }
    } else {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < needO; ++i) {
            const float z = batch_out[i] + nn.b[1];
            batch_out[i] = fast_sigmoid(z);
        }
    }
}



NeuralNetwork_Type NNbuild(int nips, int nhid, int nops) {
    /* Builder for 1-hidden-layer MLP. Allocates a single contiguous block for
     * weights (W||X), contiguous biases (bh||bo), and separate buffers for h,o.
     * On failure, frees what was allocated and returns a zero-initialized nn. */
    NeuralNetwork_Type nn;
    nn.nips = nips; nn.nhid = nhid; nn.nops = nops;

    nn.nb = nhid + nops;                 /* CHANGED: per-unit biases (bh then bo) */
    const int wih = nips * nhid;
    const int who = nhid * nops;
    nn.nw = wih + who;

    nn.w = (float*)calloc((size_t)nn.nw, sizeof *nn.w);
    if (!nn.w) goto fail_w;
    nn.x = nn.w + wih;                   /* alias into the same contiguous block */

    nn.b = (float*)calloc((size_t)nn.nb, sizeof *nn.b);  /* CHANGED size: bh||bo */
    if (!nn.b) goto fail_b;

    nn.h = (float*)calloc((size_t)nhid, sizeof *nn.h);
    if (!nn.h) goto fail_h;
    nn.o = (float*)calloc((size_t)nops, sizeof *nn.o);
    if (!nn.o) goto fail_o;

#if defined(FWC_CACHE_Z)
    nn.hz = (float*)calloc((size_t)nhid, sizeof *nn.hz);
    if (!nn.hz) goto fail_hz;
#endif

    /* Randomize parameters with appropriate fan-in/out scaling. */
    wbrand(nn);
    return nn;

#if defined(FWC_CACHE_Z)
fail_hz: free(nn.o);
#endif
fail_o:  free(nn.h);
fail_h:  free(nn.b);
fail_b:  free(nn.w);
fail_w:  return (NeuralNetwork_Type){0};
}


NeuralNetwork_Type NNbuild2(int nips, int nhid, int nhid2, int nops) {
    /* Builder for 2-hidden-layer MLP. Mirrors NNbuild but also allocates:
     *   - u ∈ R^{nhid2×nhid} (h1→h2)
     *   - h2 buffer
     * Biases total: nhid + nhid2 + nops (per-unit for each layer). */
    NeuralNetwork_Type nn;
    nn.nips = nips; nn.nhid = nhid; nn.nhid2 = nhid2 > 0 ? nhid2 : nhid; nn.nops = nops;

    nn.nb = nhid + nn.nhid2 + nops;      /* CHANGED: per-unit biases across all layers */

    const int wih  = nips * nhid;
    const int h2h1 = nhid * nn.nhid2;
    const int who  = nn.nhid2 * nops;
    nn.nw = wih + h2h1 + who;

    nn.w = (float*)calloc((size_t)nn.nw, sizeof *nn.w);
    if (!nn.w) goto fail_w;
    nn.u = nn.w + wih;                    /* second weight block starts here */
    nn.x = nn.u + h2h1;                   /* third block after U */

    nn.b = (float*)calloc((size_t)nn.nb, sizeof *nn.b);  /* CHANGED size */
    if (!nn.b) goto fail_b;

    nn.h  = (float*)calloc((size_t)nhid, sizeof *nn.h);
    if (!nn.h) goto fail_h;
    nn.h2 = (float*)calloc((size_t)nn.nhid2, sizeof *nn.h2);
    if (!nn.h2) goto fail_h2;
    nn.o  = (float*)calloc((size_t)nops, sizeof *nn.o);
    if (!nn.o) goto fail_o;

#if defined(FWC_CACHE_Z)
    nn.hz = (float*)calloc((size_t)nhid, sizeof *nn.hz);
    if (!nn.hz) goto fail_hz;
#endif

    /* Randomize parameters for all layers; zero biases. */
    wbrand2(nn);
    return nn;

#if defined(FWC_CACHE_Z)
fail_hz: free(nn.o);
#endif
fail_o:  free(nn.h2);
fail_h2: free(nn.h);
fail_h:  free(nn.b);
fail_b:  free(nn.w);
fail_w:  return (NeuralNetwork_Type){0};
}


/* ─────────────────────────────────────────────────────────── */
/* Auto-depth builder: decides inside the function             */
/* Pass dataset size N; tweak thresholds as you like           */
/* Heuristic rationale:
 *   - Smaller N → shallower model to reduce variance and overfitting.
 *   - Medium N → two layers but narrower h2 for regularization/economy.
 *   - Large N → two layers with equal width to capture complexity.       */
/* ─────────────────────────────────────────────────────────── */
#ifndef FWC_AUTO_N_SMALL
#define FWC_AUTO_N_SMALL  10000    /* <10k → 1 layer */
#endif
#ifndef FWC_AUTO_N_MED
#define FWC_AUTO_N_MED    50000    /* 10k–50k → 2 layers (smaller h2) */
#endif

NeuralNetwork_Type NNbuild_auto(int nips, int nhid, int nops, long long N)
{
    if (N < FWC_AUTO_N_SMALL) {
        /* small dataset → 1 hidden layer
         * Favor bias-variance tradeoff on limited data. */
        return NNbuild(nips, nhid, nops);
    }

    /* medium/large dataset → 2 hidden layers
     * Choose h2 based on N: narrower for medium data, same width for large. */
    int nhid2;
    if (N < FWC_AUTO_N_MED) nhid2 = nhid > 1 ? (nhid/2) : nhid;  /* conservative */
    else                    nhid2 = nhid;                         /* same width */

    return NNbuild2(nips, nhid, nhid2, nops);
}


/* nn.c (already contains static fprop & static bprop)
 * ============================================================================
 * High-level training, serialization, and lifecycle utilities
 * ----------------------------------------------------------------------------
 * This section provides:
 *   • NNtrain : one SGD step = forward → backward → return scalar loss
 *   • NNsave  : human-readable (text) checkpoint — dims, then biases, then weights
 *   • NNload  : inverse of NNsave — rebuilds the network and loads parameters
 *   • NNprint : pretty-print a vector and report the argmax index
 *   • NNfree  : free buffers (value-type 'nn' — safe with NULL fields)
 *   • NNdestroy : destructor for a pointer-held 'nn' (zeros the struct)
 *
 * Design notes (professor’s eye):
 *   - NNtrain uses toterr(tg, o) = ½‖tg − o‖² as a simple scalar loss reporter.
 *     This is independent of the internal training head (softmax/BCE).
 *   - NNsave/NNload use TEXT format for portability and debuggability.
 *     This eases diffs and manual inspection at small scale; for production,
 *     a binary format would be faster and more compact.
 *   - setvbuf(..., _IOFBF, BUFSIZ) makes I/O fully buffered to reduce syscalls.
 *   - NNfree vs. NNdestroy: one takes by value and frees fields; the other
 *     takes by pointer, frees, then memset(...) to zero to prevent reuse.
 * ==========================================================================*/

float NNtrain(const NeuralNetwork_Type nn,
              const float *in,
              const float *tg,
              float rate)
{
    /* 1. forward pass
     *    Fills nn.h and nn.o based on current parameters and input 'in'. */
    fprop(nn, in);

    /* 2. backward pass + weight update
     *    Computes deltas, applies rank-1 SGD updates to W/X and biases. */
    bprop(nn, in, tg, rate);

    /* 3. return sample loss
     *    Report ½Σ(t−o)² using the current outputs in nn.o.
     *    This is a diagnostic scalar; training head may differ internally. */
    return toterr(tg, nn.o, nn.nops);   /* 0.5 · Σ(t−o)² */
}


/**
 * And it came to pass at the fateful hour:
 * “Write down the network unto the scroll,”
 * that it might be restored when the dawn of inference breaks.
 */
void NNsave(const NeuralNetwork_Type nn, const char * path)
{
    /* Open the scroll for writing
     * Mode "w" → text; portable across platforms but not lossless like binary. */
    FILE * const file = fopen(path, "w");
    if (!file) {
        /* If the scroll is sealed, abort the saving ritual
         * perror writes a descriptive message to stderr. */
        perror("NNsave: fopen failed");
        return;
    }

    /* Buffer the writes, that many lines may flow swiftly
     * Full buffering reduces I/O overhead for many small fprintf calls. */
    setvbuf(file, NULL, _IOFBF, BUFSIZ);

    /* Hoist fields into locals for swifter access
     * (helps readability and may aid register allocation). */
    const int nips = nn.nips;
    const int nhid = nn.nhid;
    const int nops = nn.nops;
    const int nb   = nn.nb;
    const int nw   = nn.nw;
    const float *b = nn.b;
    const float *w = nn.w;

    /* Save the Header
     * A single line with dimensions: inputs, hidden, outputs. */
    fprintf(file, "%d %d %d\n", nips, nhid, nops);

    /* Save the Biases — anoint each neuron’s offset
     * Layout matches builders: first nhid hidden biases, then nops output biases. */
    for (const float *bp = b, *bend = b + nb; bp < bend; ++bp) {
        fprintf(file, "%f\n", (double)*bp);  /* printf's %f expects double */
    }

    /* Save the Weights — inscribe each connection’s strength
     * Order: W (nhid×nips) followed by X (nops×nhid), contiguous in memory. */
    for (const float *wp = w, *wend = w + nw; wp < wend; ++wp) {
        fprintf(file, "%f\n", (double)*wp);
    }

    /* Close the scroll
     * Always close to flush buffers; errors here would be silent in this path. */
    fclose(file);
}


/**
 * And it came to pass, the seeker summoned NNload,
 * that the network might rise again from the scroll of bytes.
 */
NeuralNetwork_Type NNload(const char * path)
{
    /* Open the sacred scroll for reading
     * Text mode "r" matches NNsave's output. */
    FILE * const file = fopen(path, "r");
    if (!file) {
        perror("NNload: fopen failed");
        return (NeuralNetwork_Type){0};
    }
    /* Prepare the vessel for swift reads
     * Full buffering is also beneficial on reads with fscanf in a loop. */
    setvbuf(file, NULL, _IOFBF, BUFSIZ);

    /* Read the divine dimensions: inputs, hidden, outputs
     * Validate the header to guard against corrupted or mismatched files. */
    int nips = 0, nhid = 0, nops = 0;
    if (fscanf(file, "%d %d %d\n", &nips, &nhid, &nops) != 3) {
        perror("NNload: invalid header");
        fclose(file);
        return (NeuralNetwork_Type){0};
    }

    /* Build the tabernacle of neurons
     * Allocates weights, biases, activations, then random-initializes. */
    NeuralNetwork_Type nn = NNbuild(nips, nhid, nops);
    /* If the tabernacle failed to rise, abort
     * The zeroed sentinel indicates allocation failure in the builder. */
    if (nn.nw == 0 && nn.b == NULL) {
        fclose(file);
        return nn;
    }

    /* Hoist counts and pointers for biases and weights
     * We will overwrite randomized parameters with checkpoint values. */
    const int nb     = nn.nb;
    const int nw     = nn.nw;
    float    *b      = nn.b;
    float    *w      = nn.w;

    /* Load the Biases — each offset anointed anew
     * fscanf with "%f" writes directly into float storage. */
    for (float *bp = b, *bend = b + nb; bp < bend; ++bp) {
        if (fscanf(file, "%f\n", bp) != 1) {
            perror("NNload: reading bias failed");
            NNfree(nn);
            fclose(file);
            return (NeuralNetwork_Type){0};
        }
    }

    /* Load the Weights — each connection’s strength inscribed
     * Restores both W and X in the same contiguous order as saved. */
    for (float *wp = w, *wend = w + nw; wp < wend; ++wp) {
        if (fscanf(file, "%f\n", wp) != 1) {
            perror("NNload: reading weight failed");
            NNfree(nn);
            fclose(file);
            return (NeuralNetwork_Type){0};
        }
    }

    /* Close the scroll and return the living network
     * At this point, nn is ready for inference/training continuation. */
    fclose(file);
    return nn;
}


/**
 * And it came to pass, the prophet beheld the array of outputs:
 * he spoke, “Let us declare the greatest of these,”
 * and thus this function was consecrated.
 */
void NNprint(const float * arr, const int size)
{
    /* “Let there be a vault for the highest measure”
     * Initialize max with the first element and track argmax. */
    float max = arr[0];
    int idx = 0;

    /* “Traverse the fields from the first to the last,
     * that each value may be inscribed upon the scroll.”
     * Prints values space-separated, tracking the largest as we go. */
    for (int i = 0; i < size; ++i) {
        float val = arr[i];
        printf("%f ", val);
        if (val > max) {
            max = val;
            idx = i;
        }
    }

    /* “And when the journey ends, let there be a new line.” */
    putchar('\n');
    /* “And let the index of the mightiest be proclaimed.”
     * Reports the argmax index; useful for classification demos. */
    printf("The number is: %d\n", idx);
}


/**
 * And it came to pass at the end of the epoch:
 * “Disperse ye the vessels of memory,”
 * freeing outputs, hidden, biases, and weights,
 * that no clutter remain.
 */
/* Free buffers in creation order (safe if any are NULL)
 * Value-semantics: 'nn' is copied by value; we only free its owned pointers.
 * This function is idempotent if called on already-freed fields set to NULL. */
void NNfree(const NeuralNetwork_Type nn)
{
    free(nn.o);
    free(nn.h);
    free(nn.b);
#if defined(FWC_CACHE_Z)
    free(nn.hz);
    /* free(nn.oz);  // if you add it later */
#endif
    free(nn.w);
}


/**
 * @brief Releases all resources associated with a neural network instance
 *        and resets its state to prevent dangling pointers and double-free errors.
 *
 * @param nn  Pointer to the NeuralNetwork_Type instance to be destroyed.
 *            If NULL, the function returns immediately.
 *
 * Professor’s note:
 *   This is the safer companion to NNfree when the struct is heap-allocated
 *   or shared by multiple owners — zeroing prevents accidental reuse.
 */
void NNdestroy(NeuralNetwork_Type *nn)
{
    if (!nn) return;
    free(nn->w);
    free(nn->b);
    free(nn->h);
    free(nn->o);
#if defined(FWC_CACHE_Z)
    free(nn->hz);
    /* free(nn->oz); */
#endif
    memset(nn, 0, sizeof *nn);  /* clear all fields to benign defaults */
}


#ifndef FWC_DROPOUT
#define FWC_DROPOUT 0          /* set to 1 to enable hidden-layer dropout
                                * (compile-time switch; affects training path) */
#endif
#ifndef FWC_DROPOUT_P
#define FWC_DROPOUT_P 0.20f    /* drop probability (if enabled)
                                * Typical range 0.1–0.5; tune per dataset. */
#endif
/* ============================================================================
 * Mini-batch Training with Optional Inverted Dropout (BLAS-accelerated)
 * ----------------------------------------------------------------------------
 * This section provides:
 *   • A tiny, fast RNG (xorshift32) for stochastic dropout masks
 *   • NNtrain_batch: forward → loss deltas → backprop → SGD updates
 *
 * Mathematical sketch (per batch of size B):
 *   H  = X_in · W^T                         (B×nhid)
 *   h  = φ(H + b_h)                         (elementwise; ReLU or sigmoid)
 *   O  = h · X_out^T                        (B×nops)
 *   p  = softmax(O + b_o)  (nops>1)      or σ(O + b_o) (nops==1)
 *   DO = p − Y            (CE/softmax)   or σ − Y       (BCE/sigmoid)
 *   DH = DO · X_out                           then DH ⊙ φ'(·)
 *   X_out ← X_out − η/B · DO^T · h
 *   W     ← W     − η/B · DH^T · X_in
 *   b_o   ← b_o   − η/B · sum(DO)           (shared scalar)
 *   b_h   ← b_h   − η/B · sum(DH)           (shared scalar)
 *
 * Notes:
 *   - Storage is row-major; BLAS calls use RowMajor with appropriate transposes.
 *   - "Inverted dropout" divides by keep = 1 − p_drop at train-time so that
 *     E[h_drop] = h (test-time requires no rescaling).
 *   - Static scratch buffers avoid hot-path malloc/free; not thread-safe.
 *   - Biases here are used as layer-wise scalars (nn->b[0], nn->b[1]).
 * ==========================================================================*/


/* tiny RNG for stochastic mask
 * xorshift32: extremely fast, small-state PRNG. Period 2^32−1 (nonzero seed).
 * Not cryptographically secure; sufficient for dropout masking. */
static inline uint32_t fwc_xorshift32(uint32_t *s){
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s;
}

/* Map PRNG state to [0,1):
 * We keep the lower 24 bits (like IEEE mantissa width) and normalize by 2^24. */
static inline float fwc_rand01(uint32_t *s){
    return (fwc_xorshift32(s) & 0x00FFFFFF) / 16777216.0f; /* [0,1) */
}

void NNtrain_batch(NeuralNetwork_Type *nn,
                   int B,
                   const float *X,    /* B×nips inputs, row-major */
                   const float *Y,    /* B×nops targets           */
                   float lr)
{
    const int nips = nn->nips;
    const int nhid = nn->nhid;
    const int nops = nn->nops;

    /* ----- per-unit biases (match builders & single-sample path) ----- */
    float *bh = nn->b;           /* hidden biases: nhid            */
    float *bo = nn->b + nhid;    /* output biases: nops            */

    /* ---- scratch (static to avoid hot-path malloc/free) ----
     * H : hidden pre/post-activations (B×nhid)
     * O : output logits/activations   (B×nops)
     * DO: output-layer deltas         (B×nops)
     * DH: hidden-layer deltas         (B×nhid)
     * NOTE: static → persists across calls, not re-entrant/thread-safe. */
    static float *H = NULL, *O = NULL, *DO = NULL, *DH = NULL;
    static size_t capH = 0, capO = 0, capDO = 0, capDH = 0;

#if FWC_DROPOUT
    static uint8_t *M = NULL;   /* dropout mask for hidden activations (B×nhid) */
    static size_t capM = 0;
    const float p_drop = FWC_DROPOUT_P;
    const float keep   = 1.0f - p_drop;          /* inverted dropout scale */
    uint32_t seed = 0x9e3779b9u;                 /* per-call seed; make global if you want reproducibility */
#endif

    const size_t needH  = (size_t)B * (size_t)nhid;
    const size_t needO  = (size_t)B * (size_t)nops;

    /* Grow-on-demand scratch; no shrink to avoid churn. */
    if (capH  < needH) { free(H);  H  = (float*)malloc(needH * sizeof *H);  capH  = needH; }
    if (capO  < needO) { free(O);  O  = (float*)malloc(needO * sizeof *O);  capO  = needO; }
    if (capDO < needO) { free(DO); DO = (float*)malloc(needO * sizeof *DO); capDO = needO; }
    if (capDH < needH) { free(DH); DH = (float*)malloc(needH * sizeof *DH); capDH = needH; }
#if FWC_DROPOUT
    if (capM  < needH) { free(M);  M  = (uint8_t*)malloc(needH * sizeof *M); capM  = needH; }
#endif
    if (!H || !O || !DO || !DH
#if FWC_DROPOUT
        || !M
#endif
    ) return;

    /* 1) H = X · W^T  (B×nhid)
     * Shapes: (B×nips)·(nhid×nips)^T. Uses GEMM to leverage tuned kernels. */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, nhid, nips,
                1.0f,
                X,      nips,
                nn->w,  nips,
                0.0f,
                H,      nhid);

    /* 2) Hidden epilogue: **per-unit** bias + activation (+ optional inverted dropout)
     * This used to add a single scalar bias and capped accuracy. Fix:
     *   z[r,j] = H[r,j] + bh[j]   (per-hidden-unit bias)
     *   h[r,j] = φ(z[r,j])        (ReLU or fast sigmoid)
     *   if dropout: h[r,j] = m ? h[r,j]/keep : 0
     */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t idx = 0; idx < needH; ++idx) {
        const int j = (int)(idx % (size_t)nhid);     /* column within hidden layer */
        const float z = H[idx] + bh[j];
    #if defined(FWC_RELU_HID)
        float a = (z > 0.0f) ? z : 0.0f;             /* ReLU */
    #else
        float a = fast_sigmoid(z);                   /* Sigmoid */
    #endif
    #if FWC_DROPOUT
        float u = fwc_rand01(&seed);
        uint8_t m = (u >= p_drop);                   /* keep if u>=p_drop */
        M[idx] = m;
        H[idx] = m ? (a / keep) : 0.0f;              /* inverted dropout keeps expectation */
    #else
        H[idx] = a;
    #endif
    }

    /* 3) O = H · X^T  (B×nops) — produce logits prior to output epilogue */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, nops, nhid,
                1.0f,
                H,      nhid,
                nn->x,  nhid,
                0.0f,
                O,      nops);

    /* 4) Output epilogue + DO
     * Multiclass: **per-class** bias (bo[i]) + stable softmax, DO = p − Y.
     * Binary   :  per-class bias bo[0] + sigmoid, DO = σ − Y. */
    if (nops > 1) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int r = 0; r < B; ++r) {
            float       *o_row = O  + (size_t)r * nops;  /* logits row */
            const float *y_row = Y  + (size_t)r * nops;  /* target row */
            float       *d_row = DO + (size_t)r * nops;  /* delta row  */

            /* add **per-class** biases before softmax */
            for (int i = 0; i < nops; ++i) o_row[i] += bo[i];

            /* fused softmax + cross-entropy gradient (write_probs=1) */
            (void)softmax_ce_fused_row(o_row, y_row, d_row, nops, /*bias=*/0.0f, /*write_probs=*/1);
        }
    } else {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < needO; ++i) {
            const float z  = O[i] + bo[0];          /* per-class bias: only one class */
            const float oi = fast_sigmoid(z);
            DO[i] = oi - Y[i];
            O[i]  = oi;
        }
    }

    /* 5) DH = DO · X  (B×nhid) — backprop through output linear map */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                B, nhid, nops,
                1.0f,
                DO,     nops,
                nn->x,  nhid,
                0.0f,
                DH,     nhid);

    /* 6) Multiply by activation derivative at hidden (+ dropout gate) */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t idx = 0; idx < needH; ++idx) {
    #if FWC_DROPOUT
        if (!M[idx]) { DH[idx] = 0.0f; continue; }  /* dropped units carry no gradient */
    #endif
    #if defined(FWC_RELU_HID)
        DH[idx] *= (H[idx] > 0.0f) ? 1.0f : 0.0f;   /* ReLU' using post-activation */
    #else
        const float hi = H[idx];                    /* H holds σ(z) (possibly scaled) */
        DH[idx] *= hi * (1.0f - hi);                /* sigmoid' */
    #endif
    }

    /* 7) SGD updates (averaged over batch)
     *   X ← X + nrate · (DO^T · H)
     *   W ← W + nrate · (DH^T · X_in)
     * where nrate = −lr/B encodes the negative learning rate and averaging. */
    const float nrate = -lr / (float)B;

    /* X ← X + nrate * (DO^T · H)   -> (nops×nhid) */
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                nops, nhid, B,
                nrate,
                DO, nops,
                H,  nhid,
                1.0f,
                nn->x, nhid);

    /* W ← W + nrate * (DH^T · X)   -> (nhid×nips) */
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                nhid, nips, B,
                nrate,
                DH, nhid,
                X,  nips,
                1.0f,
                nn->w, nips);

    /* 8) Bias updates (averaged) — **per-unit** reductions
     * bo[i] += nrate * Σ_r DO[r,i]
     * bh[j] += nrate * Σ_r DH[r,j]
     */
#if defined(_MSC_VER) || defined(FWC_NO_VLA)
    float *acc_bo = (float*)calloc((size_t)nops, sizeof *acc_bo);
    float *acc_bh = (float*)calloc((size_t)nhid, sizeof *acc_bh);
    if (!acc_bo || !acc_bh) { free(acc_bo); free(acc_bh); return; }
#else
    float acc_bo[nops];  memset(acc_bo, 0, sizeof acc_bo);
    float acc_bh[nhid];  memset(acc_bh, 0, sizeof acc_bh);
#endif

    /* accumulate DO over batch into acc_bo[i] */
    for (int r = 0; r < B; ++r) {
        const float *d_row = DO + (size_t)r * nops;
        for (int i = 0; i < nops; ++i) acc_bo[i] += d_row[i];
    }
    /* accumulate DH over batch into acc_bh[j] */
    for (int r = 0; r < B; ++r) {
        const float *h_row = DH + (size_t)r * nhid;
        for (int j = 0; j < nhid; ++j) acc_bh[j] += h_row[j];
    }

    /* apply updates */
    for (int i = 0; i < nops; ++i) bo[i] += nrate * acc_bo[i];
    for (int j = 0; j < nhid; ++j) bh[j] += nrate * acc_bh[j];

#if defined(_MSC_VER) || defined(FWC_NO_VLA)
    free(acc_bo);
    free(acc_bh);
#endif
}

/* ---------------------------------- FWC_OAT ------------------------------------------- */
#ifdef FWC_OAT
/* ---- simple monotonic clock in ms ---- */
static inline float fwc_now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (float)ts.tv_sec * 1000.0f + (float)ts.tv_nsec * 1e-6f;
}

/* time one training step with lr=0 (no updates) for batch size B */
static float fwc_time_step_ms(NeuralNetwork_Type *nn, int B,
                              const float *X, const float *Y,
                              int warmups, int reps)
{
    for (int i=0;i<warmups;++i) NNtrain_batch(nn, B, X, Y, 0.0f);
    float best = 1e9f;
    for (int r=0;r<reps;++r) {
        float t0 = fwc_now_ms();
        NNtrain_batch(nn, B, X, Y, 0.0f);
        float dt = fwc_now_ms() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

/* OLS fit for T ≈ α + β·B */
static void fwc_fit_alpha_beta(const int *B, const float *T, int n,
                               float *alpha_ms, float *beta_ms)
{
    double sB=0, sT=0, sBB=0, sBT=0;
    for (int i=0;i<n;++i){ sB+=B[i]; sT+=T[i]; sBB+= (double)B[i]*B[i]; sBT+= (double)B[i]*T[i]; }
    double den = n*sBB - sB*sB;
    if (den <= 0) { *alpha_ms=T[0]; *beta_ms=(n>=2? (T[1]-T[0])/(B[1]-B[0]) : 0.0f); return; }
    *beta_ms  = (float)((n*sBT - sB*sT)/den);
    *alpha_ms = (float)((sT - (*beta_ms)*sB)/n);
}

/* Pick GEMM vs GEMV for single-sample hidden matvec */
static int fwc_pick_single_kernel(const NeuralNetwork_Type *nn,
                                  const float *in, float *h_buf)
{
#if defined(FWC_USE_BLAS) && (FWC_USE_BLAS+0)==1
    const int nips = nn->nips, nhid = nn->nhid;
    float best_gemm = 1e9f, best_gemv = 1e9f;
    /* GEMM: (1×nips) · (nips×nhid) */
    for (int r=0;r<5;++r){
        float t0=fwc_now_ms();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    1, nhid, nips, 1.0f,
                    in, nips, nn->w, nips, 0.0f, h_buf, nhid);
        float dt=fwc_now_ms()-t0; if (dt<best_gemm) best_gemm=dt;
    }
    /* GEMV: (nhid×nips) · (nips×1) */
    for (int r=0;r<5;++r){
        float t0=fwc_now_ms();
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    nhid, nips, 1.0f, nn->w, nips, in, 1, 0.0f, h_buf, 1);
        float dt=fwc_now_ms()-t0; if (dt<best_gemv) best_gemv=dt;
    }
    return (best_gemv < best_gemm); /* 1 ⇒ use GEMV */
#else
    (void)in; (void)h_buf;
    return 0;
#endif
}

void NN_oat_calibrate(NeuralNetwork_Type *nn,
                      const int *Bgrid, int nB,
                      int repeats, float target_ms)
{
    if (!nn || nB <= 0) return;

    const int nips = nn->nips, nops = nn->nops;
    const int Bmax = Bgrid[nB-1];

    /* synthesize small random mini-batch once (row-major) */
    float *X = (float*)malloc((size_t)Bmax * nips * sizeof *X);
    float *Y = (float*)malloc((size_t)Bmax * nops * sizeof *Y);
    if (!X || !Y) { free(X); free(Y); return; }

    uint32_t seed = 0x12345678u;
    for (size_t i=0;i<(size_t)Bmax*nips;++i){ seed^=seed<<13; seed^=seed>>17; seed^=seed<<5;
        X[i] = ((int)(seed & 0xFFFF) / 32768.0f) - 1.0f;
    }
    for (int r=0;r<Bmax;++r){
        int cls = (seed = seed*1664525u + 1013904223u) % nops;
        for (int i=0;i<nops;++i) Y[(size_t)r*nops+i] = (i==cls)? 1.0f : 0.0f;
    }

    float *Tms = (float*)malloc(nB * sizeof *Tms);
    for (int i=0;i<nB;++i) {
        const int B = Bgrid[i];
        Tms[i] = fwc_time_step_ms(nn, B, X, Y, /*warmups*/2, /*reps*/repeats);
    }

    float a,b; fwc_fit_alpha_beta(Bgrid, Tms, nB, &a, &b);

    /* choose B* */
    int Bstar = Bgrid[0];
    if (target_ms > 0.0f) {
        for (int i=0;i<nB;++i){
            float pred = a + b * (float)Bgrid[i];
            if (pred <= target_ms) Bstar = Bgrid[i];
        }
    } else {
        float best_eff = 0.0f;
        for (int i=0;i<nB;++i){
            float eff = (float)Bgrid[i] / (a + b * (float)Bgrid[i]);
            if (eff > best_eff){ best_eff = eff; Bstar = Bgrid[i]; }
        }
    }

    /* kernel pick for single-sample predict */
    float *h_buf = (float*)malloc((size_t)nn->nhid * sizeof *h_buf);
    int use_gemv = 0;
    if (h_buf) use_gemv = fwc_pick_single_kernel(nn, X, h_buf);
    free(h_buf);

    nn->oat.enabled         = 1;
    nn->oat.alpha_ms        = a;
    nn->oat.beta_ms         = b;
    nn->oat.B_star          = Bstar;
    nn->oat.use_gemv_single = use_gemv;
    nn->oat.target_ms       = target_ms;

    free(Tms); free(X); free(Y);
}

int NN_oat_get_B(const NeuralNetwork_Type *nn){
    if (!nn) return 32;
    if (!nn->oat.enabled) return 32;
    return (nn->oat.B_star > 0 ? nn->oat.B_star : 32);
}
#endif /* FWC_OAT */
/* ---------------------------------- FWC_OAT ------------------------------------------- */