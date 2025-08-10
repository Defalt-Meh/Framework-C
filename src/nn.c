#include "nn.h"
#include <stdlib.h>
#include <math.h>          /* fabsf */
/* helpers near the top (after includes) */
#define BH_PTR(nn)  ((nn).b)                             /* hidden biases: nhid  */
#define BO_PTR(nn)  ((nn).b + (nn).nhid)                 /* output biases: nops  */

/* for 2-layer nets */
#define BH1_PTR(nn) ((nn).b)                             /* nhid   */
#define BH2_PTR(nn) ((nn).b + (nn).nhid)                 /* nhid2  */
#define BO2_PTR(nn) ((nn).b + (nn).nhid + (nn).nhid2)    /* nops   */

#ifndef FWC_RELU_HID
#error "FWC_RELU_HID not defined: you're training with sigmoid!"
#endif



/* ───────────────────── OPENMP selection ─────────────────── */
#ifdef _OPENMP
    #include <omp.h>
#endif
/* ───────────────────────────────────────────────────────────────── */

/* ───────────────────── BLAS backend selection ─────────────────── */
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
static float toterr(const float *tg, const float *o, int size);
static float pderr (float a, const float b);
static float frand (void);

/* ─────────────—— Helper implementations (unchanged) —──────────── */
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
/* Forward: σ_fast(x) = 0.5 * (x / (1 + |x|)) + 0.5
   - Uses one fabsf
   - Uses fmaf to keep precision and ILP
*/
static inline float fast_sigmoid(float x)
{
    const float a   = fabsf(x);
    const float den = 1.0f + a;
    const float inv = 1.0f / den;                  // ok with -ffast-math
    return fmaf(x, 0.5f * inv, 0.5f);              // 0.5*(x*inv) + 0.5
}

/* Backward: σ_fast'(x) = 0.5 * (1 + |x|)^(-2)
   Use when you have dL/dσ and need dL/dx = dL/dσ * σ'(x)
*/
static inline float fast_sigmoid_grad(float x)
{
    const float a   = fabsf(x);
    const float den = 1.0f + a;
    const float inv = 1.0f / den;
    return 0.5f * inv * inv;                        // 0.5 / (1+|x|)^2
}

/* Fused, numerically-stable softmax + CE for one row.
   - logits: length n; if write_probs!=0, overwritten with probabilities
   - y:      one-hot targets in {0,1}
   - delta:  output (softmax - y)
   - bias:   scalar output bias (nn->b[1])
   Returns per-row CE loss (you can ignore the return). */
static inline float softmax_ce_fused_row(float *restrict logits,
                                         const float *restrict y,
                                         float *restrict delta,
                                         const int n,
                                         const float bias,
                                         const int write_probs)
{
    float m = logits[0] + bias;
    for (int i = 1; i < n; ++i) {
        const float z = logits[i] + bias;
        if (z > m) m = z;
    }
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += expf((logits[i] + bias) - m);
    const float logZ = logf(s) + m;

    float loss = 0.0f;
    const float inv_guard = 1e-20f;  /* avoid log(0) */
    for (int i = 0; i < n; ++i) {
        const float p = expf((logits[i] + bias) - logZ);
        delta[i] = p - y[i];
        if (write_probs) logits[i] = p;
        loss -= y[i] * logf(p + inv_guard);
    }
    return loss;
}

/* Mean CE over a batch (softmax case). If DO!=NULL, also fills deltas. */
static inline float softmax_ce_batch(float *O_logits, const float *Y,
                                     float *DO, int B, int nops, float bias)
{
    float sum = 0.0f;
    for (int r = 0; r < B; ++r) {
        float *o_row       = O_logits + (size_t)r * nops;
        const float *y_row = Y        + (size_t)r * nops;
        float *d_row       = DO ? (DO + (size_t)r * nops) : (float*)0;
        sum += softmax_ce_fused_row(o_row, y_row, d_row ? d_row : (float[1]){0},
                                    nops, bias, /*write_probs=*/0);
    }
    return sum / (float)B;
}




/* ───────────────────────────────────────────────────────────────── */

/* Portable, fused, softmax-capable forward pass (ReLU-ready) */
/* Bias layout assumed: b[0..nhid-1]=hidden, b[nhid..nhid+nops-1]=output */
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

    /* -------- Hidden: h = in · w^T -------- */
#if defined(FWC_USE_BLAS) && (FWC_USE_BLAS+0)==1
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                1, nhid, nips,
                1.0f, in, nips,
                      w,  nips,
                0.0f, h,  nhid);
#else
    for (int j = 0; j < nhid; ++j) {
        const float *wj = w + (size_t)j * nips;
        float acc = 0.0f;
        for (int k = 0; k < nips; ++k) acc += in[k] * wj[k];
        h[j] = acc;
    }
#endif

    /* -------- Hidden epilogue: per-unit bias + activation -------- */
    for (int j = 0; j < nhid; ++j) {
        const float z = h[j] + bh[j];
#if defined(FWC_CACHE_Z)
        nn.hz[j] = z;                      /* keep pre-activation if needed */
#endif
#if defined(FWC_RELU_HID)
        h[j] = (z > 0.0f) ? z : 0.0f;      /* ReLU */
#else
        h[j] = fast_sigmoid(z);            /* Sigmoid */
#endif
    }

    /* -------- Output: o = h · x^T -------- */
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

    /* -------- Output epilogue -------- */
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
    /* const float *restrict oz = nn.oz; */
#endif

    /* Portable allocation for delta buffers (avoid VLAs on MSVC) */
#if defined(_MSC_VER) || defined(FWC_NO_VLA)
    float *delta_o = (float*)malloc((size_t)nops * sizeof(float));
    float *delta_h = (float*)malloc((size_t)nhid * sizeof(float));
    if (!delta_o || !delta_h) { free(delta_o); free(delta_h); return; }
#else
    float delta_o[nops];
    float delta_h[nhid];
#endif

    /* 1) Output deltas */
    if (nops > 1) {
        /* softmax + cross-entropy: grad = p - y */
        for (int j = 0; j < nops; ++j)
            delta_o[j] = o[j] - tg[j];
    } else {
        /* sigmoid + BCE (or MSE-style): grad = (σ - y) * σ'(z)  ; here σ stored in o */
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

    /* 2) Hidden deltas: delta_h = X^T * delta_o */
    cblas_sgemv(CblasRowMajor, CblasTrans,
                nops, nhid, 1.0f,
                X, nhid, delta_o, 1,
                0.0f, delta_h, 1);

    /* Multiply by activation derivative at hidden */
#if defined(FWC_RELU_HID)
    for (int i = 0; i < nhid; ++i) {
    #if defined(FWC_CACHE_Z)
        delta_h[i] *= (hz[i] > 0.0f) ? 1.0f : 0.0f;
    #else
        delta_h[i] *= (h[i] > 0.0f) ? 1.0f : 0.0f;
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

    /* 3) Parameter updates */
    const float nrate = -rate;

    /* X ← X + nrate * (delta_o · h^T) */
    cblas_sger(CblasRowMajor, nops, nhid, nrate,
               delta_o, 1, h, 1, X, nhid);

    /* W ← W + nrate * (delta_h · in^T) */
    cblas_sger(CblasRowMajor, nhid, nips, nrate,
               delta_h, 1, in, 1, W, nips);

    /* 4) Per‑unit bias updates (instead of scalar layer biases) */
    float *bh = b;            /* hidden biases: nhid elements */
    float *bo = b + nhid;     /* output biases: nops elements */

    for (int i = 0; i < nops; ++i)  bo[i] += nrate * delta_o[i];
    for (int j = 0; j < nhid; ++j)  bh[j] += nrate * delta_h[j];

#if defined(_MSC_VER) || defined(FWC_NO_VLA)
    free(delta_h);
    free(delta_o);
#endif
}



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
    */
#if defined(FWC_RELU_HID)
    const float limit_h = sqrtf(6.0f / (float)nips);                 /* He uniform */
#else
    const float limit_h = sqrtf(6.0f / ((float)nips + (float)nhid)); /* Xavier uniform */
#endif

    for (int j = 0; j < nhid; ++j) {
        for (int i = 0; i < nips; ++i) {
            w[(size_t)j * nips + i] = (2.0f * frand() - 1.0f) * limit_h;
        }
    }

    /* -------- Output layer init --------
       Use Xavier uniform for softmax/sigmoid outputs
    */
    const float limit_o = sqrtf(6.0f / ((float)nhid + (float)nops));
    for (int k = 0; k < nops; ++k) {
        for (int j = 0; j < nhid; ++j) {
            x[(size_t)k * nhid + j] = (2.0f * frand() - 1.0f) * limit_o;
        }
    }

    /* -------- Biases -------- */
    for (int j = 0; j < nhid; ++j) bh[j] = 0.0f;  /* hidden biases */
    for (int i = 0; i < nops; ++i) bo[i] = 0.0f;  /* output biases */
}


/* ─────────────────────────────────────────────────────────── */
/* 2-hidden-layer builder                                     */
/* in → h1(nhid) → h2(nhid2) → out(nops)                      */
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

    /* He/Xavier limits */
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

    b[0] = 0.0f;  /* h1 bias */
    b[1] = 0.0f;  /* h2 bias */
    b[2] = 0.0f;  /* out bias */
}



/* 0.5 * sum_i (tg[i] - o[i])^2  — portable & fast (no malloc, no vDSP) */
static inline float toterr(const float *restrict tg,
                           const float *restrict o,
                           const int size)
{
    float sum = 0.0f;
    /* Let the compiler/OMP vectorize & reduce */
    #if defined(_OPENMP)
    #pragma omp simd reduction(+:sum)
    #endif
    for (int i = 0; i < size; ++i) {
        const float d = tg[i] - o[i];
        sum = fmaf(d, d, sum);   /* sum += d*d with one rounding */
    }
    return 0.5f * sum;
}


/* Exposed Functions in Header File */

float * NNpredict(const NeuralNetwork_Type nn,
                  const float * in)
{
    /* Hoist the output pointer into a register once */
    float * const out = nn.o;

    /* Forward-propagate (fprop is already static inline, so 
       the compiler will inline it here and eliminate the call) */
    fprop(nn, in);

    /* Return the pre-loaded output buffer */
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
    static float *H = NULL;
    static size_t Hcap = 0;
    const size_t needH = (size_t)B * nhid;
    if (Hcap < needH) {
        free(H);
        H = (float*)malloc(needH * sizeof *H);
        Hcap = needH;
    }

    // 2) input→hidden
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, nhid, nips,
                1.0f,
                batch_in, nips,
                nn.w,     nips,
                0.0f,
                H,        nhid);

// 3) hidden bias + activation   (FIX: respect FWC_RELU_HID here, too)
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
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, nops, nhid,
                1.0f,
                H,    nhid,
                nn.x, nhid,
                0.0f,
                batch_out, nops);

    // 5) output epilogue: softmax (nops>1) or sigmoid
    const size_t needO = (size_t)B * nops;
    if (nops > 1) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int r = 0; r < B; ++r) {
            float *row = batch_out + (size_t)r * nops;
            float m = -FLT_MAX;
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
    NeuralNetwork_Type nn;
    nn.nips = nips; nn.nhid = nhid; nn.nops = nops;

    nn.nb = nhid + nops;                 /* CHANGED: per‑unit biases */
    const int wih = nips * nhid;
    const int who = nhid * nops;
    nn.nw = wih + who;

    nn.w = (float*)calloc((size_t)nn.nw, sizeof *nn.w);
    if (!nn.w) goto fail_w;
    nn.x = nn.w + wih;

    nn.b = (float*)calloc((size_t)nn.nb, sizeof *nn.b);  /* CHANGED size */
    if (!nn.b) goto fail_b;

    nn.h = (float*)calloc((size_t)nhid, sizeof *nn.h);
    if (!nn.h) goto fail_h;
    nn.o = (float*)calloc((size_t)nops, sizeof *nn.o);
    if (!nn.o) goto fail_o;

#if defined(FWC_CACHE_Z)
    nn.hz = (float*)calloc((size_t)nhid, sizeof *nn.hz);
    if (!nn.hz) goto fail_hz;
#endif

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
    NeuralNetwork_Type nn;
    nn.nips = nips; nn.nhid = nhid; nn.nhid2 = nhid2 > 0 ? nhid2 : nhid; nn.nops = nops;

    nn.nb = nhid + nn.nhid2 + nops;      /* CHANGED: per‑unit biases */

    const int wih  = nips * nhid;
    const int h2h1 = nhid * nn.nhid2;
    const int who  = nn.nhid2 * nops;
    nn.nw = wih + h2h1 + who;

    nn.w = (float*)calloc((size_t)nn.nw, sizeof *nn.w);
    if (!nn.w) goto fail_w;
    nn.u = nn.w + wih;
    nn.x = nn.u + h2h1;

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
/* Auto‑depth builder: decides inside the function             */
/* Pass dataset size N; tweak thresholds as you like           */
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
        /* small dataset → 1 hidden layer */
        return NNbuild(nips, nhid, nops);
    }

    /* medium/large dataset → 2 hidden layers */
    int nhid2;
    if (N < FWC_AUTO_N_MED) nhid2 = nhid > 1 ? (nhid/2) : nhid;  /* conservative */
    else                    nhid2 = nhid;                         /* same width */

    return NNbuild2(nips, nhid, nhid2, nops);
}


/* nn.c (already contains static fprop & static bprop) */

float NNtrain(const NeuralNetwork_Type nn,
              const float *in,
              const float *tg,
              float rate)
{
    /* 1. forward pass */
    fprop(nn, in);

    /* 2. backward pass + weight update */
    bprop(nn, in, tg, rate);

    /* 3. return sample loss                       */
    return toterr(tg, nn.o, nn.nops);   /* 0.5 · Σ(t−o)² */
}


/**
 * And it came to pass at the fateful hour:
 * “Write down the network unto the scroll,”
 * that it might be restored when the dawn of inference breaks.
 */
void NNsave(const NeuralNetwork_Type nn, const char * path)
{
    /* Open the scroll for writing */
    FILE * const file = fopen(path, "w");
    if (!file) {
        /* If the scroll is sealed, abort the saving ritual */
        perror("NNsave: fopen failed");
        return;
    }

    /* Buffer the writes, that many lines may flow swiftly */
    setvbuf(file, NULL, _IOFBF, BUFSIZ);

    /* Hoist fields into locals for swifter access */
    const int nips = nn.nips;
    const int nhid = nn.nhid;
    const int nops = nn.nops;
    const int nb   = nn.nb;
    const int nw   = nn.nw;
    const float *b = nn.b;
    const float *w = nn.w;

    /* Save the Header */
    fprintf(file, "%d %d %d\n", nips, nhid, nops);

    /* Save the Biases — anoint each neuron’s offset */
    for (const float *bp = b, *bend = b + nb; bp < bend; ++bp) {
        fprintf(file, "%f\n", (double)*bp);
    }

    /* Save the Weights — inscribe each connection’s strength */
    for (const float *wp = w, *wend = w + nw; wp < wend; ++wp) {
        fprintf(file, "%f\n", (double)*wp);
    }

    /* Close the scroll */
    fclose(file);
}


/**
 * And it came to pass, the seeker summoned NNload,
 * that the network might rise again from the scroll of bytes.
 */
NeuralNetwork_Type NNload(const char * path)
{
    /* Open the sacred scroll for reading */
    FILE * const file = fopen(path, "r");
    if (!file) {
        perror("NNload: fopen failed");
        return (NeuralNetwork_Type){0};
    }
    /* Prepare the vessel for swift reads */
    setvbuf(file, NULL, _IOFBF, BUFSIZ);

    /* Read the divine dimensions: inputs, hidden, outputs */
    int nips = 0, nhid = 0, nops = 0;
    if (fscanf(file, "%d %d %d\n", &nips, &nhid, &nops) != 3) {
        perror("NNload: invalid header");
        fclose(file);
        return (NeuralNetwork_Type){0};
    }

    /* Build the tabernacle of neurons */
    NeuralNetwork_Type nn = NNbuild(nips, nhid, nops);
    /* If the tabernacle failed to rise, abort */
    if (nn.nw == 0 && nn.b == NULL) {
        fclose(file);
        return nn;
    }

    /* Hoist counts and pointers for biases and weights */
    const int nb     = nn.nb;
    const int nw     = nn.nw;
    float    *b      = nn.b;
    float    *w      = nn.w;

    /* Load the Biases — each offset anointed anew */
    for (float *bp = b, *bend = b + nb; bp < bend; ++bp) {
        if (fscanf(file, "%f\n", bp) != 1) {
            perror("NNload: reading bias failed");
            NNfree(nn);
            fclose(file);
            return (NeuralNetwork_Type){0};
        }
    }

    /* Load the Weights — each connection’s strength inscribed */
    for (float *wp = w, *wend = w + nw; wp < wend; ++wp) {
        if (fscanf(file, "%f\n", wp) != 1) {
            perror("NNload: reading weight failed");
            NNfree(nn);
            fclose(file);
            return (NeuralNetwork_Type){0};
        }
    }

    /* Close the scroll and return the living network */
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
    /* “Let there be a vault for the highest measure” */
    float max = arr[0];
    int idx = 0;

    /* “Traverse the fields from the first to the last,
     * that each value may be inscribed upon the scroll.” */
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
    /* “And let the index of the mightiest be proclaimed.” */
    printf("The number is: %d\n", idx);
}


/**
 * And it came to pass at the end of the epoch:
 * “Disperse ye the vessels of memory,”
 * freeing outputs, hidden, biases, and weights,
 * that no clutter remain.
 */
/* Free buffers in creation order (safe if any are NULL) */
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
    memset(nn, 0, sizeof *nn);
}


#ifndef FWC_DROPOUT
#define FWC_DROPOUT 0          /* set to 1 to enable hidden-layer dropout */
#endif
#ifndef FWC_DROPOUT_P
#define FWC_DROPOUT_P 0.20f    /* drop probability */
#endif

/* tiny RNG for stochastic mask */
static inline uint32_t fwc_xorshift32(uint32_t *s){
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s;
}
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

    /* ---- scratch (static to avoid hot-path malloc/free) ---- */
    static float *H = NULL, *O = NULL, *DO = NULL, *DH = NULL;
    static size_t capH = 0, capO = 0, capDO = 0, capDH = 0;

#if FWC_DROPOUT
    static uint8_t *M = NULL;   /* dropout mask for hidden activations */
    static size_t capM = 0;
    const float p_drop = FWC_DROPOUT_P;
    const float keep   = 1.0f - p_drop;          /* inverted dropout scale */
    uint32_t seed = 0x9e3779b9u;                 /* per-call seed; make global if you want reproducibility across calls */
#endif

    const size_t needH  = (size_t)B * (size_t)nhid;
    const size_t needO  = (size_t)B * (size_t)nops;

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

    /* 1) H = X · W^T  (B×nhid) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, nhid, nips,
                1.0f,
                X,      nips,
                nn->w,  nips,
                0.0f,
                H,      nhid);

    /* 2) hidden bias + activation (Sigmoid or ReLU) + (optional) inverted dropout */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < needH; ++i) {
        const float z = H[i] + nn->b[0];
    #if defined(FWC_RELU_HID)
        float a = (z > 0.0f) ? z : 0.0f;   /* ReLU */
    #else
        float a = fast_sigmoid(z);         /* Sigmoid */
    #endif
    #if FWC_DROPOUT
        /* generate/record mask */
        float u = fwc_rand01(&seed);
        uint8_t m = (u >= p_drop);       /* keep if u>=p_drop */
        M[i] = m;
        H[i] = m ? (a / keep) : 0.0f;    /* inverted dropout keeps expectation */
    #else
        H[i] = a;
    #endif
    }

    /* 3) O = H · X^T  (B×nops) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, nops, nhid,
                1.0f,
                H,      nhid,
                nn->x,  nhid,
                0.0f,
                O,      nops);

    /* 4) output epilogue + deltas DO (softmax + CE fused for nops>1) */
    if (nops > 1) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int r = 0; r < B; ++r) {
            float       *o_row = O  + (size_t)r * nops;  /* logits row */
            const float *y_row = Y  + (size_t)r * nops;  /* target row */
            float       *d_row = DO + (size_t)r * nops;  /* delta row  */
            (void)softmax_ce_fused_row(o_row, y_row, d_row, nops, nn->b[1], /*write_probs=*/1);
        }
    } else {
        /* sigmoid output: DO = σ(O+bo) - Y  (BCE-style gradient) */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < needO; ++i) {
            const float z  = O[i] + nn->b[1];
            const float oi = fast_sigmoid(z);
            DO[i] = oi - Y[i];
            O[i]  = oi;
        }
    }

    /* 5) DH = DO · X  (B×nhid) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                B, nhid, nops,
                1.0f,
                DO,     nops,
                nn->x,  nhid,
                0.0f,
                DH,     nhid);

    /* 6) multiply by activation derivative at hidden (and gate with dropout mask) */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < needH; ++i) {
    #if FWC_DROPOUT
        if (!M[i]) { DH[i] = 0.0f; continue; }  /* dropped units carry no gradient */
    #endif
    #if defined(FWC_RELU_HID)
        /* ReLU' using post-activation (H already includes dropout scaling) */
        DH[i] *= (H[i] > 0.0f) ? 1.0f : 0.0f;
    #else
        const float hi = H[i];                  /* H holds σ(z) (possibly scaled) */
        DH[i] *= hi * (1.0f - hi);              /* sigmoid' */
    #endif
    }

    /* 7) SGD update (averaged over batch) */
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

    /* 8) Biases (averaged) */
    float sum_do = 0.0f, sum_dh = 0.0f;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:sum_do)
#endif
    for (size_t i = 0; i < needO; ++i) sum_do += DO[i];

#ifdef _OPENMP
    #pragma omp parallel for reduction(+:sum_dh)
#endif
    for (size_t i = 0; i < needH; ++i) sum_dh += DH[i];

    nn->b[1] += nrate * sum_do;
    nn->b[0] += nrate * sum_dh;
}
