#include "nn.h"
#include <stdlib.h>
#include <math.h>          /* fabsf */

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





/* ───────────────────────────────────────────────────────────────── */

/* Portable, fused, softmax-capable forward pass (fixed & polished) */
static void fprop(const NeuralNetwork_Type nn, const float * const in)
{
    const int nips = nn.nips;
    const int nhid = nn.nhid;
    const int nops = nn.nops;

    const float *w = nn.w;  /* input→hidden  (nhid × nips), row-major */
    const float *x = nn.x;  /* hidden→output (nops × nhid), row-major */
    const float *b = nn.b;  /* b[0]=hidden bias, b[1]=output bias (scalar) */
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

    /* Fuse bias + activation for hidden */
    const float bh = b[0];
    for (int j = 0; j < nhid; ++j) {
        const float z = h[j] + bh;
#if defined(FWC_CACHE_Z)
        nn.hz[j] = z;            /* <-- needed for fast_sigmoid_grad in bprop */
#endif
        h[j] = fast_sigmoid(z);
        /* For ReLU speed, swap to: h[j] = z > 0.f ? z : 0.f; */
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
        const float bo = b[1];
        float m = -FLT_MAX;
        for (int i = 0; i < nops; ++i) { o[i] += bo; if (o[i] > m) m = o[i]; }
        float s = 0.0f;
        for (int i = 0; i < nops; ++i) { o[i] = expf(o[i] - m); s += o[i]; }
        const float invs = s > 0.0f ? (1.0f / s) : 1.0f;
        for (int i = 0; i < nops; ++i) o[i] *= invs;
        return;
    }

    /* binary / regression-like path: sigmoid */
    {
        const float bo = b[1];
        for (int i = 0; i < nops; ++i) {
            const float z = o[i] + bo;
            o[i] = fast_sigmoid(z);
        }
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
    float *restrict b = nn.b;
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
        for (int j = 0; j < nops; ++j)
            delta_o[j] = o[j] - tg[j];                  /* softmax + CE */
    } else {
        for (int j = 0; j < nops; ++j) {
            const float oj  = o[j];
            const float err = oj - tg[j];
#if defined(FWC_CACHE_Z)
            /* If you cache oz and use fast_sigmoid at output, prefer:
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
        /* exact grad of fast_sigmoid using cached z */
        delta_h[i] *= fast_sigmoid_grad(hz[i]);
    #else
        /* fallback: classic logistic derivative using h */
        const float hi = h[i];
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

    /* Biases (scalar per layer) */
    float sum_do = 0.0f, sum_dh = 0.0f;
    int j = 0;
    for (; j <= nops - 4; j += 4)
        sum_do += delta_o[j] + delta_o[j+1] + delta_o[j+2] + delta_o[j+3];
    for (; j < nops; ++j) sum_do += delta_o[j];

    int i = 0;
    for (; i <= nhid - 4; i += 4)
        sum_dh += delta_h[i] + delta_h[i+1] + delta_h[i+2] + delta_h[i+3];
    for (; i < nhid; ++i) sum_dh += delta_h[i];

    b[1] += nrate * sum_do;
    b[0] += nrate * sum_dh;

#if defined(_MSC_VER) || defined(FWC_NO_VLA)
    free(delta_h);
    free(delta_o);
#endif
}


static inline void wbrand(const NeuralNetwork_Type nn) {
    const int nips = nn.nips;
    const int nhid = nn.nhid;
    const int nops = nn.nops;

    float *w = nn.w;  /* input→hidden (nhid×nips), row-major */
    float *x = nn.x;  /* hidden→output (nops×nhid), row-major */
    float *b = nn.b;  /* b[0]=hidden bias, b[1]=output bias */

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
       Use Xavier uniform for softmax/sigmoid outputs (works well regardless of hidden nonlinearity).
    */
    const float limit_o = sqrtf(6.0f / ((float)nhid + (float)nops)); /* Xavier uniform */
    for (int k = 0; k < nops; ++k) {
        for (int j = 0; j < nhid; ++j) {
            x[(size_t)k * nhid + j] = (2.0f * frand() - 1.0f) * limit_o;
        }
    }

    /* -------- Biases -------- */
    b[0] = 0.0f;  /* hidden bias */
    b[1] = 0.0f;  /* output bias */
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

    // 3) hidden bias + activation
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < needH; ++i) {
        const float z = H[i] + nn.b[0];
        H[i] = fast_sigmoid(z);
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

    // 1) Store dimensions up-front
    nn.nips = nips;
    nn.nhid = nhid;
    nn.nops = nops;

    // 2) Fixed bias count
    const int nb = 2;
    nn.nb = nb;

    // 3) Compute weight-matrix sizes once
    const int wih = nips * nhid;        // input→hidden
    const int who = nhid * nops;        // hidden→output
    nn.nw = wih + who;

    // 4) Allocate weights (w) and set x to the hidden→output subarray
    nn.w = (float*)calloc((size_t)nn.nw, sizeof *nn.w);
    if (!nn.w) goto fail_w;
    nn.x = nn.w + wih;

    // 5) Allocate biases
    nn.b = (float*)calloc((size_t)nb, sizeof *nn.b);
    if (!nn.b) goto fail_b;

    // 6) Allocate hidden-layer buffer
    nn.h = (float*)calloc((size_t)nhid, sizeof *nn.h);
    if (!nn.h) goto fail_h;

    // 7) Allocate output-layer buffer
    nn.o = (float*)calloc((size_t)nops, sizeof *nn.o);
    if (!nn.o) goto fail_o;

#if defined(FWC_CACHE_Z)
    // 8) Allocate hidden pre-activations (z) cache (optional)
    nn.hz = (float*)calloc((size_t)nhid, sizeof *nn.hz);
    if (!nn.hz) goto fail_hz;
    /* If you ever enable oz for binary output, allocate it here as well. */
#endif

    // 9) Initialize weights & biases
    wbrand(nn);
    return nn;

  // --- error cleanup ---
#if defined(FWC_CACHE_Z)
fail_hz:
    free(nn.o);
#endif
fail_o:
    free(nn.h);
fail_h:
    free(nn.b);
fail_b:
    free(nn.w);
fail_w:
    // return zeroed struct on failure
    return (NeuralNetwork_Type){0};
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

    const size_t needH  = (size_t)B * (size_t)nhid;
    const size_t needO  = (size_t)B * (size_t)nops;

    if (capH  < needH) { free(H);  H  = (float*)malloc(needH * sizeof *H);  capH  = needH; }
    if (capO  < needO) { free(O);  O  = (float*)malloc(needO * sizeof *O);  capO  = needO; }
    if (capDO < needO) { free(DO); DO = (float*)malloc(needO * sizeof *DO); capDO = needO; }
    if (capDH < needH) { free(DH); DH = (float*)malloc(needH * sizeof *DH); capDH = needH; }
    if (!H || !O || !DO || !DH) return;

    /* 1) H = X · W^T  (B×nhid) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, nhid, nips,
                1.0f,
                X,      nips,
                nn->w,  nips,
                0.0f,
                H,      nhid);

    /* 2) hidden bias + activation (fused) */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < needH; ++i) {
        const float z = H[i] + nn->b[0];
        H[i] = fast_sigmoid(z);  /* swap to ReLU if you enable FWC_RELU_HID */
    }

    /* 3) O = H · X^T  (B×nops) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, nops, nhid,
                1.0f,
                H,      nhid,
                nn->x,  nhid,
                0.0f,
                O,      nops);

    /* 4) output epilogue + deltas DO */
    if (nops > 1) {
        /* softmax + CE: DO = softmax(O+bo) - Y */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int r = 0; r < B; ++r) {
            float *o = O + (size_t)r * nops;
            const float *y = Y + (size_t)r * nops;

            float m = -FLT_MAX;
            for (int i = 0; i < nops; ++i) { o[i] += nn->b[1]; if (o[i] > m) m = o[i]; }
            float s = 0.0f;
            for (int i = 0; i < nops; ++i) { o[i] = expf(o[i] - m); s += o[i]; }
            const float invs = (s > 0.0f) ? (1.0f / s) : 1.0f;
            for (int i = 0; i < nops; ++i) {
                o[i] *= invs;
                DO[(size_t)r * nops + i] = o[i] - y[i];
            }
        }
    } else {
        /* sigmoid: DO = σ(O+bo) - Y  (BCE-style gradient) */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < needO; ++i) {
            const float z  = O[i] + nn->b[1];
            const float oi = fast_sigmoid(z);
            DO[i] = oi - Y[i];
            O[i]  = oi; /* keep post-activation if you want it later */
        }
    }

    /* 5) DH = DO · X (hidden deltas before nonlinearity)  (B×nhid) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                B, nhid, nops,
                1.0f,
                DO,     nops,
                nn->x,  nhid,
                0.0f,
                DH,     nhid);

    /* 6) multiply by activation derivative at hidden (sigmoid) */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < needH; ++i) {
        const float hi = H[i];               /* H holds σ(z) */
        DH[i] *= hi * (1.0f - hi);           /* σ'(z) = σ(z)(1-σ(z)) */
        /* If you switch to ReLU and cache z: DH[i] *= (z > 0.0f); */
    }

    /* 7) SGD update (ONCE per batch) — use AVERAGED gradient */
    const float nrate = -lr / (float)B;

    /* nn->x ← nn->x + nrate * (DO^T · H)   -> (nops×nhid) */
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                nops, nhid, B,
                nrate,
                DO, nops,
                H,  nhid,
                1.0f,
                nn->x, nhid);

    /* nn->w ← nn->w + nrate * (DH^T · X)   -> (nhid×nips) */
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                nhid, nips, B,
                nrate,
                DH, nhid,
                X,  nips,
                1.0f,
                nn->w, nips);

    /* 8) bias updates (also averaged) */
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
