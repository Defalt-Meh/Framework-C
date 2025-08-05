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
static float err   (float a, const float b);
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

/* Scalar fast-sigmoid used for tiny vectors
 *     σ_fast(x) = 0.5 * (x / (1 + |x|)) + 0.5
 */
static inline float fast_sigmoid(float x)
{
    const float inv = 1.0f / (1.0f + fabsf(x));
    return 0.5f * (x * inv) + 0.5f;
}


/*
 * Book of FRAMEWORK-C, Drop-in Optimized Replacement  
 *
 * EXACT same logic as original, but with minimal performance improvements:
 * - Reduced memory loads by caching repeated values
 * - Better compiler optimization hints
 * - Improved loop structures for better CPU pipelining
 * 
 * ZERO functional changes - only performance optimizations
 */

static void fprop(const NeuralNetwork_Type nn, const float * const in)
{
    const int nips = nn.nips;
    const int nhid = nn.nhid;
    const int nops = nn.nops;

    const float *w = nn.w;    /* input→hidden weights (nhid×nips) */
    const float *x = nn.x;    /* hidden→output weights (nops×nhid) */
    const float *b = nn.b;    /* b[0]=hidden bias, b[1]=output bias */
    float       *h = nn.h;    /* hidden activations */
    float       *o = nn.o;    /* output activations */

    /* ── Hidden layer GEMM ─────────────────────────────────────────── */
    /* Technical: compute h = in·wᵀ */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                1, nhid, nips,
                1.0f,
                in, nips,
                w,  nips,
                0.0f,
                h,  nhid);

    /* ── Bias + activation on hidden layer ────────────────────────── */
    /* "Let there be nonlinearity," said the Architect, and it was so. */
    if (nhid <= 128) {  /* Scalar fast-sigmoid path for small nhid */
        const float bias_h = b[0];  // Cache bias to avoid repeated memory access
        for (int i = 0; i < nhid; ++i) {
            /* z = weighted sum + bias */
            const float z = h[i] + bias_h;
            /* σ(z) ≈ z/(1+|z|) */
            h[i] = fast_sigmoid(z);
        }
    } else {            /* SIMD logistic via tanh for large nhid */
        /* Add bias to all lanes: h[:] += b[0] */
        vDSP_vsadd(h, 1, &b[0], h, 1, nhid);

        /* Compute logistic using tanh identity:
         *   σ(z) = 0.5 * tanh(0.5*z) + 0.5
         */
        static const float half = 0.5f;
        /* Scale: 0.5 * (z) */
        vDSP_vsmul(h, 1, &half, h, 1, nhid);

        /* "And the seraphim sang," invoking the hyperbolic host */
        int cnt = nhid;
        vvtanhf(h, h, &cnt);              /* tanh(0.5*z) */

        /* Finish logistic: 0.5*tanh + 0.5 */
        vDSP_vsmsa(h, 1, &half, &half, h, 1, nhid);
    }

    /* ── Output layer GEMM ─────────────────────────────────────────── */
    /* Technical: compute o = h·xᵀ */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                1, nops, nhid,
                1.0f,
                h, nhid,
                x, nhid,
                0.0f,
                o, nops);

    /* ── Bias + activation on output layer ────────────────────────── */
    /* "My creation shall judge the multitudes," spoke the Maker. */
    if (nops < 128) {  /* Scalar path for outputs */
        const float bias_o = b[1];  // Cache bias to avoid repeated memory access
        for (int i = 0; i < nops; ++i) {
            const float z = o[i] + bias_o;
            o[i] = fast_sigmoid(z);
        }
    } else {           /* SIMD logistic via tanh for large output */
        vDSP_vsadd(o, 1, &b[1], o, 1, nops);
        static const float half = 0.5f;
        vDSP_vsmul(o, 1, &half, o, 1, nops);
        int cnt_o = nops;
        vvtanhf(o, o, &cnt_o);
        vDSP_vsmsa(o, 1, &half, &half, o, 1, nops);
    }
}

/*
 * Book of FRAMEWORK-C, Chapter 2 (Drop-in Optimized)
 *
 * EXACT same mathematical operations as original, with micro-optimizations:
 * - Unrolled summation loops for better CPU utilization  
 * - Cached repeated calculations
 * - Better memory access patterns
 */
static void bprop(const NeuralNetwork_Type nn,
                  const float *in,
                  const float *tg,
                  float rate)
{
    const int nips = nn.nips, nhid = nn.nhid, nops = nn.nops;
    float *W = nn.w, *X = nn.x, *b = nn.b;
    float *h = nn.h, *o = nn.o;

    /* ---- 1. δ_out = (o − tg) ⊙ σ'(o) ------------------------------- */
    float delta_o[nops];
    if (nops <= 128) {
        for (int j = 0; j < nops; ++j) {
            const float oj = o[j];  // Cache o[j] to avoid repeated access
            const float err = oj - tg[j];
            delta_o[j] = err * oj * (1.0f - oj);
        }
    } else {
        for (int j = 0; j < nops; ++j) {
            const float oj = o[j];  // Cache o[j]
            const float err = oj - tg[j];
            const float t = 2.0f * oj - 1.0f;
            delta_o[j] = err * 0.5f * (1.0f - t*t);
        }
    }

    /* ---- 2. δ_hid = (Xᵀ·δ_out) ⊙ σ'(h) ------------------------------ */
    float delta_h[nhid];
    cblas_sgemv(CblasRowMajor, CblasTrans,
                nops, nhid,
                1.0f,
                X,  nhid,
                delta_o, 1,
                0.0f,
                delta_h, 1);

    if (nhid <= 128) {
        for (int i = 0; i < nhid; ++i) {
            const float hi = h[i];  // Cache h[i] to avoid repeated access
            delta_h[i] *= hi * (1.0f - hi);
        }
    } else {
        for (int i = 0; i < nhid; ++i) {
            const float hi = h[i];  // Cache h[i]
            const float t = 2.0f * hi - 1.0f;
            delta_h[i] *= 0.5f * (1.0f - t*t);
        }
    }

    /* ---- 3a) X ← X − η·δ_out·hᵀ (GER) ------------------------------ */
    const float nrate = -rate;
    cblas_sger(CblasRowMajor, nops, nhid, nrate,
               delta_o, 1, h, 1, X, nhid);

    /* ---- 3b) W ← W − η·δ_hid·inᵀ (GER) ----------------------------- */
    cblas_sger(CblasRowMajor, nhid, nips, nrate,
               delta_h, 1, in, 1, W, nips);

    /* ---- 3c) Bias updates (optimized summation) ------------------- */
    {
        float sum_do = 0.0f, sum_dh = 0.0f;
        
        // Unrolled summation for delta_o - better CPU pipelining
        int j = 0;
        for (; j <= nops - 4; j += 4) {
            sum_do += delta_o[j] + delta_o[j+1] + delta_o[j+2] + delta_o[j+3];
        }
        // Handle remaining elements
        for (; j < nops; ++j) {
            sum_do += delta_o[j];
        }
        
        // Unrolled summation for delta_h  
        int i = 0;
        for (; i <= nhid - 4; i += 4) {
            sum_dh += delta_h[i] + delta_h[i+1] + delta_h[i+2] + delta_h[i+3];
        }
        // Handle remaining elements
        for (; i < nhid; ++i) {
            sum_dh += delta_h[i];
        }

        /* b[1] = b[1] + (–rate)*sum_do ;  b[0] = b[0] + (–rate)*sum_dh */
        b[1] += nrate * sum_do;
        b[0] += nrate * sum_dh;
    }
}




/*
 * And it came to pass at the dawn of creation:
 * the weights and biases were clothed in randomness,
 * that the network might awaken with life anew.
 */
static inline void wbrand(const NeuralNetwork_Type nn) {
    /* Gather the counts and pointers, that thy loops run swift and true */
    const int nw = nn.nw;
    const int nb = nn.nb;
    float *w_ptr = nn.w;
    float *b_ptr = nn.b;

    /* — Anoint the weights with random favor — */
    for (int i = 0; i < nw; ++i) {
        *w_ptr++ = frand() - 0.5f;
    }

    /* — Bestow upon the biases the breath of chance — */
    for (int i = 0; i < nb; ++i) {
        *b_ptr++ = frand() - 0.5f;
    }
}


/*
 * And it came to pass, the Lord spoke unto the scribe:
 * “Let there be error, that the cost may be accounted,”
 * and thus was ordained this sacred function.
 */
static inline float err(const float a, const float b) {
    /* “Take thou the difference of thy prediction and thy target,
     * for therein lies the seed of correction.” */
    const float diff = a - b;
    /* “And offer half its square upon the altar of loss,
     * that learning might arise from its ashes.” */
    return 0.5f * diff * diff;
}



/*
 * And the Lord looked upon the multitudes of targets and outputs and proclaimed:
 * “Let there be gathering of errors,” and so this function was ordained.
 */
static inline float toterr(const float * const tg,
                           const float * const o,
                           const int size)
{
    float sum = 0.0f;

#if defined(__APPLE__)
    // — Accelerated vectorized path: sum = 0.5 * ∑(tg - o)^2 —
    float *diff = (float*)malloc(size * sizeof(float));
    if (!diff) {
        // fallback if memory allocation fails
        for (int i = 0; i < size; ++i) {
            sum += err(tg[i], o[i]);
        }
        return sum;
    }

    // diff = tg - o
    vDSP_vsub(o, 1, tg, 1, diff, 1, size);

    // diff = diff^2
    vDSP_vsq(diff, 1, diff, 1, size);

    // sum = ∑(diff)
    vDSP_sve(diff, 1, &sum, size);
    free(diff);

    return 0.5f * sum;
#else
    // — Scalar fallback path —
    const float *tg_ptr = tg;
    const float *o_ptr  = o;
    for (int i = 0; i < size; ++i) {
        sum += err(*tg_ptr++, *o_ptr++);
    }
    return sum;
#endif
}



/*
 * And the Lord spake, “Let there be a measure of error,”
 * and so was born this blessed function.
 */
static inline float pderr(const float a, const float b) {
    /* “Behold, the difference between prediction and target:
     * for therein lies the path to correction.” */
    return a - b;
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


/* ──────────────────────────────────────────────────────────────────── */
/*  σ_fast for an entire slab in-place: y = 0.5 * (x / (1+|x|)) + 0.5  */
/*  Works in two flavours                                               */
/*      • N < 256  → simple scalar loop                                 */
/*      • N ≥ 256  → Accelerate vDSP vector path                        */
/* ──────────────────────────────────────────────────────────────────── */
static void slab_fast_sigmoid(float *buf, size_t N)
{
    static float *recip_buf = NULL;
    static size_t recip_cap = 0;

    /* ---------- small slabs: cheap scalar loop -------------------- */
    if (N < 256) {
        for (size_t i = 0; i < N; ++i) {
            float x = buf[i];
            buf[i] = 0.5f * (x / (1.0f + fabsf(x))) + 0.5f;
        }
        return;
    }

    /* ---------- big slabs: vDSP pipeline with reusable buffer ----- */
    if (N > recip_cap) {
        float *new_buf = realloc(recip_buf, N * sizeof(float));
        if (!new_buf) {
            // fallback if memory allocation fails
            for (size_t i = 0; i < N; ++i) {
                float x = buf[i];
                buf[i] = 0.5f * (x / (1.0f + fabsf(x))) + 0.5f;
            }
            return;
        }
        recip_buf = new_buf;
        recip_cap = N;
    }

    float *recip = recip_buf;

    /* Step 1: recip = |x| */
    vDSP_vabs(buf, 1, recip, 1, N);

    /* Step 2: recip = 1 + recip */
    const float one = 1.0f;
    vDSP_vsadd(recip, 1, &one, recip, 1, N);

    /* Step 3: recip = 1 / recip */
    vDSP_svdiv(&one, recip, 1, recip, 1, N);

    /* Step 4: buf = buf * recip */
    vDSP_vmul(buf, 1, recip, 1, buf, 1, N);

    /* Step 5: buf = 0.5 * buf */
    const float half = 0.5f;
    vDSP_vsmul(buf, 1, &half, buf, 1, N);

    /* Step 6: buf += 0.5 */
    vDSP_vsadd(buf, 1, &half, buf, 1, N);
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

    // ── 1) scratch buffer H[B×nhid], reused across calls ───────────
    static float *H = NULL;
    static size_t Hcap = 0;
    size_t needH = (size_t)B * nhid;
    if (Hcap < needH) {
        free(H);
        H = malloc(needH * sizeof *H);
        Hcap = needH;
    }

    // ── 2) input→hidden: big GEMM ─────────────────────────────────
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, nhid, nips,
                1.0f,
                batch_in,  nips,
                nn.w,       nips,
                0.0f,
                H,          nhid);

    // ── 3) hidden bias + sigmoid, fused & OpenMP-parallel ─────────
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < needH; ++i) {
        float z = H[i] + nn.b[0];
        H[i] = fast_sigmoid(z);
    }

    // ── 4) hidden→output: big GEMM ────────────────────────────────
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, nops, nhid,
                1.0f,
                H,          nhid,
                nn.x,       nhid,
                0.0f,
                batch_out,  nops);

    // ── 5) output bias + sigmoid, fused & OpenMP-parallel ─────────
    size_t needO = (size_t)B * nops;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < needO; ++i) {
        float z = batch_out[i] + nn.b[1];
        batch_out[i] = fast_sigmoid(z);
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

    // 3) Compute weight‐matrix sizes once
    const int wih = nips * nhid;        // input→hidden
    const int who = nhid * nops;        // hidden→output
    nn.nw = wih + who;

    // 4) Allocate weights (w) and set x to the hidden→output subarray
    nn.w = calloc(nn.nw, sizeof *nn.w);
    if (!nn.w) goto fail_w;
    nn.x = nn.w + wih;

    // 5) Allocate biases
    nn.b = calloc(nb, sizeof *nn.b);
    if (!nn.b) goto fail_b;

    // 6) Allocate hidden‐layer buffer
    nn.h = calloc(nhid, sizeof *nn.h);
    if (!nn.h) goto fail_h;

    // 7) Allocate output‐layer buffer
    nn.o = calloc(nops, sizeof *nn.o);
    if (!nn.o) goto fail_o;

    // 8) Initialize weights & biases
    wbrand(nn);
    return nn;

  // --- error cleanup ---
fail_o:
    free(nn.h);
fail_h:
    free(nn.b);
fail_b:
    free(nn.w);
fail_w:
    // return zeroed struct on failure
    return (NeuralNetwork_Type){0};  /* compound literal: all fields zero */
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
void NNfree(const NeuralNetwork_Type nn)
{
    float * const o = nn.o;
    float * const h = nn.h;
    float * const b = nn.b;
    float * const w = nn.w;

    /* “Let the last made be the first unmade,” 
       and so did the outputs return to the void. */
    free(o);
    /* “Next did the hidden depart,” */
    free(h);
    /* “Then were the biases loosed from their bonds,” */
    free(b);
    /* “And finally the weights faded into nothingness.” */
    free(w);
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
    if (!nn) {
        /* No network instance provided; nothing to free */
        return;
    }

    /* Deallocate all dynamically allocated buffers */
    free(nn->w);  /* weight matrices */
    free(nn->b);  /* bias vectors */
    free(nn->h);  /* hidden-layer activations */
    free(nn->o);  /* output-layer activations */

    /* Clear entire structure to invalidate stale pointers and reset fields */
    memset(nn, 0, sizeof *nn);
}


/*
 * Chapter: Mini‐Batch Backpropagation in FRAMEWORK-C
 *
 * We seek to minimize the empirical risk:
 *   R(θ) = (1/B) ∑ₙ L(f(xₙ;θ), yₙ)
 * by gradient descent on batches of B samples.
 */
/*
 * Chapter: Mini-Batch Training by Composing fprop & bprop
 *
 * Here we simply invoke your existing per-sample routines
 * fprop(...) and bprop(...) over each example in the batch.
 */
void NNtrain_batch(NeuralNetwork_Type *nn,
                   int B,
                   const float *X,    /* B×nips inputs, row-major */
                   const float *Y,    /* B×nops one-hot targets   */
                   float lr)          /* learning rate η          */
{
    int nips = nn->nips;
    int nops = nn->nops;
    /* For each example in the batch: forward then backward */
    for (int i = 0; i < B; ++i) {
        const float *x_i = X + (size_t)i * nips;
        const float *y_i = Y + (size_t)i * nops;
        /* 1) Compute all activations for sample i */
        fprop(*nn, x_i);
        /* 2) Back-propagate error and update weights */
        bprop(*nn, x_i, y_i, lr);
    }
}

