#include "nn.h"
#include <stdlib.h>
#include <math.h>          /* fabsf */

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
 * And the Lord spoke, “Let there be fprop,”
 * and lo, the hidden and output layers were awakened.
 */
static void fprop(const NeuralNetwork_Type nn, const float * const in)
{
    const int nips = nn.nips;
    const int nhid = nn.nhid;
    const int nops = nn.nops;

    const float *w = nn.w;    /* input→hidden weights (nhid×nips) */
    const float *x = nn.x;    /* hidden→output weights (nops×nhid) */
    const float *b = nn.b;    /* biases */
    float       *h = nn.h;    /* hidden activations */
    float       *o = nn.o;    /* output activations */

    /* ── Hidden layer GEMM ─────────────────────────────────────────── */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                1, nhid, nips,
                1.0f,
                in, nips,
                w,  nips,
                0.0f,
                h,  nhid);

    /* ── Bias + activation on hidden layer ────────────────────────── */
    if (nhid < 128) {                        /* scalar fast path */
        for (int i = 0; i < nhid; ++i) {
            float z = h[i] + b[0];
            h[i] = fast_sigmoid(z);
        }
    } else {                                 /* SIMD tanh path   */
        vDSP_vsadd(h, 1, &b[0], h, 1, nhid);       /* h += b0 */
        static const float half = 0.5f;
        vDSP_vsmul(h, 1, &half, h, 1, nhid);       /* 0.5*h */
        int cnt = nhid;
        vvtanhf(h, h, &cnt);                        /* tanh */
        vDSP_vsmsa(h, 1, &half, &half, h, 1, nhid); /* 0.5*tanh+0.5 */
    }

    /* ── Output layer GEMM ─────────────────────────────────────────── */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                1, nops, nhid,
                1.0f,
                h, nhid,
                x, nhid,
                0.0f,
                o, nops);

    /* ── Bias + activation on output layer ────────────────────────── */
    if (nops < 128) {                         /* scalar fast path */
        for (int i = 0; i < nops; ++i) { 
            float z = o[i] + b[1];
            o[i] = fast_sigmoid(z);
        }
    } else {                                  /* SIMD tanh path   */
        vDSP_vsadd(o, 1, &b[1], o, 1, nops);
        static const float half = 0.5f;
        vDSP_vsmul(o, 1, &half, o, 1, nops);
        int cnt = nops;
        vvtanhf(o, o, &cnt);
        vDSP_vsmsa(o, 1, &half, &half, o, 1, nops);
    }
}


/* ------------------------------------------------------------------ */
/*  Back-propagation  (vectorised, BLAS + vDSP)                       */
/* ------------------------------------------------------------------ */
static void bprop(const NeuralNetwork_Type nn,
                  const float *in,         /* 1×nips input sample        */
                  const float *tg,         /* 1×nops target              */
                  float rate)              /* learning-rate              */
{
    const int nips = nn.nips;
    const int nhid = nn.nhid;
    const int nops = nn.nops;

    float *W = nn.w;        /* input → hidden  (nhid×nips, row-major) */
    float *X = nn.x;        /* hidden → output (nops×nhid, row-major) */
    float *b = nn.b;        /* biases: b[0]=hidden, b[1]=output       */
    float *h = nn.h;        /* hidden activations                     */
    float *o = nn.o;        /* output activations                     */

    /* ---- 1.  δ_out  =  (o − t) ⊙ σ'(o)  ------------------------- */
    float delta_o[nops];
    for (int j = 0; j < nops; ++j) {
        float err = o[j] - tg[j];            /* o − t                     */
        delta_o[j] = err * o[j] * (1.0f - o[j]);   /* * σ'(o)               */
    }

    /* ---- 2.  δ_hid  =  (Xᵀ · δ_out) ⊙ σ'(h)  -------------------- */
    float delta_h[nhid];
    /* tmp = Xᵀ · δ_out   (BLAS GEMV) */
    cblas_sgemv(CblasRowMajor, CblasTrans,
                nops, nhid,
                1.0f,
                X,  nhid,
                delta_o, 1,
                0.0f,
                delta_h, 1);

    /* element-wise multiply by σ'(h) */
    for (int i = 0; i < nhid; ++i)
        delta_h[i] *= h[i] * (1.0f - h[i]);

    /* ---- 3.  Parameter updates  --------------------------------- */
    const float nrate = -rate;              /* BLAS “alpha” is *added*   */

    /* 3a)  X  ←  X  −  η · δ_out · hᵀ    (rank-1 update, BLAS GER) */
    cblas_sger(CblasRowMajor,
               nops, nhid,
               nrate,
               delta_o, 1,          /* δ_out               */
               h,       1,          /* hᵀ                  */
               X,       nhid);      /* weight matrix       */

    /* 3b)  W  ←  W  −  η · δ_hid · inᵀ   (another GER) */
    cblas_sger(CblasRowMajor,
               nhid, nips,
               nrate,
               delta_h, 1,          /* δ_hid               */
               in,      1,          /* inᵀ                 */
               W,       nips);

    /* 3c)  Biases:   b₂ -= η·δ_out ,   b₁ -= η·δ_hid          */
    vDSP_vsma  (delta_o, 1, &nrate, b + 1, 1, b + 1, 1, nops); /* output bias */
    vDSP_vsma  (delta_h, 1, &nrate, b + 0, 1, b + 0, 1, nhid); /* hidden bias */
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

    /* 1) allocate one hidden-slab on stack (B×nhid) */
    float *H = malloc((size_t)B * nhid * sizeof(float));
    if (!H) { fprintf(stderr,"OOM in NNpredict_batch\n"); return; }

    /* 2) GEMM:  [B×nips] · [nhid×nips]^T  →  H [B×nhid] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, nhid, nips,
                1.0f,
                batch_in,  nips,
                nn.w,      nips,
                0.0f,
                H,         nhid);

    /* 3) add hidden bias & fast-sigmoid on the whole slab */
    for (int i = 0; i < nhid; ++i)
        vDSP_vsadd(H + i, nhid, &nn.b[0], H + i, nhid, B); /* bias col-wise */

    slab_fast_sigmoid(H, (size_t)B * nhid);

    /* 4) GEMM:  [B×nhid] · [nops×nhid]^T  →  batch_out [B×nops] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, nops, nhid,
                1.0f,
                H,        nhid,
                nn.x,     nhid,
                0.0f,
                batch_out,nops);

    /* 5) add output bias & sigmoid */
    for (int i = 0; i < nops; ++i)
        vDSP_vsadd(batch_out + i, nops, &nn.b[1],
                   batch_out + i, nops, B);

    slab_fast_sigmoid(batch_out, (size_t)B * nops);

    free(H);
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
