#ifndef NN_H
#define NN_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>     // sysconf
#include <float.h>

/* Define FWC_CACHE_Z to store hidden-layer pre-activations (z) for exact grads.
   This does not change external APIs; only allocates an extra buffer when enabled. */
/* #define FWC_CACHE_Z 1 */

typedef struct
{
    float *w;   /* All weights: input->hidden (nhid×nips) */
    float *x;   /* Hidden->output weights (nops×nhid) */
    float *b;   /* Biases (scalar per layer): b[0]=hidden, b[1]=output */
    float *h;   /* Hidden activations (nhid) */
    float *o;   /* Output activations (nops) */
#if defined(FWC_CACHE_Z)
    float *hz;  /* Hidden pre-activations z (nhid) — filled in fprop, used in bprop */
    /* For softmax outputs, oz is not needed. Uncomment only for binary-sigmoid output:
       float *oz;  // Output pre-activations z (nops)
    */
#endif
    int nb;     /* Number of biases (kept for compatibility) */
    int nw;     /* Number of weights (w + x total if you track) */
    int nips;   /* Number of inputs */
    int nhid;   /* Number of hidden neurons */
    int nops;   /* Number of outputs */
} NeuralNetwork_Type;

/* Exposed Functions */
float * NNpredict(const NeuralNetwork_Type nn, const float * in);
NeuralNetwork_Type NNbuild (int nips, int nhid, int nops);
float NNtrain(const NeuralNetwork_Type nn, const float * in, const float * tg, float rate);
void NNsave(const NeuralNetwork_Type nn, const char * path);
NeuralNetwork_Type NNload(const char * path);
void NNprint (const float * arr, const int size);
void NNfree(const NeuralNetwork_Type nn);
void NNdestroy(NeuralNetwork_Type *nn);

void NNpredict_batch(const NeuralNetwork_Type nn,
                     const float *batch_in, int B,
                     float *batch_out); /* Inference for a mini-batch (B×nips -> B×nops) */

void NNtrain_batch(NeuralNetwork_Type *nn,
                   int B,
                   const float *X,    /* B×nips */
                   const float *Y,    /* B×nops */
                   float lr);

#endif /* NN_H */
