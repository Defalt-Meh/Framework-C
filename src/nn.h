#ifndef NN_H
#define NN_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>     // sysconf

typedef struct
{
    float *w;   /*All weights*/
    float *x;   /*hidden layer to output layer weights*/
    float *b;   /*Biases*/
    float *h;   /*Hidden layer*/ 
    float *o;   /*output layer*/
    int nb;     /*Number of Biases*/ 
    int nw;     /*Number of weights*/
    int nips;   /*Number of inputs*/
    int nhid;   /*Number of Hidden Neurons*/
    int nops;   /*Number of Outputs*/

}NeuralNetwork_Type;

/*Exposed Functions*/
float * NNpredict(const NeuralNetwork_Type nn, const float * in);
NeuralNetwork_Type NNbuild (int nips, int nhid, int nops);
float NNtrain(const NeuralNetwork_Type nn, const float * in, const float * tg, float rate);
void NNsave(const NeuralNetwork_Type nn, const char * path);
NeuralNetwork_Type NNload(const char * path);
void NNprint (const float * arr, const int size);
void NNfree(const NeuralNetwork_Type nn);
void NNdestroy(NeuralNetwork_Type *nn);
void NNpredict_batch(const NeuralNetwork_Type nn,const float *batch_in,int B,float *batch_out);/* Inference for an entire mini-batch (B samples) in one call.
                                                                                                        batch_in  : row-major  B × nips   (float32)
                                                                                                        batch_out : row-major  B × nops   (float32) – caller allocates
                                                                                                */

#endif /* NN_H */