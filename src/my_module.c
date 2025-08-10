/*
 *  src/my_module.c  –  Python bridge for FRAMEWORK-C
 *  Auto-depth selection is decided INSIDE this module.
 */
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>    /* memcpy */
#include "nn.h"

/* ───────────────────────────── Auto-depth thresholds ───────────────────────── */
#ifndef FWC_AUTO_N_SMALL
#define FWC_AUTO_N_SMALL  10000   /* <10k → 1 layer */
#endif
#ifndef FWC_AUTO_N_MED
#define FWC_AUTO_N_MED    50000   /* 10k–50k → 2 layers (smaller h2) */
#endif

/* Try to use NNbuild2 if it exists at link time; otherwise this will be NULL. */
#if defined(__GNUC__) || defined(__clang__)
extern NeuralNetwork_Type NNbuild2(int nips, int nhid, int nhid2, int nops)
    __attribute__((weak));
#else
/* Non-GNU/Clang: declare and assume unavailable (will fallback to NNbuild). */
extern NeuralNetwork_Type NNbuild2(int nips, int nhid, int nhid2, int nops);
#pragma message("Warning: weak reference to NNbuild2 not supported; using NNbuild fallback.")
#endif

/* ─────────────────────────── Internal wrapper handle ───────────────────────── */
typedef struct {
    int nips, nhid, nops;
    unsigned seed;
    int is_built;                /* 0 until first materialization */
    NeuralNetwork_Type net;      /* valid only when is_built == 1 */
} FC_Handle;

/* ─────────────────────────── Capsule helpers & dtor ────────────────────────── */
static void capsule_destruct(PyObject *capsule)
{
    if (!PyCapsule_IsValid(capsule, "frameworkc.nn"))
        return;

    FC_Handle *h = (FC_Handle*)PyCapsule_GetPointer(capsule, "frameworkc.nn");
    if (!h) return;

    if (h->is_built) {
        NNdestroy(&h->net);   /* frees internal buffers */
    }
    free(h);
    (void)PyCapsule_SetPointer(capsule, NULL);
}

/* Build the actual C net inside the handle, if not yet built.
   maybe_N: pass B from train_batch as a proxy for dataset size; 0 if unknown. */
static int ensure_built(FC_Handle *h, long long maybe_N)
{
    if (h->is_built) return 1;

    /* Seed RNG (used by wbrand inside NNbuild) */
    srand(h->seed);

    /* Decide architecture:
       If NNbuild2 exists and N is "large", use 2 hidden layers with nhid2 heuristic. */
    int use_two = 0;
    int nhid2   = 0;

    if (maybe_N >= FWC_AUTO_N_SMALL && &NNbuild2) {
        use_two = 1;
        if (maybe_N < FWC_AUTO_N_MED) nhid2 = (h->nhid > 1 ? h->nhid/2 : h->nhid);
        else                          nhid2 = h->nhid;
    }

    if (use_two) {
        /* Attempt 2-layer build */
        h->net = NNbuild2(h->nips, h->nhid, nhid2, h->nops);
        if (h->net.nw == 0) { /* fallback to 1-layer if allocation failed */
            h->net = NNbuild(h->nips, h->nhid, h->nops);
        }
    } else {
        /* 1-layer build */
        h->net = NNbuild(h->nips, h->nhid, h->nops);
    }

    if (h->net.nw == 0) return 0;
    h->is_built = 1;
    return 1;
}

/* ───────────────────────────────── py_build ────────────────────────────────── */
/* build(nips, nhid, nops [, seed]) → capsule
 * NOTE: does NOT immediately allocate the heavy network;
 *       we lazily materialize on the first training/predict call.
 */
static PyObject *py_build(PyObject *self, PyObject *args)
{
    int nips, nhid, nops, seed = 0;
    if (!PyArg_ParseTuple(args, "iii|i", &nips, &nhid, &nops, &seed))
        return NULL;

    FC_Handle *h = (FC_Handle*)calloc(1, sizeof *h);
    if (!h) return PyErr_NoMemory();

    h->nips = nips; h->nhid = nhid; h->nops = nops;
    h->seed = (unsigned)seed;
    h->is_built = 0;

    return PyCapsule_New(h, "frameworkc.nn", capsule_destruct);
}

/* ───────────────────────────── forward decls ──────────────────────────────── */
static PyObject *py_predict(PyObject *self, PyObject *args);
static PyObject *py_predict_batch_fast(PyObject *self, PyObject *args);
static PyObject *py_train_one(PyObject *self, PyObject *args);
static PyObject *py_train_batch(PyObject *self, PyObject *args);

/* ─────────────────────────── Method table & init ───────────────────────────── */
static PyMethodDef Methods[] = {
    {"build",         py_build,              METH_VARARGS,
     "build(nips, nhid, nops [, seed]) -> net_handle"},
    {"predict",       py_predict,            METH_VARARGS,
     "predict(net, 1-D float32) -> float32[nops]"},
    {"predict_batch", py_predict_batch_fast, METH_VARARGS,
     "predict_batch(net, float32[B,nips]) -> float32[B,nops]"},
    {"train_one",     py_train_one,          METH_VARARGS,
     "train_one(net, x[nips], t[nops], lr) -> float loss"},
    {"train_batch",   py_train_batch,        METH_VARARGS,
     "train_batch(net, X[B,nips], Y[B,nops], lr) -> None"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mod = {
    PyModuleDef_HEAD_INIT,
    "frameworkc",
    "Pure-C neural network (auto-depth decided in module)",
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_frameworkc(void)
{
    import_array();  /* NumPy C-API init */
    return PyModule_Create(&mod);
}

/* ────────────────────────────── predict (1D) ──────────────────────────────── */
static PyObject *py_predict(PyObject *self, PyObject *args)
{
    PyObject *capsule, *obj;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &obj)) return NULL;

    FC_Handle *h = (FC_Handle*)PyCapsule_GetPointer(capsule, "frameworkc.nn");
    if (!h) return NULL;

    /* If predicting before any training, build a 1-layer net by default. */
    if (!ensure_built(h, /*maybe_N=*/0)) return PyErr_NoMemory();

    NeuralNetwork_Type *nn = &h->net;

    PyArrayObject *x_arr = (PyArrayObject*)
        PyArray_FROM_OTF(obj, NPY_FLOAT32,
                         NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!x_arr) return NULL;

    if (PyArray_NDIM(x_arr) != 1 || PyArray_DIM(x_arr, 0) != nn->nips) {
        Py_DECREF(x_arr);
        PyErr_SetString(PyExc_ValueError, "input must be shape [nips] float32");
        return NULL;
    }

    npy_intp odims[1] = { nn->nops };
    PyArrayObject *out = (PyArrayObject*)
        PyArray_SimpleNew(1, odims, NPY_FLOAT32);
    if (!out) { Py_DECREF(x_arr); return NULL; }

    float *xin  = (float*)PyArray_DATA(x_arr);
    float *yout = (float*)PyArray_DATA(out);

    Py_BEGIN_ALLOW_THREADS
    float *o = NNpredict(*nn, xin);
    memcpy(yout, o, (size_t)nn->nops * sizeof(float));
    Py_END_ALLOW_THREADS

    Py_DECREF(x_arr);
    return (PyObject*)out;
}

/* ───────────────────────────── predict_batch ──────────────────────────────── */
/* signature: predict_batch(net_capsule, numpy_in[B,nips]) → numpy_out[B,nops] */
static PyObject *py_predict_batch_fast(PyObject *self, PyObject *args)
{
    PyObject *capsule, *in_obj;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &in_obj)) return NULL;

    FC_Handle *h = (FC_Handle*)PyCapsule_GetPointer(capsule, "frameworkc.nn");
    if (!h) return NULL;

    /* Predict before training: build a 1-layer net by default. */
    if (!ensure_built(h, /*maybe_N=*/0)) return PyErr_NoMemory();

    NeuralNetwork_Type *net = &h->net;

    PyArrayObject *in_arr = (PyArrayObject*)
        PyArray_FROM_OTF(in_obj, NPY_FLOAT32,
                         NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!in_arr) return NULL;

    if (PyArray_NDIM(in_arr) != 2 || PyArray_DIM(in_arr, 1) != net->nips) {
        Py_DECREF(in_arr);
        PyErr_Format(PyExc_ValueError, "input must be float32[B,%d]", net->nips);
        return NULL;
    }

    const int B = (int)PyArray_DIM(in_arr, 0);

    npy_intp out_dims[2] = { B, net->nops };
    PyArrayObject *out_arr = (PyArrayObject*)
        PyArray_SimpleNew(2, out_dims, NPY_FLOAT32);
    if (!out_arr) { Py_DECREF(in_arr); return NULL; }

    float *inp  = (float*)PyArray_DATA(in_arr);
    float *outp = (float*)PyArray_DATA(out_arr);

    Py_BEGIN_ALLOW_THREADS
    NNpredict_batch(*net, inp, B, outp);
    Py_END_ALLOW_THREADS

    Py_DECREF(in_arr);
    return (PyObject*)out_arr;
}

/* ────────────────────────────── train_one ─────────────────────────────────── */
static PyObject *py_train_one(PyObject *self, PyObject *args)
{
    PyObject *capsule, *x_obj, *t_obj;
    double lr;
    if (!PyArg_ParseTuple(args, "OOOd", &capsule, &x_obj, &t_obj, &lr))
        return NULL;

    FC_Handle *h = (FC_Handle*)PyCapsule_GetPointer(capsule, "frameworkc.nn");
    if (!h) return NULL;

    /* train_one lacks dataset size; treat as "small" → 1-layer default. */
    if (!ensure_built(h, /*maybe_N=*/0)) return PyErr_NoMemory();

    NeuralNetwork_Type *nn = &h->net;

    PyArrayObject *x_arr = (PyArrayObject*)
        PyArray_FROM_OTF(x_obj, NPY_FLOAT32,
                         NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyArrayObject *t_arr = (PyArrayObject*)
        PyArray_FROM_OTF(t_obj, NPY_FLOAT32,
                         NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!x_arr || !t_arr) { Py_XDECREF(x_arr); Py_XDECREF(t_arr); return NULL; }

    if (PyArray_NDIM(x_arr)!=1 || PyArray_DIM(x_arr,0)!=nn->nips ||
        PyArray_NDIM(t_arr)!=1 || PyArray_DIM(t_arr,0)!=nn->nops) {
        Py_DECREF(x_arr); Py_DECREF(t_arr);
        PyErr_SetString(PyExc_ValueError, "x,t must be float32 [nips] and [nops]");
        return NULL;
    }

    float *xin = (float*)PyArray_DATA(x_arr);
    float *tgt = (float*)PyArray_DATA(t_arr);

    float loss;
    Py_BEGIN_ALLOW_THREADS
    loss = NNtrain(*nn, xin, tgt, (float)lr);
    Py_END_ALLOW_THREADS

    Py_DECREF(x_arr); Py_DECREF(t_arr);
    return PyFloat_FromDouble((double)loss);
}

/* ────────────────────────────── train_batch ───────────────────────────────── */
static PyObject *py_train_batch(PyObject *self, PyObject *args)
{
    PyObject *capsule;
    PyArrayObject *x_arr, *t_arr;
    double lr_double;

    if (!PyArg_ParseTuple(args, "OO!O!d",
                          &capsule,
                          &PyArray_Type, &x_arr,
                          &PyArray_Type, &t_arr,
                          &lr_double))
        return NULL;

    FC_Handle *h = (FC_Handle*)PyCapsule_GetPointer(capsule, "frameworkc.nn");
    if (!h) return NULL;

    /* Require float32, 2-D, C-contiguous for both arrays */
    if (PyArray_TYPE(x_arr) != NPY_FLOAT32 || PyArray_NDIM(x_arr) != 2 ||
        !PyArray_IS_C_CONTIGUOUS(x_arr) ||
        PyArray_TYPE(t_arr) != NPY_FLOAT32 || PyArray_NDIM(t_arr) != 2 ||
        !PyArray_IS_C_CONTIGUOUS(t_arr))
    {
        PyErr_SetString(PyExc_TypeError,
                        "train_batch requires C-contiguous float32 2-D arrays");
        return NULL;
    }

    const int B    = (int)PyArray_DIM(x_arr, 0);
    const int nips = (int)PyArray_DIM(x_arr, 1);
    const int nops = (int)PyArray_DIM(t_arr, 1);

    if (nips != h->nips || nops != h->nops) {
        PyErr_Format(PyExc_ValueError,
                     "expected shapes [*,%d] and [*,%d]", h->nips, h->nops);
        return NULL;
    }
    if (PyArray_DIM(t_arr, 0) != B) {
        PyErr_SetString(PyExc_ValueError,
                        "X and Y must have the same first dimension (batch size)");
        return NULL;
    }

    /* Lazily build here, using B as a proxy for dataset size.
       If you pass the FULL dataset in one go, you'll get auto 2-layer where available. */
    if (!ensure_built(h, /*maybe_N=*/(long long)B)) return PyErr_NoMemory();

    NeuralNetwork_Type *net = &h->net;

    float *restrict X = (float*)PyArray_DATA(x_arr);
    float *restrict Y = (float*)PyArray_DATA(t_arr);
    float lr = (float)lr_double;

    Py_BEGIN_ALLOW_THREADS
    NNtrain_batch(net, B, X, Y, lr);
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}
