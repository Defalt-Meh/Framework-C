/*
 *  src/my_module.c  –  Python bridge for FRAMEWORK-C
 */
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "nn.h"

/* ------------------------------------------------------------------ */
/*  PyCapsule finaliser – releases both the struct’s buffers          */
/*  *and* the malloc’ed struct itself                                 */
/* ------------------------------------------------------------------ */
/*------------------------------------------------------------------*/
/*  Capsule helpers                                                 */
/*------------------------------------------------------------------*/
static void capsule_destruct(PyObject *capsule)
{
    /* NULL on second entry ⇒ nothing happens. */
    NeuralNetwork_Type *p = PyCapsule_GetPointer(capsule, "frameworkc.nn");
    if (!p) return;

    NNdestroy(p);    /* frees w, b, h, o and zeroes the struct */
    free(p);         /* <- free the struct itself */

    /* mark the capsule as empty so another call is safe */
    PyCapsule_SetPointer(capsule, NULL);
}




/* ------------------------------------------------------------------ */
/*  build(nips, nhid, nops [, seed]) → capsule                         */
/* ------------------------------------------------------------------ */
static PyObject *py_build(PyObject *self, PyObject *args) {
    int nips, nhid, nops, seed = 0;
    if (!PyArg_ParseTuple(args, "iii|i", &nips, &nhid, &nops, &seed))
        return NULL;

    srand(seed);
    NeuralNetwork_Type *nn = malloc(sizeof *nn);
    if (!nn) return PyErr_NoMemory();

    *nn = NNbuild(nips, nhid, nops);
    if (nn->nw == 0) { free(nn); return PyErr_NoMemory(); }

    return PyCapsule_New(nn, "frameworkc.nn", capsule_destruct);
}

/* ------------------------------------------------------------------ */
/*  predict(capsule, seq) → list                                       */
/* ------------------------------------------------------------------ */
static PyObject *py_predict(PyObject *self, PyObject *args) {
    PyObject *capsule, *seq;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &seq)) return NULL;

    NeuralNetwork_Type *nn = PyCapsule_GetPointer(capsule, "frameworkc.nn");
    if (!nn) return NULL;

    PyObject *fast = PySequence_Fast(seq, "input must be a sequence");
    if (!fast) return NULL;

    if (PySequence_Fast_GET_SIZE(fast) != nn->nips) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "length mismatch");
        return NULL;
    }

    float *in = malloc(nn->nips * sizeof *in);
    if (!in) { Py_DECREF(fast); return PyErr_NoMemory(); }

    for (Py_ssize_t i = 0; i < nn->nips; ++i)
        in[i] = (float)PyFloat_AsDouble(PySequence_Fast_GET_ITEM(fast, i));

    Py_DECREF(fast);

    float *out = NNpredict(*nn, in);
    PyObject *py_out = PyList_New(nn->nops);
    for (int i = 0; i < nn->nops; ++i)
        PyList_SET_ITEM(py_out, i, PyFloat_FromDouble(out[i]));

    free(in);
    return py_out;
}

/* ------------------------------------------------------------------ */
/*  predict_batch(capsule, ndarray[B,nips]) → ndarray[B,nops]          */
/* ------------------------------------------------------------------ */
// static PyObject *py_predict_batch(PyObject *self, PyObject *args) {
//     PyObject *capsule; PyArrayObject *arr;
//     if (!PyArg_ParseTuple(args, "OO!", &capsule, &PyArray_Type, &arr))
//         return NULL;

//     NeuralNetwork_Type *nn = PyCapsule_GetPointer(capsule, "frameworkc.nn");
//     if (!nn) return NULL;

//     /* Validate dtype, ndim, shape */
//     if (PyArray_TYPE(arr) != NPY_FLOAT32 || PyArray_NDIM(arr) != 2) {
//         PyErr_SetString(PyExc_TypeError, "array must be float32[*, nips]");
//         return NULL;
//     }
//     if (PyArray_DIM(arr, 1) != nn->nips) {
//         PyErr_Format(PyExc_ValueError,
//                      "second dimension must be %d", nn->nips);
//         return NULL;
//     }

//     const npy_intp B = PyArray_DIM(arr, 0);
//     const float   *inp = (const float*)PyArray_DATA(arr);

//     npy_intp dims[2] = {B, nn->nops};
//     PyArrayObject *out =
//         (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
//     float *outp = (float*)PyArray_DATA(out);

//     for (npy_intp i = 0; i < B; ++i) {
//         const float *x = inp  + i * nn->nips;
//         float       *y = NNpredict(*nn, x);        /* nn->o */
//         memcpy(outp + i * nn->nops, y, nn->nops * sizeof(float));
//     }
//     return (PyObject*)out;
// }
/* forward-decl so the table knows this symbol exists */
static PyObject *py_predict_batch_fast(PyObject *self, PyObject *args);
static PyObject *py_train_one(PyObject*, PyObject*);      /* NEW */
/* ------------------------------------------------------------------ */
/*  Method table & module init                                         */
/* ------------------------------------------------------------------ */
static PyMethodDef Methods[] = {
    {"build",          py_build,                METH_VARARGS,
     "build(nips, nhid, nops [, seed]) -> net_handle"},
    {"predict",        py_predict,              METH_VARARGS,
     "predict(net, 1-D sequence) -> list"},
    {"predict_batch",  py_predict_batch_fast,   METH_VARARGS,
     "predict_batch(net, float32 array[B,nips]) -> float32 array[B,nops]"},
    {"train_one",      py_train_one,         METH_VARARGS,
    "train_one(net, x, t, lr) -> loss"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mod = {
    PyModuleDef_HEAD_INIT, "frameworkc",
    "Pure-C neural network", -1, Methods
};

PyMODINIT_FUNC PyInit_frameworkc(void) {
    import_array();                 /* NumPy C-API */
    return PyModule_Create(&mod);
}

/*  signature: predict_batch(net_capsule, numpy_in[B,nips]) → numpy_out[B,nops] */
static PyObject *py_predict_batch_fast(PyObject *self, PyObject *args)
{
    /* ------------------------------------------------------------------
       Parse arguments:  capsule  +  NumPy ndarray (float32, 2-D)
    ------------------------------------------------------------------ */
    PyObject      *capsule;
    PyArrayObject *in_arr;
    if (!PyArg_ParseTuple(args, "OO!", &capsule, &PyArray_Type, &in_arr))
        return NULL;

    /* ------------------------------------------------------------------
       Grab the C network; the tag **must** match the one used in py_build
       ("frameworkc.nn").  If it doesn’t, the API sets a Python error that
       we simply propagate upward by returning NULL.
    ------------------------------------------------------------------ */
    NeuralNetwork_Type *net =
        PyCapsule_GetPointer(capsule, "frameworkc.nn");
    if (!net) return NULL;                      /* bad capsule → Python error */

    /* ------------------------------------------------------------------
       Quick shape / dtype validation
    ------------------------------------------------------------------ */
    if (PyArray_TYPE(in_arr) != NPY_FLOAT32 || PyArray_NDIM(in_arr) != 2) {
        PyErr_SetString(PyExc_TypeError, "input must be float32[*, nips]");
        return NULL;
    }
    const int B = (int)PyArray_DIM(in_arr, 0);
    const int nips = (int)PyArray_DIM(in_arr, 1);
    if (nips != net->nips) {
        PyErr_Format(PyExc_ValueError, "second dimension must be %d", net->nips);
        return NULL;
    }

    /* ------------------------------------------------------------------
       Allocate output NumPy array B × nops (float32)
    ------------------------------------------------------------------ */
    npy_intp out_dims[2] = {B, net->nops};
    PyArrayObject *out_arr =
        (PyArrayObject*)PyArray_SimpleNew(2, out_dims, NPY_FLOAT32);
    if (!out_arr) return NULL;                  /* prop. NumPy OOM as Python */

    /* ------------------------------------------------------------------
       Call the high-performance C batching kernel
    ------------------------------------------------------------------ */
    NNpredict_batch(
        *net,                                   /* pass the struct itself   */
        (float*)PyArray_DATA(in_arr),
        B,
        (float*)PyArray_DATA(out_arr));

    return (PyObject*)out_arr;
}



/* ------------------------------------------------------------------ */
/*  train_one(net, x1D, t1D, lr)  →  float  (sample loss)              */
/* ------------------------------------------------------------------ */
static PyObject *py_train_one(PyObject *self, PyObject *args)
{
    PyObject *capsule, *x_seq, *t_seq;
    double    lr;
    if (!PyArg_ParseTuple(args, "OOOd", &capsule, &x_seq, &t_seq, &lr))
        return NULL;

    NeuralNetwork_Type *nn = PyCapsule_GetPointer(capsule, "frameworkc.nn");
    if (!nn) return NULL;

    /* --- convert python sequences to C float arrays ---------------- */
    PyObject *x_fast = PySequence_Fast(x_seq, "x must be a sequence");
    PyObject *t_fast = PySequence_Fast(t_seq, "t must be a sequence");
    if (!x_fast || !t_fast) { Py_XDECREF(x_fast); Py_XDECREF(t_fast); return NULL; }

    if (PySequence_Fast_GET_SIZE(x_fast) != nn->nips ||
        PySequence_Fast_GET_SIZE(t_fast) != nn->nops) {
        Py_DECREF(x_fast); Py_DECREF(t_fast);
        PyErr_SetString(PyExc_ValueError, "length mismatch (x or t)");
        return NULL;
    }

    /* allocate two small temp buffers on the C-heap */
    float *xin = malloc(nn->nips * sizeof *xin);
    float *tgt = malloc(nn->nops * sizeof *tgt);
    if (!xin || !tgt) { PyErr_NoMemory(); goto cleanup; }

    for (int i = 0; i < nn->nips; ++i)
        xin[i] = (float)PyFloat_AsDouble(PySequence_Fast_GET_ITEM(x_fast, i));
    for (int i = 0; i < nn->nops; ++i)
        tgt[i] = (float)PyFloat_AsDouble(PySequence_Fast_GET_ITEM(t_fast, i));

    float loss = NNtrain(*nn, xin, tgt, (float)lr);

cleanup:
    free(xin); free(tgt);
    Py_DECREF(x_fast); Py_DECREF(t_fast);
    if (PyErr_Occurred()) return NULL;
    return PyFloat_FromDouble(loss);
}
