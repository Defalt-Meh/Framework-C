/*
 *  src/my_module.c  –  Python bridge for FRAMEWORK-C
 *  Auto-depth selection is decided INSIDE this module.
 *
 *  Overview:
 *  This file exposes a thin, zero-copy bridge between NumPy arrays and the
 *  C engine. We pass raw pointers into the C core (while ensuring dtype,
 *  contiguity, and shape), release the GIL during numerical kernels, and
 *  employ *lazy materialization*: model memory is only allocated when first
 *  used. A small heuristic (based on an observed sample count) selects the
 *  network depth on the fly. Weak-linking NNbuild2 enables optional 2-layer
 *  builds without hard dependency at link time.
 */
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>    /* memcpy */
#include "nn.h"

/* ───────────────────────────── Auto-depth thresholds ─────────────────────────
 * Heuristic guidance:
 *  - For “small” data, prefer shallower models to control variance.
 *  - For medium data, two layers but narrower second hidden layer.
 *  - For large data, two layers of equal width.
 * These thresholds are module-local defaults; the C core remains agnostic.  */
#ifndef FWC_AUTO_N_SMALL
#define FWC_AUTO_N_SMALL  10000   /* <10k → 1 layer */
#endif
#ifndef FWC_AUTO_N_MED
#define FWC_AUTO_N_MED    50000   /* 10k–50k → 2 layers (smaller h2) */
#endif

/* Try to use NNbuild2 if it exists at link time; otherwise this will be NULL.
 * Pedagogical note: a weak symbol allows optional functionality without
 * link failure. We test its address (&NNbuild2) before calling it. */
#if defined(__GNUC__) || defined(__clang__)
extern NeuralNetwork_Type NNbuild2(int nips, int nhid, int nhid2, int nops)
    __attribute__((weak));
#else
/* Non-GNU/Clang: declare and assume unavailable (will fallback to NNbuild).
 * The pragma is merely advisory; functionality falls back deterministically. */
extern NeuralNetwork_Type NNbuild2(int nips, int nhid, int nhid2, int nops);
#pragma message("Warning: weak reference to NNbuild2 not supported; using NNbuild fallback.")
#endif

/* ─────────────────────────── Internal wrapper handle ─────────────────────────
 * We encapsulate configuration and (eventually) the realized C network inside
 * a PyCapsule-managed heap object. The 'is_built' flag enforces lazy init.   */
typedef struct {
    int nips, nhid, nops;
    unsigned seed;              /* forwarded to srand for initialization noise */
    int is_built;               /* 0 until first materialization */
    NeuralNetwork_Type net;     /* valid only when is_built == 1 */
} FC_Handle;

/* ─────────────────────────── Capsule helpers & dtor ──────────────────────────
 * Capsule destructor: called when Python GC drops last reference (or module
 * teardown). We free C-side buffers via NNdestroy and then the handle itself.
 * NB: Capsule tag "frameworkc.nn" guards against type confusion.             */
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
   maybe_N: pass B from train_batch as a proxy for dataset size; 0 if unknown.
 *
 * Mathematical/engineering rationale:
 *  - Lazy construction avoids paying allocation/init unless we really need it.
 *  - Random seeding occurs here so repeated builds can be reproducible if
 *    the same 'seed' is supplied to py_build.
 *  - Auto-depth logic mirrors the thresholds at the top of the file.
 *  - If NNbuild2 is not present (weak symbol missing), we degrade gracefully. */
static int ensure_built(FC_Handle *h, long long maybe_N)
{
    if (h->is_built) return 1;

    /* Seed RNG (used by wbrand inside NNbuild)
     * The seed affects initial weight distributions (He/Xavier). */
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
        /* Attempt 2-layer build
         * Fallback to 1-layer on allocation failure to preserve liveness. */
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
#ifdef FWC_OAT
    {   static const int grid[] = {1,4,8,16,32,64,128};
        NN_oat_calibrate(&h->net, grid, (int)(sizeof(grid)/sizeof(grid[0])),
                         /*repeats*/5, /*target_ms*/-1.0f);
    }
#endif

    return 1;
}

/* ───────────────────────────────── py_build ──────────────────────────────────
 * build(nips, nhid, nops [, seed]) → capsule
 * NOTE: does NOT immediately allocate the heavy network;
 *       we lazily materialize on the first training/predict call.
 *
 * API contract:
 *   - Returns a PyCapsule that owns an FC_Handle*. Life-cycle is tied to the
 *     capsule; its destructor ensures we do not leak C resources.             */
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

/* ───────────────────────────── forward decls ────────────────────────────────
 * We expose four user-facing entry points: predict, predict_batch, train_one,
 * and train_batch. All use NumPy C-API for zero-copy access to buffers.     */
static PyObject *py_predict(PyObject *self, PyObject *args);
static PyObject *py_predict_batch_fast(PyObject *self, PyObject *args);
static PyObject *py_train_one(PyObject *self, PyObject *args);
static PyObject *py_train_batch(PyObject *self, PyObject *args);
#ifdef FWC_OAT
static PyObject *py_calibrate(PyObject *self, PyObject *args);
#endif


/* ─────────────────────────── Method table & init ─────────────────────────────
 * The module definition is conventional: a small docstring, -1 m_size (no
 * per-interpreter state), and an array of PyMethodDef entries.               */
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
#ifdef FWC_OAT
    {"calibrate",     py_calibrate,          METH_VARARGS,  "calibrate(net[, latency_ms, repeats]) -> dict"},
#endif
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
    import_array();  /* NumPy C-API init (must be called before NumPy use) */
    return PyModule_Create(&mod);
}

/* ────────────────────────────── predict (1D) ────────────────────────────────
 * Signature: predict(net_capsule, x: float32[nips]) -> float32[nops]
 * Steps:
 *   1) Validate capsule/tag and lazily build the net (1-layer default).
 *   2) Convert input to aligned, C-contiguous float32 without copying if
 *      possible (FROM_OTF honors flags).
 *   3) Release the GIL around the pure-C NNpredict call.
 *   4) Copy the output buffer nn->o into a freshly allocated NumPy array.   */
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

    /* Numerical kernels are thread-agnostic; release the GIL for throughput. */
    Py_BEGIN_ALLOW_THREADS
    float *o = NNpredict(*nn, xin);
    memcpy(yout, o, (size_t)nn->nops * sizeof(float));
    Py_END_ALLOW_THREADS

    Py_DECREF(x_arr);
    return (PyObject*)out;
}

/* ───────────────────────────── predict_batch ────────────────────────────────
 * signature: predict_batch(net_capsule, numpy_in[B,nips]) → numpy_out[B,nops]
 * Batch semantics:
 *   - Uses a BLAS-accelerated forward path under the hood.
 *   - The output array is allocated by NumPy; we write into it directly.     */
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

    /* Release GIL while the C engine computes on B samples. */
    Py_BEGIN_ALLOW_THREADS
    NNpredict_batch(*net, inp, B, outp);
    Py_END_ALLOW_THREADS

    Py_DECREF(in_arr);
    return (PyObject*)out_arr;
}

/* ────────────────────────────── train_one ───────────────────────────────────
 * signature: train_one(net, x[nips], t[nops], lr) -> float loss
 * Didactic note:
 *   This is classic online SGD. We again ensure dtype/contiguity, then
 *   call into C and return a Python float with the ½‖t−o‖² loss.            */
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
    /* Release GIL during compute-intensive step. */
    Py_BEGIN_ALLOW_THREADS
    loss = NNtrain(*nn, xin, tgt, (float)lr);
    Py_END_ALLOW_THREADS

    Py_DECREF(x_arr); Py_DECREF(t_arr);
    return PyFloat_FromDouble((double)loss);
}

/* ────────────────────────────── train_batch ─────────────────────────────────
 * signature: train_batch(net, X[B,nips], Y[B,nops], lr) -> None
 *
 * Array discipline:
 *   - Both X and Y must be float32, C-contiguous, and 2-D matrices.
 *   - Shapes must agree on the batch dimension.
 *
 * Auto-depth remark:
 *   We pass B to ensure_built as a *proxy* for dataset size. When you feed
 *   the full dataset in one call, the heuristic will likely select a 2-layer
 *   architecture (if NNbuild2 is available).                                  */
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

    /* Require float32, 2-D, C-contiguous for both arrays
     * The checks below are defensive: they avoid silent reinterpretation
     * (e.g., wrong dtype/strides) that would corrupt learning. */
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

    /* Training is pure C and CPU-bound; release the GIL for parallel Python. */
    Py_BEGIN_ALLOW_THREADS
    NNtrain_batch(net, B, X, Y, lr);
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

#ifdef FWC_OAT
static PyObject *py_calibrate(PyObject *self, PyObject *args)
{
    PyObject *cap = NULL;
    double latency_ms = -1.0;
    int repeats = 5;
    if (!PyArg_ParseTuple(args, "O|di", &cap, &latency_ms, &repeats))
        return NULL;

    FC_Handle *h = (FC_Handle*)PyCapsule_GetPointer(cap, "frameworkc.nn");
    if (!h) return NULL;

    if (!ensure_built(h, 0)) {
        PyErr_SetString(PyExc_RuntimeError, "failed to build network");
        return NULL;
    }

    static const int grid[] = {1,4,8,16,32,64,128};
    NN_oat_calibrate(&h->net, grid, (int)(sizeof(grid)/sizeof(grid[0])),
                     repeats, (float)latency_ms);

    return Py_BuildValue("{s:f,s:f,s:i,s:i,s:f}",
                         "alpha_ms",        h->net.oat.alpha_ms,
                         "beta_ms",         h->net.oat.beta_ms,
                         "B_star",          h->net.oat.B_star,
                         "use_gemv_single", h->net.oat.use_gemv_single,
                         "target_ms",       h->net.oat.target_ms);
}
#endif
