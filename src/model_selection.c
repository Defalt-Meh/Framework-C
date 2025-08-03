/*
 * Multithreaded model selection with pthreads for hyperparameter sweep,
 * using random sampling, early stopping, k-fold cross-validation,
 * and size-penalized scoring.
 */
#include "model_selection.h"

// Thread argument structure
typedef struct {
    const Data *data;       // full dataset for cross-validation
    int nips, nops;
    int min_hid, max_hid, step;
    float rate, eta;
    int epochs;
    int patience;
    int folds;
    float alpha;            // penalty weight

    int next_hid;
    int best_hid;
    float best_score;
    pthread_mutex_t lock;
    unsigned int seed;
} ThreadArg;

// Helper: evaluate one hidden size with random sampling & early stopping
static float train_validate(int hid, const Data *train, const Data *val,
                            int nips, int nops,
                            float rate, float eta,
                            int epochs, int patience,
                            unsigned int *seed)
{
    NeuralNetwork_Type nn = NNbuild(nips, hid, nops);
    if (nn.nw == 0) return FLT_MAX;

    int rows = train->rows;
    int vrows = val->rows;
    float lr = rate;
    float best_val = FLT_MAX;
    int no_improve = 0;
    for (int e = 0; e < epochs; ++e) {
        // random-sample SGD
        for (int i = 0; i < rows; ++i) {
            int idx = rand_r(seed) % rows;
            NNtrain(nn, train->in[idx], train->tg[idx], lr);
        }
        lr *= eta;
        // validate
        float val_err = 0.0f;
        for (int i = 0; i < vrows; ++i) {
            const float *out = NNpredict(nn, val->in[i]);
            for (int j = 0; j < nops; ++j) {
                float d = out[j] - val->tg[i][j];
                val_err += 0.5f * d * d;
            }
        }
        val_err /= vrows;
        // early stop
        if (val_err + 1e-6f < best_val) {
            best_val = val_err;
            no_improve = 0;
        } else if (++no_improve >= patience) {
            break;
        }
    }
    NNfree(nn);
    return best_val;
}

// Optimized cross_validate: hoists allocations, removes per‐fold malloc/free,
// and streamlines pointer assignments for maximum speed.

static float cross_validate(int hid, const Data *full,
                            int nips, int nops,
                            float rate, float eta,
                            int epochs, int patience,
                            int folds, unsigned int *seed)
{
    int N = full->rows;
    int fold_size  = N / folds;
    int train_size = N - fold_size;
    float sum_err  = 0.0f;
    unsigned int local_seed = *seed;

    // —— Pre‐allocate all needed buffers once ——
    int    *idx       = malloc(N * sizeof(int));
    float **train_in  = malloc(train_size * sizeof(float*));
    float **train_tg  = malloc(train_size * sizeof(float*));
    float **val_in    = malloc(fold_size  * sizeof(float*));
    float **val_tg    = malloc(fold_size  * sizeof(float*));
    if (!idx || !train_in || !train_tg || !val_in || !val_tg) {
        free(idx); free(train_in); free(train_tg); free(val_in); free(val_tg);
        return FLT_MAX;
    }

    // —— Initialize & shuffle indices once —— 
    for (int i = 0; i < N; ++i) idx[i] = i;
    for (int i = N - 1; i > 0; --i) {
        int j = rand_r(&local_seed) % (i + 1);
        int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }

    // —— k-fold loop with zero mallocs/frees —— 
    for (int f = 0; f < folds; ++f) {
        int start = f * fold_size;
        int end   = start + fold_size;
        int ti = 0, vi = 0;

        // fill first train block [0..start)
        for (int i = 0; i < start; ++i) {
            int id = idx[i];
            train_in[ti] = full->in[id];
            train_tg[ti] = full->tg[id];
            ++ti;
        }
        // fill validation block [start..end)
        for (int i = start; i < end; ++i) {
            int id = idx[i];
            val_in[vi] = full->in[id];
            val_tg[vi] = full->tg[id];
            ++vi;
        }
        // fill second train block [end..N)
        for (int i = end; i < N; ++i) {
            int id = idx[i];
            train_in[ti] = full->in[id];
            train_tg[ti] = full->tg[id];
            ++ti;
        }

        // alias into Data structs (no copies)
        Data train = { train_in, train_tg, nips, nops, train_size };
        Data val   = { val_in,   val_tg,   nips, nops, fold_size };

        sum_err += train_validate(hid, &train, &val,
                                  nips, nops,
                                  rate, eta,
                                  epochs, patience,
                                  &local_seed);
    }

    // —— Cleanup and propagate RNG state —— 
    free(idx);
    free(train_in); free(train_tg);
    free(val_in);   free(val_tg);

    *seed = local_seed;
    return sum_err / folds;
}


// Worker thread: sweeps hidden sizes using CV and penalized score
static void *worker_func(void *arg) {
    ThreadArg *ta = (ThreadArg*)arg;
    unsigned int seed = ta->seed ^ (unsigned int)(size_t)pthread_self();
    for (;;) {
        pthread_mutex_lock(&ta->lock);
        int hid = ta->next_hid;
        if (hid > ta->max_hid) { pthread_mutex_unlock(&ta->lock); break; }
        ta->next_hid += ta->step;
        pthread_mutex_unlock(&ta->lock);
        // cross-validated error
        float cv_err = cross_validate(hid, ta->data,
                                      ta->nips, ta->nops,
                                      ta->rate, ta->eta,
                                      ta->epochs, ta->patience,
                                      ta->folds, &seed);
        // penalized score
        float score = cv_err + ta->alpha * ((float)hid * ta->nips) / (ta->max_hid * ta->nips);
        pthread_mutex_lock(&ta->lock);
        if (score < ta->best_score) {
            ta->best_score = score;
            ta->best_hid = hid;
        }
        pthread_mutex_unlock(&ta->lock);
    }
    return NULL;
}

int select_optimal_hidden(const Data *train,
                          const Data *val,
                          int nips, int nops,
                          int min_hid, int max_hid, int step,
                          float rate, float eta, int epochs)
{
    if (!train || !val || min_hid < 1 || max_hid < min_hid || step < 1)
        return -1;
    // combine train+val for CV
    Data full = { .rows = train->rows + val->rows,
        .in = malloc((train->rows+val->rows)*sizeof(float*)),
        .tg = malloc((train->rows+val->rows)*sizeof(float*)) };
    for (int i = 0; i < train->rows; ++i) { full.in[i] = train->in[i]; full.tg[i] = train->tg[i]; }
    for (int i = 0; i < val->rows; ++i) { full.in[train->rows+i] = val->in[i]; full.tg[train->rows+i] = val->tg[i]; }

    ThreadArg ta = {
        .data      = &full,
        .nips      = nips,
        .nops      = nops,
        .min_hid   = min_hid,
        .max_hid   = max_hid,
        .step      = step,
        .rate      = rate,
        .eta       = eta,
        .epochs    = epochs,
        .patience  = 5,
        .folds     = 5,
        .alpha     = 0.01f,
        .next_hid  = min_hid,
        .best_hid  = min_hid,
        .best_score= FLT_MAX,
        .seed      = (unsigned int)time(NULL)
    };
    pthread_mutex_init(&ta.lock, NULL);

    int num_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (num_threads < 1) num_threads = 1;
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    for (int t = 0; t < num_threads; ++t) pthread_create(&threads[t], NULL, worker_func, &ta);
    for (int t = 0; t < num_threads; ++t) pthread_join(threads[t], NULL);
    free(threads);

    pthread_mutex_destroy(&ta.lock);
    free(full.in); free(full.tg);
    return ta.best_hid;
}
