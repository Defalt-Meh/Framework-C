#ifndef MODEL_SELECTION_H
#define MODEL_SELECTION_H

#include <pthread.h>
#include <unistd.h>     // sysconf
#include <float.h>      // FLT_MAX
#include <stdlib.h>     // rand_r, malloc, free
#include <time.h>       // time
#include <string.h>     // memcpy
#include "nn.h"
#include "utils.h"
#include "data_split.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Heuristic guess for hidden units: 2/3*(nips+nops).
 */
static inline int heuristic_hidden(int nips, int nops) {
    return ((nips + nops) * 2) / 3;
}

/**
 * @brief  Try out several hidden‐layer sizes and pick the best one.
 *
 * For each hid in [min_hid…max_hid] stepping by `step`, this
 * function:
 *   1. Builds a network nips→hid→nops
 *   2. Trains it for `epochs` on `train` (no val/test leakage!)
 *   3. Computes mean squared error on `val`
 *   4. Frees the network
 *
 * Returns the hidden‐unit count with the lowest validation error.
 *
 * @returns best_hid on success, or -1 on invalid args.
 */
int select_optimal_hidden(const Data *train,
                          const Data *val,
                          int nips, int nops,
                          int min_hid, int max_hid, int step,
                          float rate, float eta, int epochs);

#ifdef __cplusplus
}
#endif

#endif  /* MODEL_SELECTION_H */
