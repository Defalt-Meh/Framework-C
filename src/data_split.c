#include "data_split.h"
#include <stddef.h>

int split_dataset(const Data *full,
                  float train_frac,
                  float val_frac,
                  Data *out_train,
                  Data *out_val,
                  Data *out_test)
{
    if (!full || !out_train || !out_val || !out_test) {
        return -1;
    }
    if (train_frac <= 0.0f || val_frac < 0.0f || train_frac + val_frac >= 1.0f) {
        return -1;
    }

    int total = full->rows;
    int train_count = (int)(train_frac * total);
    int val_count   = (int)(val_frac   * total);
    int test_count  = total - train_count - val_count;

    if (train_count < 1 || val_count < 0 || test_count < 1) {
        return -1;
    }

    // Training subset
    out_train->rows = train_count;
    out_train->nips = full->nips;
    out_train->nops = full->nops;
    out_train->in   = full->in;
    out_train->tg   = full->tg;

    // Validation subset
    out_val->rows   = val_count;
    out_val->nips   = full->nips;
    out_val->nops   = full->nops;
    out_val->in     = full->in + train_count;
    out_val->tg     = full->tg + train_count;

    // Test subset
    out_test->rows  = test_count;
    out_test->nips  = full->nips;
    out_test->nops  = full->nops;
    out_test->in    = full->in + train_count + val_count;
    out_test->tg    = full->tg + train_count + val_count;

    return 0;
}
