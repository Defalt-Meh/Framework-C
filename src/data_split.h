#ifndef DATA_SPLIT_H
#define DATA_SPLIT_H 

#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Split a dataset into training, validation, and test subsets.
 *
 * Given a full dataset, this function partitions it according to the
 * specified fractions. The three output Data structs share the same
 * underlying arrays as `full`; no deep copies are made.
 *
 * @param full         Pointer to the original Data.
 * @param train_frac   Fraction of samples to assign to training (0 < train_frac < 1).
 * @param val_frac     Fraction of samples to assign to validation
 *                     (0 ≤ val_frac < 1 − train_frac).
 *                     The remaining fraction (1 − train_frac − val_frac)
 *                     is assigned to the test set.
 * @param out_train    Pointer to a Data struct to receive the training subset.
 * @param out_val      Pointer to a Data struct to receive the validation subset.
 * @param out_test     Pointer to a Data struct to receive the test subset.
 * @return             0 on success, or non-zero if the fractions are invalid
 *                     (e.g. train_frac + val_frac ≥ 1, or fractions out of [0,1]).
 *
 * @note The function does NOT shuffle the data; call shuffle(full) beforehand
 *       if you require a random partition.
 * @note Do NOT call dfree() on out_train, out_val, or out_test individually—
 *       they alias the same buffers as `full`. Only call dfree(full) when
 *       you are done with all subsets.
 */
int split_dataset(const Data *full,
                  float train_frac,
                  float val_frac,
                  Data *out_train,
                  Data *out_val,
                  Data *out_test);

#ifdef __cplusplus
}
#endif

#endif /* DATA_SPLIT_H */
