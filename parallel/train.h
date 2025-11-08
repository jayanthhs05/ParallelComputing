#ifndef TRAIN_H
#define TRAIN_H

#include "data_structures.h"

void train_model_parallel(Model *model, Dataset *train_data, int num_iterations,
                          int rank, int size);
float compute_rmse(Model *model, Dataset *test_data, int rank, int size);

#endif
