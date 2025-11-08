#ifndef MODEL_H
#define MODEL_H

#include "data_structures.h"

Model *create_model(int num_users, int num_movies, int num_factors,
                    float learning_rate, float regularization);
void free_model(Model *model);
void initialize_model(Model *model, int rank);
void compute_global_mean(Model *model, Dataset *dataset);

#endif