#include "config.h"
#include "data_loader.h"
#include "data_structures.h"
#include "model.h"
#include "train.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <ratings_file.csv>\n", argv[0]);
    return 1;
  }

  clock_t start_time = clock();

  Dataset *dataset = load_dataset(argv[1]);
  IDMapper *mapper = create_id_mapper(dataset);
  remap_ids(dataset, mapper);

  printf("Dataset: %d ratings, %d users, %d movies\n", dataset->num_ratings,
         dataset->num_users, dataset->num_movies);

  Dataset *train_data, *test_data;
  split_data(dataset, &train_data, &test_data, TRAIN_TEST_SPLIT);

  Model *model = create_model(dataset->num_users, dataset->num_movies,
                              NUM_FACTORS, LEARNING_RATE, REGULARIZATION);

  compute_global_mean(model, train_data);
  printf("Global mean rating: %.4f\n", model->global_mean);

  initialize_model(model);

  printf("Training model with %d factors for %d iterations\n", NUM_FACTORS,
         NUM_ITERATIONS);
  train_model(model, train_data, NUM_ITERATIONS);

  float rmse = compute_rmse(model, test_data);
  printf("Test RMSE: %.4f\n", rmse);

  clock_t end_time = clock();
  double cpu_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

  printf("EXECUTION_TIME: %.2f\n", cpu_time);

  free_model(model);
  free_dataset(train_data);
  free_dataset(test_data);
  free_dataset(dataset);
  free_id_mapper(mapper);

  return 0;
}