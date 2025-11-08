#include "config.h"
#include "data_loader.h"
#include "data_structures.h"
#include "model.h"
#include "train.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  int rank, size;
  double start_time, end_time;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc < 2) {
    if (rank == 0) {
      printf("Usage: %s <ratings_file.csv>\n", argv[0]);
    }
    MPI_Finalize();
    return 1;
  }

  start_time = MPI_Wtime();

  if (rank == 0) {
    printf("Loading dataset\n");
  }
  Dataset *dataset = load_dataset(argv[1], rank);

  if (rank == 0) {
    printf("Creating ID mappings\n");
  }
  IDMapper *mapper = create_id_mapper(dataset);
  remap_ids(dataset, mapper);

  if (rank == 0) {
    printf("Dataset: %d ratings, %d users, %d movies\n", dataset->num_ratings,
           dataset->num_users, dataset->num_movies);
  }

  Dataset *train_data, *test_data;
  if (rank == 0) {
    printf("Splitting data\n");
  }
  split_data(dataset, &train_data, &test_data, TRAIN_TEST_SPLIT, rank);

  if (rank == 0) {
    printf("Train: %d, Test: %d\n", train_data->num_ratings,
           test_data->num_ratings);
    printf("Creating model\n");
  }

  Model *model = create_model(dataset->num_users, dataset->num_movies,
                              NUM_FACTORS, LEARNING_RATE, REGULARIZATION);

  compute_global_mean(model, train_data);
  if (rank == 0) {
    printf("Global mean rating: %.4f\n", model->global_mean);
  }

  initialize_model(model, rank);

  if (rank == 0) {
    printf("Training model with %d factors for %d iterations\n", NUM_FACTORS,
           NUM_ITERATIONS);
  }

  train_model_parallel(model, train_data, NUM_ITERATIONS, rank, size);

  if (rank == 0) {
    printf("Computing RMSE on test set\n");
  }
  float rmse = compute_rmse(model, test_data, rank, size);

  if (rank == 0) {
    printf("Test RMSE: %.4f\n", rmse);

    printf("Saving model\n");
    save_model("model.bin", model);

    FILE *f = fopen("movie_mapping.bin", "wb");
    if (f) {
      fwrite(&dataset->num_movies, sizeof(int), 1, f);
      fwrite(mapper->reverse_movie_map, sizeof(int), dataset->num_movies, f);
      fclose(f);
    }
    printf("Model saved to model.bin and movie_mapping.bin\n");
  }

  end_time = MPI_Wtime();

  if (rank == 0) {
    printf("\nTotal execution time: %.2f seconds\n", end_time - start_time);
  }

  free_model(model);
  free_dataset(train_data);
  free_dataset(test_data);
  free_dataset(dataset);
  free_id_mapper(mapper);

  MPI_Finalize();
  return 0;
}