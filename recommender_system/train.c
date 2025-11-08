#include "train.h"
#include "config.h"
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static float predict_rating(Model *model, int user_id, int movie_id) {
  float prediction = 0.0;

  for (int k = 0; k < model->num_factors; k++) {
    prediction +=
        model->user_features[user_id][k] * model->movie_features[movie_id][k];
  }

  if (prediction > 5.0)
    prediction = 5.0;
  if (prediction < 0.5)
    prediction = 0.5;

  return prediction;
}

void train_model_parallel(Model *model, Dataset *train_data, int num_iterations,
                          int rank, int size) {
  int local_start = (train_data->num_ratings / size) * rank;
  int local_end = (rank == size - 1)
                      ? train_data->num_ratings
                      : (train_data->num_ratings / size) * (rank + 1);

  int sync_interval = 5;
  if (size <= 2)
    sync_interval = 3;
  else if (size >= 8)
    sync_interval = 10;

  if (rank == 0) {
    printf("Using synchronization interval: %d iterations\n", sync_interval);
  }

  int user_feature_size = model->num_users * model->num_factors;
  int movie_feature_size = model->num_movies * model->num_factors;
  float *flat_user_features =
      (float *)malloc(user_feature_size * sizeof(float));
  float *flat_movie_features =
      (float *)malloc(movie_feature_size * sizeof(float));

  double comm_time = 0.0, comp_time = 0.0;
  int sync_count = 0;

  for (int iter = 0; iter < num_iterations; iter++) {
    double iter_start = MPI_Wtime();

    for (int idx = local_start; idx < local_end; idx++) {
      int user_id = train_data->ratings[idx].user_id;
      int movie_id = train_data->ratings[idx].movie_id;
      float actual_rating = train_data->ratings[idx].rating;

      float predicted_rating = predict_rating(model, user_id, movie_id);
      float error = actual_rating - predicted_rating;

      for (int k = 0; k < model->num_factors; k++) {
        float user_feature = model->user_features[user_id][k];
        float movie_feature = model->movie_features[movie_id][k];

        float user_grad =
            error * movie_feature - model->regularization * user_feature;
        float movie_grad =
            error * user_feature - model->regularization * movie_feature;

        model->user_features[user_id][k] += model->learning_rate * user_grad;
        model->movie_features[movie_id][k] += model->learning_rate * movie_grad;
      }
    }

    comp_time += MPI_Wtime() - iter_start;

    if (((iter + 1) % sync_interval == 0) || (iter == num_iterations - 1)) {
      double comm_start = MPI_Wtime();
      sync_count++;

      for (int i = 0; i < model->num_users; i++) {
        int base_idx = i * model->num_factors;
        for (int k = 0; k < model->num_factors; k++) {
          flat_user_features[base_idx + k] = model->user_features[i][k];
        }
      }

      for (int i = 0; i < model->num_movies; i++) {
        int base_idx = i * model->num_factors;
        for (int k = 0; k < model->num_factors; k++) {
          flat_movie_features[base_idx + k] = model->movie_features[i][k];
        }
      }

      MPI_Allreduce(MPI_IN_PLACE, flat_user_features, user_feature_size,
                    MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

      MPI_Allreduce(MPI_IN_PLACE, flat_movie_features, movie_feature_size,
                    MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

      float scale = 1.0f / size;

      for (int i = 0; i < model->num_users; i++) {
        int base_idx = i * model->num_factors;
        for (int k = 0; k < model->num_factors; k++) {
          model->user_features[i][k] = flat_user_features[base_idx + k] * scale;
        }
      }

      for (int i = 0; i < model->num_movies; i++) {
        int base_idx = i * model->num_factors;
        for (int k = 0; k < model->num_factors; k++) {
          model->movie_features[i][k] =
              flat_movie_features[base_idx + k] * scale;
        }
      }

      comm_time += MPI_Wtime() - comm_start;

      if (rank == 0) {
        printf("Iteration %d completed (synchronized)\n", iter + 1);
      }
    } else if (rank == 0 && iter % 5 == 0) {
      printf("Iteration %d completed (local)\n", iter + 1);
    }
  }

  if (rank == 0) {
    printf("\nTraining Performance Breakdown\n");
    printf("Computation time: %.2f seconds (%.1f%%)\n", comp_time,
           comp_time / (comp_time + comm_time) * 100);
    printf("Communication time: %.2f seconds (%.1f%%)\n", comm_time,
           comm_time / (comp_time + comm_time) * 100);
    printf("Total synchronizations: %d\n", sync_count);
    printf("Communication reduction: %.1f%%\n",
           (1.0 - (float)sync_count / num_iterations) * 100);
  }

  free(flat_user_features);
  free(flat_movie_features);
}

float compute_rmse(Model *model, Dataset *test_data, int rank, int size) {
  int local_start = (test_data->num_ratings / size) * rank;
  int local_end = (rank == size - 1)
                      ? test_data->num_ratings
                      : (test_data->num_ratings / size) * (rank + 1);

  float local_squared_error = 0.0;

  for (int idx = local_start; idx < local_end; idx++) {
    int user_id = test_data->ratings[idx].user_id;
    int movie_id = test_data->ratings[idx].movie_id;
    float actual_rating = test_data->ratings[idx].rating;

    float predicted_rating = predict_rating(model, user_id, movie_id);
    float error = actual_rating - predicted_rating;
    local_squared_error += error * error;
  }

  float total_squared_error = 0.0;
  MPI_Reduce(&local_squared_error, &total_squared_error, 1, MPI_FLOAT, MPI_SUM,
             0, MPI_COMM_WORLD);

  float rmse = 0.0;
  if (rank == 0) {
    rmse = sqrt(total_squared_error / test_data->num_ratings);
  }

  MPI_Bcast(&rmse, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  return rmse;
}
