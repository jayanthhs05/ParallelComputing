#include "model.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

Model *create_model(int num_users, int num_movies, int num_factors,
                    float learning_rate, float regularization) {
  Model *model = (Model *)malloc(sizeof(Model));
  model->num_users = num_users;
  model->num_movies = num_movies;
  model->num_factors = num_factors;
  model->learning_rate = learning_rate;
  model->regularization = regularization;
  model->global_mean = 0.0f;

  model->user_features = (float **)malloc(num_users * sizeof(float *));
  for (int i = 0; i < num_users; i++) {
    model->user_features[i] = (float *)malloc(num_factors * sizeof(float));
  }

  model->movie_features = (float **)malloc(num_movies * sizeof(float *));
  for (int i = 0; i < num_movies; i++) {
    model->movie_features[i] = (float *)malloc(num_factors * sizeof(float));
  }

  model->user_bias = (float *)calloc(num_users, sizeof(float));
  model->movie_bias = (float *)calloc(num_movies, sizeof(float));

  return model;
}

void free_model(Model *model) {
  if (model) {
    for (int i = 0; i < model->num_users; i++) {
      free(model->user_features[i]);
    }
    free(model->user_features);

    for (int i = 0; i < model->num_movies; i++) {
      free(model->movie_features[i]);
    }
    free(model->movie_features);

    free(model->user_bias);
    free(model->movie_bias);

    free(model);
  }
}

void initialize_model(Model *model, int rank) {
  unsigned int seed = time(NULL) + rank;

  for (int i = 0; i < model->num_users; i++) {
    for (int j = 0; j < model->num_factors; j++) {
      model->user_features[i][j] = ((float)rand_r(&seed) / RAND_MAX) * 0.1;
    }
  }

  for (int i = 0; i < model->num_movies; i++) {
    for (int j = 0; j < model->num_factors; j++) {
      model->movie_features[i][j] = ((float)rand_r(&seed) / RAND_MAX) * 0.1;
    }
  }
}

void compute_global_mean(Model *model, Dataset *dataset) {
  double sum = 0.0;
  for (int i = 0; i < dataset->num_ratings; i++) {
    sum += dataset->ratings[i].rating;
  }
  model->global_mean = sum / dataset->num_ratings;
}