#include "train.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static float predict_rating(Model *model, int user_id, int movie_id) {

  float prediction = model->global_mean + model->user_bias[user_id] +
                     model->movie_bias[movie_id];

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

void train_model(Model *model, Dataset *train_data, int num_iterations) {
  for (int iter = 0; iter < num_iterations; iter++) {
    for (int idx = 0; idx < train_data->num_ratings; idx++) {
      int user_id = train_data->ratings[idx].user_id;
      int movie_id = train_data->ratings[idx].movie_id;
      float actual_rating = train_data->ratings[idx].rating;

      float predicted_rating = predict_rating(model, user_id, movie_id);
      float error = actual_rating - predicted_rating;

      model->user_bias[user_id] +=
          model->learning_rate *
          (error - model->regularization * model->user_bias[user_id]);
      model->movie_bias[movie_id] +=
          model->learning_rate *
          (error - model->regularization * model->movie_bias[movie_id]);

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

    if (iter % 5 == 0) {
      printf("Iteration %d completed\n", iter + 1);
    }
  }
}

float compute_rmse(Model *model, Dataset *test_data) {
  float squared_error = 0.0;

  for (int idx = 0; idx < test_data->num_ratings; idx++) {
    int user_id = test_data->ratings[idx].user_id;
    int movie_id = test_data->ratings[idx].movie_id;
    float actual_rating = test_data->ratings[idx].rating;

    float predicted_rating = predict_rating(model, user_id, movie_id);
    float error = actual_rating - predicted_rating;
    squared_error += error * error;
  }

  return sqrt(squared_error / test_data->num_ratings);
}