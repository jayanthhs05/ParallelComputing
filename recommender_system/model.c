#include "model.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

Model *create_model(int num_users, int num_movies, int num_factors,
                    float learning_rate, float regularization) {
  Model *model = (Model *)malloc(sizeof(Model));
  model->num_users = num_users;
  model->num_movies = num_movies;
  model->num_factors = num_factors;
  model->learning_rate = learning_rate;
  model->regularization = regularization;

  model->user_features = (float **)malloc(num_users * sizeof(float *));
  for (int i = 0; i < num_users; i++) {
    model->user_features[i] = (float *)malloc(num_factors * sizeof(float));
  }

  model->movie_features = (float **)malloc(num_movies * sizeof(float *));
  for (int i = 0; i < num_movies; i++) {
    model->movie_features[i] = (float *)malloc(num_factors * sizeof(float));
  }

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

void save_model(const char *filename, Model *model) {
  FILE *f = fopen(filename, "wb");
  if (!f) {
    fprintf(stderr, "Error saving model to %s: %s\n", filename,
            strerror(errno));
    return;
  }
  fwrite(&model->num_users, sizeof(int), 1, f);
  fwrite(&model->num_movies, sizeof(int), 1, f);
  fwrite(&model->num_factors, sizeof(int), 1, f);
  for (int i = 0; i < model->num_users; i++) {
    fwrite(model->user_features[i], sizeof(float), model->num_factors, f);
  }
  for (int i = 0; i < model->num_movies; i++) {
    fwrite(model->movie_features[i], sizeof(float), model->num_factors, f);
  }
  fclose(f);
}

Model *load_model(const char *filename) {
  FILE *f = fopen(filename, "rb");
  if (!f) {
    fprintf(stderr, "Error opening model file '%s': %s\n", filename,
            strerror(errno));
    fprintf(stderr,
            "Make sure the file exists and you have read permissions.\n");
    return NULL;
  }

  Model *model = (Model *)malloc(sizeof(Model));

  if (fread(&model->num_users, sizeof(int), 1, f) != 1) {
    fprintf(stderr, "Error reading num_users from model file\n");
    free(model);
    fclose(f);
    return NULL;
  }

  if (fread(&model->num_movies, sizeof(int), 1, f) != 1) {
    fprintf(stderr, "Error reading num_movies from model file\n");
    free(model);
    fclose(f);
    return NULL;
  }

  if (fread(&model->num_factors, sizeof(int), 1, f) != 1) {
    fprintf(stderr, "Error reading num_factors from model file\n");
    free(model);
    fclose(f);
    return NULL;
  }

  model->learning_rate = 0.005f;
  model->regularization = 0.02f;

  model->user_features = (float **)malloc(model->num_users * sizeof(float *));
  for (int i = 0; i < model->num_users; i++) {
    model->user_features[i] =
        (float *)malloc(model->num_factors * sizeof(float));
    if (fread(model->user_features[i], sizeof(float), model->num_factors, f) !=
        (size_t)model->num_factors) {
      fprintf(stderr, "Error reading user features at index %d\n", i);
      for (int j = 0; j <= i; j++) {
        free(model->user_features[j]);
      }
      free(model->user_features);
      free(model);
      fclose(f);
      return NULL;
    }
  }

  model->movie_features = (float **)malloc(model->num_movies * sizeof(float *));
  for (int i = 0; i < model->num_movies; i++) {
    model->movie_features[i] =
        (float *)malloc(model->num_factors * sizeof(float));
    if (fread(model->movie_features[i], sizeof(float), model->num_factors, f) !=
        (size_t)model->num_factors) {
      fprintf(stderr, "Error reading movie features at index %d\n", i);
      for (int j = 0; j < model->num_users; j++) {
        free(model->user_features[j]);
      }
      free(model->user_features);
      for (int j = 0; j <= i; j++) {
        free(model->movie_features[j]);
      }
      free(model->movie_features);
      free(model);
      fclose(f);
      return NULL;
    }
  }

  fclose(f);
  return model;
}