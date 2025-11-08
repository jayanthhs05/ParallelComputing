#include "data_loader.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Dataset *load_dataset(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "Error opening file: %s\n", filename);
    exit(1);
  }

  int num_ratings = 0;
  int max_user = 0, max_movie = 0;
  char line[MAX_LINE_LENGTH];
  fgets(line, MAX_LINE_LENGTH, file);

  while (fgets(line, MAX_LINE_LENGTH, file)) {
    num_ratings++;
  }
  rewind(file);
  fgets(line, MAX_LINE_LENGTH, file);

  Dataset *dataset = (Dataset *)malloc(sizeof(Dataset));
  dataset->num_ratings = num_ratings;
  dataset->ratings = (Rating *)malloc(num_ratings * sizeof(Rating));

  for (int i = 0; i < num_ratings; i++) {
    fgets(line, MAX_LINE_LENGTH, file);
    sscanf(line, "%d,%d,%f,%ld", &dataset->ratings[i].user_id,
           &dataset->ratings[i].movie_id, &dataset->ratings[i].rating,
           &dataset->ratings[i].timestamp);

    if (dataset->ratings[i].user_id > max_user)
      max_user = dataset->ratings[i].user_id;
    if (dataset->ratings[i].movie_id > max_movie)
      max_movie = dataset->ratings[i].movie_id;
  }
  fclose(file);

  dataset->max_user_id = max_user;
  dataset->max_movie_id = max_movie;
  return dataset;
}

void free_dataset(Dataset *dataset) {
  if (dataset) {
    free(dataset->ratings);
    free(dataset);
  }
}

IDMapper *create_id_mapper(Dataset *dataset) {
  IDMapper *mapper = (IDMapper *)malloc(sizeof(IDMapper));
  mapper->user_map = (int *)calloc(dataset->max_user_id + 1, sizeof(int));
  mapper->movie_map = (int *)calloc(dataset->max_movie_id + 1, sizeof(int));

  int *user_exists = (int *)calloc(dataset->max_user_id + 1, sizeof(int));
  int *movie_exists = (int *)calloc(dataset->max_movie_id + 1, sizeof(int));

  for (int i = 0; i < dataset->num_ratings; i++) {
    user_exists[dataset->ratings[i].user_id] = 1;
    movie_exists[dataset->ratings[i].movie_id] = 1;
  }

  int user_count = 0;
  for (int i = 0; i <= dataset->max_user_id; i++) {
    if (user_exists[i]) {
      mapper->user_map[i] = user_count++;
    }
  }

  int movie_count = 0;
  for (int i = 0; i <= dataset->max_movie_id; i++) {
    if (movie_exists[i]) {
      mapper->movie_map[i] = movie_count++;
    }
  }

  dataset->num_users = user_count;
  dataset->num_movies = movie_count;

  mapper->reverse_user_map = (int *)malloc(user_count * sizeof(int));
  mapper->reverse_movie_map = (int *)malloc(movie_count * sizeof(int));

  for (int i = 0; i <= dataset->max_user_id; i++) {
    if (user_exists[i]) {
      mapper->reverse_user_map[mapper->user_map[i]] = i;
    }
  }

  for (int i = 0; i <= dataset->max_movie_id; i++) {
    if (movie_exists[i]) {
      mapper->reverse_movie_map[mapper->movie_map[i]] = i;
    }
  }

  free(user_exists);
  free(movie_exists);
  return mapper;
}

void free_id_mapper(IDMapper *mapper) {
  if (mapper) {
    free(mapper->user_map);
    free(mapper->movie_map);
    free(mapper->reverse_user_map);
    free(mapper->reverse_movie_map);
    free(mapper);
  }
}

void remap_ids(Dataset *dataset, IDMapper *mapper) {
  for (int i = 0; i < dataset->num_ratings; i++) {
    dataset->ratings[i].user_id = mapper->user_map[dataset->ratings[i].user_id];
    dataset->ratings[i].movie_id =
        mapper->movie_map[dataset->ratings[i].movie_id];
  }
}

void split_data(Dataset *dataset, Dataset **train, Dataset **test,
                float split_ratio) {
  int train_size = (int)(dataset->num_ratings * split_ratio);
  int test_size = dataset->num_ratings - train_size;

  *train = (Dataset *)malloc(sizeof(Dataset));
  *test = (Dataset *)malloc(sizeof(Dataset));

  (*train)->num_ratings = train_size;
  (*train)->num_users = dataset->num_users;
  (*train)->num_movies = dataset->num_movies;
  (*train)->ratings = (Rating *)malloc(train_size * sizeof(Rating));

  (*test)->num_ratings = test_size;
  (*test)->num_users = dataset->num_users;
  (*test)->num_movies = dataset->num_movies;
  (*test)->ratings = (Rating *)malloc(test_size * sizeof(Rating));

  for (int i = 0; i < train_size; i++) {
    (*train)->ratings[i] = dataset->ratings[i];
  }
  for (int i = 0; i < test_size; i++) {
    (*test)->ratings[i] = dataset->ratings[train_size + i];
  }
}
