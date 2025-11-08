#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

typedef struct {
  int user_id;
  int movie_id;
  float rating;
  long timestamp;
} Rating;

typedef struct {
  Rating *ratings;
  int num_ratings;
  int num_users;
  int num_movies;
  int max_user_id;
  int max_movie_id;
} Dataset;

typedef struct {
  int *user_map;
  int *movie_map;
  int *reverse_user_map;
  int *reverse_movie_map;
} IDMapper;

typedef struct {
  float **user_features;
  float **movie_features;
  int num_users;
  int num_movies;
  int num_factors;
  float learning_rate;
  float regularization;
} Model;

#endif
