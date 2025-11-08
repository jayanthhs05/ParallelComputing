#ifndef MOVIES_H
#define MOVIES_H

#include "data_structures.h"

typedef struct {
  char *title;
  char *genres;
} Movie;

int *load_movie_mapping(const char *filename, int *num_movies);
void load_movies(Movie *movies, int num_movies, const char *filename,
                 int *remap_table, int max_id);
void search_titles(Movie *movies, int num_movies, const char *query,
                   int **results, int *num_results);

#endif
