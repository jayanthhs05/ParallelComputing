#include "movies.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int *load_movie_mapping(const char *filename, int *num_out) {
  FILE *f = fopen(filename, "rb");
  if (!f) {
    perror("fopen mapping");
    exit(1);
  }
  int num;
  if (fread(&num, sizeof(int), 1, f) != 1) {
    fclose(f);
    exit(1);
  }
  int *ids = (int *)malloc(num * sizeof(int));
  if (fread(ids, sizeof(int), num, f) != (unsigned)num) {
    free(ids);
    fclose(f);
    exit(1);
  }
  fclose(f);
  *num_out = num;
  return ids;
}

void load_movies(Movie *movies, int num, const char *filename, int *remap_table,
                 int max_id) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    perror("fopen movies");
    exit(1);
  }
  char line[2048];
  if (!fgets(line, sizeof(line), f)) {
    fclose(f);
    return;
  }
  while (fgets(line, sizeof(line), f)) {
    char *ptr = line;
    char *id_start = ptr;
    char *first_comma = strchr(ptr, ',');
    if (!first_comma)
      continue;
    *first_comma = '\0';
    int orig_id = atoi(id_start);
    ptr = first_comma + 1;
    char *title_start = ptr;
    int quoted = (*ptr == '"');
    char *field_end;
    if (quoted) {
      field_end = strchr(ptr + 1, '"');
      if (!field_end)
        continue;
      char *after_quote = field_end + 1;
      if (*after_quote == ',') {
        ptr = after_quote + 1;
      } else {
        ptr = after_quote;
      }
      int title_len = field_end - title_start - 1;
      memmove(title_start, title_start + 1, title_len);
      title_start[title_len] = '\0';
      field_end = title_start + title_len;
    } else {
      field_end = strchr(ptr, ',');
      if (!field_end)
        continue;
      *field_end = '\0';
      ptr = field_end + 1;
    }
    char *genres_start = ptr;
    char *newline = strchr(genres_start, '\n');
    if (newline)
      *newline = '\0';
    int rem = (orig_id < max_id && orig_id >= 0) ? remap_table[orig_id] : -1;
    if (rem >= 0 && rem < num && movies[rem].title == NULL) {
      movies[rem].title = (char *)malloc(strlen(title_start) + 1);
      strcpy(movies[rem].title, title_start);
      movies[rem].genres = (char *)malloc(strlen(genres_start) + 1);
      strcpy(movies[rem].genres, genres_start);
    }
    *first_comma = ',';
  }
  fclose(f);
}

void search_titles(Movie *movies, int num, const char *query, int **results,
                   int *num_res) {
  *num_res = 0;
  *results = NULL;
  int qlen = strlen(query);
  if (qlen == 0)
    return;
  for (int i = 0; i < num; i++) {
    if (movies[i].title && strstr(movies[i].title, query)) {
      (*num_res)++;
      *results = (int *)realloc(*results, (*num_res) * sizeof(int));
      (*results)[*num_res - 1] = i;
    }
  }
}
