#include "config.h"
#include "model.h"
#include "movies.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
  float score;
  int id;
} Candidate;

int compare_candidates(const void *a, const void *b) {
  float sa = ((Candidate *)a)->score;
  float sb = ((Candidate *)b)->score;
  if (sa > sb)
    return -1;
  if (sa < sb)
    return 1;
  return 0;
}

static float predict_for_movie(Model *model, float *user_profile,
                               int movie_id) {
  float prediction = model->global_mean + model->movie_bias[movie_id];

  for (int k = 0; k < model->num_factors; k++) {
    prediction += user_profile[k] * model->movie_features[movie_id][k];
  }

  return prediction;
}

int main() {
  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) != NULL) {
    printf("Current working directory: %s\n", cwd);
  }

  FILE *test = fopen("model.bin", "rb");
  if (test) {
    printf("model.bin can be opened for reading\n");
    fclose(test);
  } else {
    printf("✗ Cannot open model.bin\n");
    perror("fopen");
    return 1;
  }

  printf("Loading model\n");
  Model *model = load_model("model.bin");
  if (!model) {
    printf("Failed to load model\n");
    return 1;
  }
  printf("Model loaded: %d users, %d movies, %d factors\n", model->num_users,
         model->num_movies, model->num_factors);
  printf("Global mean: %.4f\n", model->global_mean);

  int num_movies;
  int *original_ids = load_movie_mapping("movie_mapping.bin", &num_movies);
  printf("Loaded mapping for %d movies\n", num_movies);

  int max_orig = 0;
  for (int i = 0; i < num_movies; i++) {
    if (original_ids[i] > max_orig)
      max_orig = original_ids[i];
  }
  int *remap_table = (int *)malloc((max_orig + 1) * sizeof(int));
  for (int j = 0; j <= max_orig; j++)
    remap_table[j] = -1;
  for (int i = 0; i < num_movies; i++) {
    remap_table[original_ids[i]] = i;
  }
  
  Movie *movies = (Movie *)malloc(num_movies * sizeof(Movie));
  for (int i = 0; i < num_movies; i++) {
    movies[i].title = NULL;
    movies[i].genres = NULL;
  }
  load_movies(movies, num_movies, "data/movies.csv", remap_table, max_orig);

  bool *picked = (bool *)calloc(num_movies, sizeof(bool));
  int *picked_list = NULL;
  int num_picked = 0;
  printf("\nWelcome to the Movie Recommender System!\n");
  printf("Enter partial movie titles to select movies you like.\n");
  printf("Type 'done' when ready for recommendations.\n\n");
  char input[256];
  while (1) {
    printf("Enter movie title: ");
    if (!fgets(input, 256, stdin))
      break;
    input[strcspn(input, "\n")] = 0;
    if (strcmp(input, "done") == 0)
      break;
    int *matches = NULL;
    int num_matches = 0;
    search_titles(movies, num_movies, input, &matches, &num_matches);
    if (num_matches == 0) {
      printf("No matching movies found.\n");
      continue;
    }
    if (num_matches == 1) {
      int mid = matches[0];
      if (picked[mid]) {
        printf("Already picked.\n");
        free(matches);
        continue;
      }
      picked[mid] = true;
      picked_list = (int *)realloc(picked_list, (num_picked + 1) * sizeof(int));
      picked_list[num_picked++] = mid;
      printf("Added: %s\n", movies[mid].title);
    } else {
      printf("Matching movies:\n");
      for (int i = 0; i < num_matches; i++) {
        printf("%d. %s\n", i + 1, movies[matches[i]].title);
      }
      int choice;
      printf("Pick one (1-%d): ", num_matches);
      if (scanf("%d", &choice) == 1 && choice >= 1 && choice <= num_matches) {
        int mid = matches[choice - 1];
        if (picked[mid]) {
          printf("Already picked.\n");
        } else {
          picked[mid] = true;
          picked_list =
              (int *)realloc(picked_list, (num_picked + 1) * sizeof(int));
          picked_list[num_picked++] = mid;
          printf("Added: %s\n", movies[mid].title);
        }
      } else {
        printf("Invalid choice.\n");
      }
      int c;
      while ((c = getchar()) != '\n' && c != EOF)
        ;
    }
    free(matches);
  }
  if (num_picked == 0) {
    printf("No movies picked. Exiting.\n");
    free(original_ids);
    free(remap_table);
    for (int i = 0; i < num_movies; i++) {
      if (movies[i].title)
        free(movies[i].title);
      if (movies[i].genres)
        free(movies[i].genres);
    }
    free(movies);
    free(picked);
    free_model(model);
    return 0;
  }

  float *avg_vec = (float *)calloc(model->num_factors, sizeof(float));
  for (int p = 0; p < num_picked; p++) {
    int mid = picked_list[p];
    for (int k = 0; k < model->num_factors; k++) {
      avg_vec[k] += model->movie_features[mid][k];
    }
  }
  for (int k = 0; k < model->num_factors; k++) {
    avg_vec[k] /= num_picked;
  }

  Candidate *candidates = (Candidate *)malloc(num_movies * sizeof(Candidate));
  int cand_count = 0;
  for (int j = 0; j < num_movies; j++) {
    if (picked[j] || !movies[j].title)
      continue;

    float score = predict_for_movie(model, avg_vec, j);

    candidates[cand_count].score = score;
    candidates[cand_count].id = j;
    cand_count++;
  }
  qsort(candidates, cand_count, sizeof(Candidate), compare_candidates);
  printf("\nTop 10 Recommendations:\n");
  int top_n = (cand_count < 10) ? cand_count : 10;
  for (int i = 0; i < top_n; i++) {
    int mid = candidates[i].id;
    printf("%d. %s (predicted rating: %.2f)\n", i + 1, movies[mid].title,
           candidates[i].score);
  }
  free(avg_vec);
  free(candidates);
  free(picked_list);
  free(picked);
  for (int i = 0; i < num_movies; i++) {
    if (movies[i].title)
      free(movies[i].title);
    if (movies[i].genres)
      free(movies[i].genres);
  }
  free(movies);
  free(remap_table);
  free(original_ids);
  free_model(model);
  return 0;
}