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

typedef struct {
  int movie_id;
  float rating;
} UserRating;

int compare_candidates(const void *a, const void *b) {
  float sa = ((Candidate *)a)->score;
  float sb = ((Candidate *)b)->score;
  if (sa > sb)
    return -1;
  if (sa < sb)
    return 1;
  return 0;
}

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

static void compute_user_profile(Model *model, UserRating *user_ratings,
                                 int num_ratings, float *user_profile) {
  // Initialize user profile to zeros
  for (int k = 0; k < model->num_factors; k++) {
    user_profile[k] = 0.0f;
  }

  // Use simple gradient descent to find user features that best fit the ratings
  float learning_rate = 0.1f;
  int iterations = 100;

  for (int iter = 0; iter < iterations; iter++) {
    for (int i = 0; i < num_ratings; i++) {
      int movie_id = user_ratings[i].movie_id;
      float actual_rating = user_ratings[i].rating;

      // Predict rating with current user profile
      float prediction = model->global_mean + model->movie_bias[movie_id];
      for (int k = 0; k < model->num_factors; k++) {
        prediction += user_profile[k] * model->movie_features[movie_id][k];
      }

      // Compute error
      float error = actual_rating - prediction;

      // Update user profile
      for (int k = 0; k < model->num_factors; k++) {
        user_profile[k] +=
            learning_rate * error * model->movie_features[movie_id][k];
      }
    }
    learning_rate *= 0.95f; // Decay learning rate
  }
}

static float predict_for_new_user(Model *model, float *user_profile,
                                  int movie_id) {
  float prediction = model->global_mean + model->movie_bias[movie_id];

  for (int k = 0; k < model->num_factors; k++) {
    prediction += user_profile[k] * model->movie_features[movie_id][k];
  }

  if (prediction > 5.0)
    prediction = 5.0;
  if (prediction < 0.5)
    prediction = 0.5;

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
  UserRating *user_ratings = NULL;
  int num_user_ratings = 0;

  printf("\n═══════════════════════════════════════════════════════════\n");
  printf("         Movie Recommender System with Ratings\n");
  printf("═══════════════════════════════════════════════════════════\n\n");
  printf("Enter movie titles and rate them (0.5 - 5.0 stars).\n");
  printf("Type 'done' when ready for recommendations.\n");
  printf("Type 'list' to see your rated movies.\n\n");

  char input[256];
  while (1) {
    printf("Enter movie title (or 'done'/'list'): ");
    if (!fgets(input, 256, stdin))
      break;
    input[strcspn(input, "\n")] = 0;

    if (strcmp(input, "done") == 0)
      break;

    if (strcmp(input, "list") == 0) {
      if (num_user_ratings == 0) {
        printf("  No movies rated yet.\n\n");
      } else {
        printf("\n  Your rated movies:\n");
        for (int i = 0; i < num_user_ratings; i++) {
          int mid = user_ratings[i].movie_id;
          printf("  %d. %s - %.1f stars\n", i + 1, movies[mid].title,
                 user_ratings[i].rating);
        }
        printf("\n");
      }
      continue;
    }

    int *matches = NULL;
    int num_matches = 0;
    search_titles(movies, num_movies, input, &matches, &num_matches);

    if (num_matches == 0) {
      printf("  No matching movies found.\n\n");
      continue;
    }

    int selected_movie = -1;

    if (num_matches == 1) {
      selected_movie = matches[0];
      if (picked[selected_movie]) {
        printf("  Already rated this movie.\n\n");
        free(matches);
        continue;
      }
      printf("  Found: %s\n", movies[selected_movie].title);
    } else {
      printf("\n  Found %d matching movies:\n", num_matches);
      for (int i = 0; i < num_matches; i++) {
        printf("  %d. %s\n", i + 1, movies[matches[i]].title);
      }

      int choice;
      printf("  Pick one (1-%d, 0 to cancel): ", num_matches);
      if (scanf("%d", &choice) == 1 && choice >= 1 && choice <= num_matches) {
        selected_movie = matches[choice - 1];
        if (picked[selected_movie]) {
          printf("  Already rated this movie.\n\n");
          int c;
          while ((c = getchar()) != '\n' && c != EOF)
            ;
          free(matches);
          continue;
        }
      } else if (choice == 0) {
        printf("  Cancelled.\n\n");
        int c;
        while ((c = getchar()) != '\n' && c != EOF)
          ;
        free(matches);
        continue;
      } else {
        printf("  Invalid choice.\n\n");
        int c;
        while ((c = getchar()) != '\n' && c != EOF)
          ;
        free(matches);
        continue;
      }
      int c;
      while ((c = getchar()) != '\n' && c != EOF)
        ;
    }

    free(matches);

    // Get rating for the selected movie
    float rating;
    printf("  Rate this movie (0.5 - 5.0): ");
    if (scanf("%f", &rating) == 1) {
      if (rating < 0.5f)
        rating = 0.5f;
      if (rating > 5.0f)
        rating = 5.0f;

      picked[selected_movie] = true;
      user_ratings = (UserRating *)realloc(
          user_ratings, (num_user_ratings + 1) * sizeof(UserRating));
      user_ratings[num_user_ratings].movie_id = selected_movie;
      user_ratings[num_user_ratings].rating = rating;
      num_user_ratings++;

      printf("  ✓ Added: %s - %.1f stars\n\n", movies[selected_movie].title,
             rating);
    } else {
      printf("  Invalid rating.\n\n");
    }

    int c;
    while ((c = getchar()) != '\n' && c != EOF)
      ;
  }

  if (num_user_ratings == 0) {
    printf("\nNo movies rated. Exiting.\n");
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

  printf("\n═══════════════════════════════════════════════════════════\n");
  printf("Computing personalized recommendations...\n");
  printf("═══════════════════════════════════════════════════════════\n\n");

  // Compute user profile based on rated movies
  float *user_profile = (float *)malloc(model->num_factors * sizeof(float));
  compute_user_profile(model, user_ratings, num_user_ratings, user_profile);

  // Generate recommendations
  Candidate *candidates = (Candidate *)malloc(num_movies * sizeof(Candidate));
  int cand_count = 0;
  for (int j = 0; j < num_movies; j++) {
    if (picked[j] || !movies[j].title)
      continue;

    float score = predict_for_new_user(model, user_profile, j);

    candidates[cand_count].score = score;
    candidates[cand_count].id = j;
    cand_count++;
  }

  qsort(candidates, cand_count, sizeof(Candidate), compare_candidates);

  printf("Based on your ratings:\n");
  for (int i = 0; i < num_user_ratings; i++) {
    int mid = user_ratings[i].movie_id;
    printf("  • %s - %.1f stars\n", movies[mid].title, user_ratings[i].rating);
  }
  printf("\n");

  printf("Top 10 Recommendations:\n\n");
  int top_n = (cand_count < 10) ? cand_count : 10;
  for (int i = 0; i < top_n; i++) {
    int mid = candidates[i].id;
    printf("  %2d. %s\n", i + 1, movies[mid].title);
    printf("      Predicted rating: %.2f ⭐\n", candidates[i].score);
    if (movies[mid].genres) {
      printf("      Genres: %s\n", movies[mid].genres);
    }
    printf("\n");
  }

  free(user_profile);
  free(candidates);
  free(user_ratings);
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