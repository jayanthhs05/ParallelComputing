# Movie Recommender System

A parallel implementation of a matrix factorization-based movie recommendation system using MPI and OpenMP. The system learns user preferences from historical ratings and generates personalized movie recommendations.

## Project Structure

```
ParallelRecommender/
├── serial/              # Serial implementation baseline
├── parallel/            # MPI+OpenMP parallel implementation
├── recommender_system/  # Interactive recommendation system
└── run_comparison.sh    # Performance comparison script
```

## Requirements

- GCC compiler
- MPI implementation (OpenMPI or MPICH)
- OpenMP support
- Make build system

## Installation

1. Clone or download the repository
2. Ensure MPI is installed on your system
3. Prepare your dataset in CSV format with columns: userId, movieId, rating, timestamp

## Usage

### Training and Testing

#### Serial Version
```bash
cd serial
make
./recommender <path_to_ratings.csv>
```

#### Parallel Version
```bash
cd parallel
make
mpirun -np <num_processes> ./recommender <path_to_ratings.csv>
```

Set OpenMP threads before running:
```bash
export OMP_NUM_THREADS=<num_threads>
```

### Interactive Recommendation System

The recommender system allows users to select movies they like and receive personalized recommendations.

#### Training the Model
```bash
cd recommender_system
./train.sh [num_processes]
```
Default processes: 4

#### Running Recommendations
```bash
./run.sh
```

The system will:
1. Load the trained model
2. Prompt for movie title searches
3. Generate top 10 personalized recommendations based on selected movies

### Performance Comparison

Compare serial and parallel implementations:
```bash
./run_comparison.sh <path_to_ratings.csv> [num_processes] [num_threads]
```

Defaults: 8 processes, 2 threads per process

Results are saved to `comparison_results.csv`

## Configuration

Model hyperparameters can be adjusted in `config.h`:

```c
#define NUM_FACTORS 50           // Latent feature dimensions
#define LEARNING_RATE 0.001      // SGD learning rate
#define REGULARIZATION 0.01      // L2 regularization
#define NUM_ITERATIONS 50        // Training epochs
#define TRAIN_TEST_SPLIT 0.8     // Train/test ratio
```

## Algorithm

The system implements matrix factorization using Stochastic Gradient Descent (SGD) to decompose the user-movie rating matrix into latent feature vectors. The parallel implementation distributes training data across MPI processes and uses periodic synchronization to reduce communication overhead.

## Output

Training output includes:
- Dataset statistics
- Training progress
- Performance metrics (computation/communication time breakdown)
- Test RMSE (Root Mean Square Error)
- Total execution time

## Cleaning

Remove compiled binaries and object files:
```bash
cd serial && make clean
cd parallel && make clean
cd recommender_system && make clean
```

Remove model files:
```bash
cd recommender_system && make clean-all
```

## Data Format

Expected CSV format:
```
userId,movieId,rating,timestamp
1,2,3.5,1112486027
1,29,3.5,1112484676
```

For the interactive system, also provide `movies.csv`:
```
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
```

## Performance Notes

- The parallel implementation uses adaptive synchronization intervals based on the number of processes
- Communication overhead is minimized through batched parameter updates
- OpenMP threads parallelize local computations within each MPI process