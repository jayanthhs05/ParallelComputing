#!/bin/bash

DATA_FILE="../data/ratings.csv"
NUM_PROCS=${1:-4}

if [ ! -f "$DATA_FILE" ]; then
    echo "Error: $DATA_FILE not found"
    echo "Please ensure the ratings.csv file is in the ../data/ directory"
    exit 1
fi

echo "Cleaning old files"
rm -f model.bin movie_mapping.bin train_save *.o

echo "Building train_save"
make train_save

if [ ! -f "train_save" ]; then
    echo "Error: Build failed"
    echo "Check for compilation errors above"
    exit 1
fi

echo "Training model with $NUM_PROCS processes"
echo "This may take a few minutes"
mpirun -np $NUM_PROCS ./train_save "$DATA_FILE"

if [ $? -ne 0 ]; then
    echo "Error: Training failed with exit code $?"
    exit 1
fi

if [ -f "model.bin" ] && [ -f "movie_mapping.bin" ]; then
    echo ""
    echo "Training complete!"
    echo "Model saved to: model.bin"
    echo "Mapping saved to: movie_mapping.bin"
    echo ""
    echo "Model file size: $(ls -lh model.bin | awk '{print $5}')"
    echo "Mapping file size: $(ls -lh movie_mapping.bin | awk '{print $5}')"
    echo ""
    echo "You can now run: ./run.sh"
else
    echo "Error: Training completed but model files were not created"
    echo "Check the output above for errors"
    exit 1
fi