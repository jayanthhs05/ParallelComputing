#!/bin/bash

MODEL_FILE="model.bin"
MAPPING_FILE="movie_mapping.bin"
MOVIES_FILE="../data/movies.csv"

echo "Current directory: $(pwd)"
echo ""

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: $MODEL_FILE not found in current directory"
    echo "Available files:"
    ls -la *.bin 2>/dev/null || echo "  No .bin files found"
    echo ""
    echo "Please run './train.sh' first to train the model"
    exit 1
fi

if [ ! -f "$MAPPING_FILE" ]; then
    echo "Error: $MAPPING_FILE not found"
    echo "Please run './train.sh' first to train the model"
    exit 1
fi

echo "Found model files:"
echo "  - model.bin ($(ls -lh model.bin | awk '{print $5}'))"
echo "  - movie_mapping.bin ($(ls -lh movie_mapping.bin | awk '{print $5}'))"
echo ""

if [ ! -f "$MOVIES_FILE" ]; then
    echo "Error: $MOVIES_FILE not found"
    echo "Please ensure movies.csv is in the ../data/ directory"
    exit 1
fi

if [ ! -d "data" ]; then
    echo "Creating data directory"
    mkdir -p data
fi

if [ ! -f "data/movies.csv" ]; then
    echo "Copying movies.csv to data/"
    cp "$MOVIES_FILE" data/movies.csv
fi

if [ ! -f "recommend" ] || [ "recommend.c" -nt "recommend" ] || [ "model.c" -nt "recommend" ]; then
    echo "Building recommend"
    rm -f *.o recommend
    make recommend
else
    echo "recommend binary is up to date"
fi

if [ ! -f "recommend" ]; then
    echo "Error: Build failed"
    echo "Run 'make recommend' manually to see errors"
    exit 1
fi

echo "Build successful"
echo ""
echo "  Movie Recommender System"
echo ""

./recommend