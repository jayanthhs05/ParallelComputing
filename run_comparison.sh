#!/bin/bash

DATA_FILE="$1"
NUM_PROCESSES=${2:-8}
NUM_THREADS=${3:-2}

if [ ! -f "$DATA_FILE" ]; then
    echo "Error: File $DATA_FILE not found"
    exit 1
fi

cp "$DATA_FILE" serial/
cp "$DATA_FILE" parallel/

echo "Building serial"
cd serial
make clean > /dev/null 2>&1
make > /dev/null 2>&1
cd ..

echo "Building parallel"
cd parallel
make clean > /dev/null 2>&1
make > /dev/null 2>&1
cd ..

echo "Running serial"
cd serial
./recommender "$(basename $DATA_FILE)" > ../serial_output.txt 2>&1
cd ..
SERIAL_TIME=$(grep "EXECUTION_TIME:" serial_output.txt | awk '{print $2}')
SERIAL_RMSE=$(grep "Test RMSE:" serial_output.txt | awk '{print $3}')

echo "Running parallel"
export OMP_NUM_THREADS=$NUM_THREADS
cd parallel
mpirun -np $NUM_PROCESSES ./recommender "$(basename $DATA_FILE)" > ../parallel_output.txt 2>&1
cd ..
PARALLEL_TIME=$(grep "Total execution time:" parallel_output.txt | awk '{print $4}')
PARALLEL_RMSE=$(grep "Test RMSE:" parallel_output.txt | head -1 | awk '{print $3}')

SPEEDUP=$(echo "scale=2; $SERIAL_TIME / $PARALLEL_TIME" | bc)

echo "Serial Time: $SERIAL_TIME s"
echo "Parallel Time: $PARALLEL_TIME s"
echo "Speedup: ${SPEEDUP}x"
echo "Serial RMSE: $SERIAL_RMSE"
echo "Parallel RMSE: $PARALLEL_RMSE"

echo "Version,Time,RMSE,Speedup" > comparison_results.csv
echo "Serial,$SERIAL_TIME,$SERIAL_RMSE,1.00" >> comparison_results.csv
echo "Parallel,$PARALLEL_TIME,$PARALLEL_RMSE,$SPEEDUP" >> comparison_results.csv
