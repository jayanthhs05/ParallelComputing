CC = mpicc
CFLAGS = -O3 -fopenmp -Wall -std=c99
LDFLAGS = -lm -fopenmp
TARGET = recommender
OBJS = main.o data_loader.o model.o train.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o
