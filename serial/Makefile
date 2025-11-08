CC = gcc
CFLAGS = -O3 -Wall -std=c99
LDFLAGS = -lm

TARGET = recommender
OBJS = main.o data_loader.o model.o train.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
