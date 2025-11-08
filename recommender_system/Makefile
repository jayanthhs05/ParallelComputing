CC = mpicc
GCC = gcc
CFLAGS = -O3 -Wall -std=c99
LDFLAGS = -lm

TRAIN_SAVE_OBJS = train_save.o data_loader.o model.o train.o

train_save: $(TRAIN_SAVE_OBJS)
	$(CC) $(CFLAGS) -o train_save $(TRAIN_SAVE_OBJS) $(LDFLAGS)

RECOMMEND_OBJS = recommend.o model_standalone.o movies.o

recommend: $(RECOMMEND_OBJS)
	$(GCC) $(CFLAGS) -o recommend $(RECOMMEND_OBJS) $(LDFLAGS)

train_save.o: train_save.c
	$(CC) $(CFLAGS) -c train_save.c

data_loader.o: data_loader.c
	$(CC) $(CFLAGS) -c data_loader.c

model.o: model.c
	$(CC) $(CFLAGS) -c model.c

train.o: train.c
	$(CC) $(CFLAGS) -c train.c

recommend.o: recommend.c
	$(GCC) $(CFLAGS) -c recommend.c

model_standalone.o: model.c
	$(GCC) $(CFLAGS) -c model.c -o model_standalone.o

movies.o: movies.c
	$(GCC) $(CFLAGS) -c movies.c

clean:
	rm -f *.o train_save recommend

clean-all:
	rm -f *.o train_save recommend model.bin movie_mapping.bin

.PHONY: clean clean-all train_save recommend