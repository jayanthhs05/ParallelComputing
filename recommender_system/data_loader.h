#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "data_structures.h"

Dataset *load_dataset(const char *filename, int rank);
void free_dataset(Dataset *dataset);
IDMapper *create_id_mapper(Dataset *dataset);
void free_id_mapper(IDMapper *mapper);
void remap_ids(Dataset *dataset, IDMapper *mapper);
void split_data(Dataset *dataset, Dataset **train, Dataset **test,
                float split_ratio, int rank);

#endif
