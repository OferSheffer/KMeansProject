#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#pragma warning(disable:4996)

// SoA: reduce load/store operations
typedef struct _xyArrays {
	float *x;
	float *y;
} xyArrays;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

void readPointsFromFile();
void mallocSoA(xyArrays** soa, long size);
void freeSoA(xyArrays* soa);
void initK(long ksize);
