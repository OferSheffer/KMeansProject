#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#pragma warning(disable:4996)

// SoA: reduce load/store operations
typedef struct _xyArrays {
	float *x;
	float *y;
} xyArrays;

//cudaError_t kCentersWithCuda(xyArrays* kCenters, xyArrays* xya, long N, int ksize);
cudaError_t kCentersWithCuda(xyArrays* kCenters, xyArrays* xya, long N, int ksize);
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

void readPointsFromFile();
void ompGo();





void populateSoA(FILE* fp);


bool reCluster(int ksize);

void mallocSoA(xyArrays** soa, long size);
void freeSoA(xyArrays* soa);
void initK(long ksize);
void prepK(int* ompCntPArr, long ksize);
void initClusterAssociationArrays();
void getNewPointKCenterAssociation(long i, int ksize);
void getNewPointKCenterAssociationXYBarrier(long i, int ksize);