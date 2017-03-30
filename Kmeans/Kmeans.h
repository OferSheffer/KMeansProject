#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#pragma warning(disable:4996)

#define FF fflush(stdout);
//#define _DEBUG1 // temp values
//#define _DEBUG2 // low level progress
//#define _DEBUG3 // omp k-centers results -- for comparison
#define _DEBUG4 // mid level values test -- diameters final
#define _DEBUG5 // mid level values test

// SoA: reduce load/store operations
typedef struct _xyArrays {
	float *x;
	float *y;
} xyArrays;

cudaError_t kCentersWithCuda(xyArrays* kCenters, int ksize, xyArrays* xya, int* pka, long N, int LIMIT);
cudaError_t kDiametersWithCuda(float* kDiameters, int ksize, xyArrays* xya, int* pka, long N, int myid, int numprocs);

void readPointsFromFile();
void ompGoTest(int initSize, int maxSize);


bool ompReduceCudaFlags(bool* flags, int size);
void ompRecenterFromCuda(int ksize);
void ompMaxVectors(float** kDiameters, float* kDiametersTempAnswer, int ksize);

void populateSoA(FILE* fp);


bool reCluster(int ksize);

void mallocSoA(xyArrays** soa, long size);
void freeSoA(xyArrays* soa);
void initK(long ksize);
void prepK(int* ompCntPArr, long ksize);
void initClusterAssociationArrays();
void getNewPointKCenterAssociation(long i, int ksize);
void getNewPointKCenterAssociationXYBarrier(long i, int ksize);
void loopBcast(long size);
void serialSendRecv(long size);
int* initJobArray(int NO_BLOCKS, int fact);

void printArrTestPrint(int myid, float* arr, int size, const char* arrName);