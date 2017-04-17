#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#pragma warning(disable:4996)

#define FF fflush(0)
#define DDD { printf("****DDD!****\n"); FF; }

//**********************************
// GPU DEFINITIONS
//***********************************
#define BASE_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32
#define NUM_CONCUR_KERNELS 16	// concurrent kernels

void initializeWithGpuReduction();

//**********************************
//#define _RUNAFEKA
//#define _DEBUGV						// verbose
//#define _PROFILE_BLOCKS_TO_JOBS		// print #blocks and #jobs
#define _TIME						// time output
#define _TIMEK						// time per ksize
//#define _PROF_DIAM_BLOCK_KERNEL		// time diametersCuda kernel operation on 0,0 (0.008*5050 ~= 40sec)



//#define _DEBUGPOINTSREADFROMFILE		// xya values read from 
//#define _DEBUGNVAL					// verify N in malloc
//#define _DEBUGJOBS					// which process is working on which job block
//#define _DEBUG1						// temp kDiameters values
//#define _DEBUG2						// low level progress
//#define _DEBUG4						// diameters final for ksize
#define _DEBUG5						// QM values test

#ifndef INFINITY
#define INFINITY 1126000000000000
#define _MANUAL_INFINITY
#endif

// SoA: reduce load/store operations
typedef struct _xyArrays {
	float *x;
	float *y;
} xyArrays;

cudaError_t kCentersWithCuda(xyArrays* kCenters, int ksize, xyArrays* xya, int* pka, long N, int LIMIT, int THREADS_PER_BLOCK);
cudaError_t kDiametersWithCuda(float* kDiameters, int ksize, xyArrays* xya, int* pka, long N, int myid, int numprocs, int THREADS_PER_BLOCK);

void readPointsFromFile();

bool ompReduceCudaFlags(bool* flags, int size);
void ompRecenterFromCuda(int ksize);
void ompMaxVectors(float** kDiameters, float* kDiametersTempAnswer, int ksize);

void populateSoA(FILE* fp);


void mallocSoA(xyArrays** soa, long size);
void freeSoA(xyArrays* soa);
void initK(long ksize);
void prepK(int* ompCntPArr, long ksize);
void initClusterAssociationArrays();
void getNewPointKCenterAssociation(long i, int ksize);
int* initJobArray(int NO_BLOCKS, int fact);

void printArrTestPrint(int myid, float* arr, int size, const char* arrName);

void initArrToZeroes(float** arr, int size);
