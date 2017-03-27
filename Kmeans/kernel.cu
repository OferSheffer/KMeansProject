#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "Kmeans.h"


#define CHKMAL_ERROR	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
#define CHKMEMCPY_ERROR if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
#define CHKSYNC_ERROR	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize failed! Error code %d\n", cudaStatus); goto Error; }

// arrSize indices; THREADS_PER_BLOCK * NO_BLOCKS total threads;
// Each thread in charge of THREAD_BLOCK_SIZE contigeous indices

#define THREADS_PER_BLOCK 1024

__global__ void reClusterWithCuda(xyArrays* d_kCenters, const int ksize, xyArrays* d_xya, int* pka, bool* d_kaFlags, const int size)
{
	__shared__ bool dShared_kaFlags[1024]; // array to flag changes in point-to-cluster association

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int prevPka;
	// for every point: save idx where min(distance from k[idx]	
	if (tid < size)
	{
		dShared_kaFlags[tid] = false; // no changes yet
		prevPka = pka[tid]; // save associated cluster idx
		float minSquareDist = INFINITY;
		float curSquareDist;
		for (int idx = 0; idx < ksize; idx++)
		{
			curSquareDist = powf(d_xya->x[tid] - d_kCenters->x[idx], 2) + powf(d_xya->y[tid] - d_kCenters->y[idx], 2);
			if (curSquareDist < minSquareDist)
			{
				minSquareDist = curSquareDist;
				pka[tid] = idx;
			}
		}
		if (pka[tid] != prevPka)
		{
			dShared_kaFlags[tid] = true;
		}
		// reduction for d_kaFlag
		__syncthreads();
		// do reduction in shared mem
		//reduce(dShared_kaFlags);
		// each thread loads one element from global to shared mem
		unsigned int ltid = threadIdx.x;
#if 0
		unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
		blockIdx.x * blockDim.x + threadIdx.x;

		if (i < size) dShared_kaFlags[ltid] = dShared_kaFlags[i] | dShared_kaFlags[i + blockDim.x];
		__syncthreads();
#endif
		// do reduction in shared mem
		for (unsigned int s = blockDim.x / 2; s>32; s >>= 1)
		{
			if (tid < s)
				dShared_kaFlags[ltid] |= dShared_kaFlags[ltid + s];
			__syncthreads();
		}
		if (ltid < 32) //unroll warp
		{
			dShared_kaFlags[ltid] += dShared_kaFlags[ltid + 32];
			dShared_kaFlags[ltid] += dShared_kaFlags[ltid + 16];
			dShared_kaFlags[ltid] += dShared_kaFlags[ltid + 8];
			dShared_kaFlags[ltid] += dShared_kaFlags[ltid + 4];
			dShared_kaFlags[ltid] += dShared_kaFlags[ltid + 2];
			dShared_kaFlags[ltid] += dShared_kaFlags[ltid + 1];
		}

		// write result for this block to global mem
		if (tid == 0) d_kaFlags[blockIdx.x] = dShared_kaFlags[0];
	}
}

// Helper function for finding best centers for ksize clusters
cudaError_t kCentersWithCuda(xyArrays* kCenters, int ksize, xyArrays* xya, int* pka, long N, int LIMIT)
{
	cudaError_t cudaStatus;
	const int NO_BLOCKS = (N % THREADS_PER_BLOCK == 0) ? N / THREADS_PER_BLOCK : N / THREADS_PER_BLOCK + 1;
	const int THREAD_BLOCK_SIZE = THREADS_PER_BLOCK;
	/*
	if (N % (THREADS_PER_BLOCK * NO_BLOCKS) != 0) {
	fprintf(stderr, "reClusterWithCuda launch failed:\n"
	"Array size (%d) modulo Total threads (%d) != 0.\n"
	"Try changing number of threads.\n", N, (THREADS_PER_BLOCK * NO_BLOCKS));
	goto Error;
	} */
	initK(ksize);				// K-centers = first points in data (on host)
	int iter = 0;
	size_t SharedMemBytes = N * sizeof(bool); // shared memory for flag work
	bool flag;

	// memory init block
	//{
	size_t nDataBytes = N * sizeof(*xya);
	size_t nKCenterBytes = ksize * sizeof(*kCenters);
	bool	 *h_kaFlags;
	int	 *d_pka;					// array to associate xya points with their closest cluster
	bool     *d_kaFlags;				// array to flags changes in point-to-cluster association	
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// size helpers

	// allocate host-side helpers
	h_kaFlags = (bool*)malloc(NO_BLOCKS * sizeof(bool));

	// allocate device memory
	xyArrays *d_xya,
		*d_kCenters;				// data and k-centers xy information
	xyArrays h_xya, h_kCenters;

	printf("A");
	cudaMalloc(&d_xya, sizeof(xyArrays));

	cudaMalloc(&(h_xya.x), nDataBytes / 2); CHKMAL_ERROR;
	cudaMalloc(&(h_xya.y), nDataBytes / 2); CHKMAL_ERROR;
	cudaMemcpy(h_xya.x, xya->x, nDataBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
	cudaMemcpy(h_xya.y, xya->y, nDataBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;


	printf("A");
	cudaMalloc(&d_kCenters, sizeof(xyArrays));
	cudaMalloc(&(h_kCenters.x), nKCenterBytes / 2); CHKMAL_ERROR;
	cudaMalloc(&(h_kCenters.y), nKCenterBytes / 2); CHKMAL_ERROR;
	cudaMemcpy(h_kCenters.x, kCenters->x, nKCenterBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
	cudaMemcpy(h_kCenters.y, kCenters->y, nKCenterBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
	cudaMemcpy(d_xya, &h_xya, sizeof(xyArrays), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
	cudaMemcpy(d_kCenters, &h_kCenters, sizeof(xyArrays), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
	printf("A");


	//cudaMalloc((xyArrays**)&d_xya, nDataBytes); CHKMAL_ERROR;
	//cudaMalloc((xyArrays**)&d_kCenters, nKCenterBytes); CHKMAL_ERROR;
	cudaMalloc(&d_pka, N * sizeof(int)); CHKMAL_ERROR;
	cudaMalloc(&d_kaFlags, N * sizeof(bool)); CHKMAL_ERROR;
	printf("B");

	// copy data from host to device
	cudaMemcpy(d_pka, pka, N * sizeof(int), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;


	cudaStatus = cudaMemset((void*)d_kaFlags, 0, N * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!\n");
		goto Error;
	}
	//}

	// *** phase 1 ***
	do {
		//KernelFunc << <DimGrid, DimBlock, SharedMemBytes >> >
		reClusterWithCuda << <NO_BLOCKS, THREADS_PER_BLOCK, SharedMemBytes >> > (d_kCenters, ksize, d_xya, d_pka, d_kaFlags, N); // THREADS_PER_BLOCK, THREAD_BLOCK_SIZE
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "reClusterWithCuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaStatus = cudaDeviceSynchronize(); CHKSYNC_ERROR;


		cudaStatus = cudaMemcpy(h_kaFlags, d_kaFlags, NO_BLOCKS * sizeof(bool), cudaMemcpyDeviceToHost); CHKMEMCPY_ERROR;
		printf("iteration %d\n", iter);
		
		flag = ompReduceCudaFlags(h_kaFlags, NO_BLOCKS);

	} while (++iter < LIMIT && flag);  // association changes: need to re-cluster


	cudaMemcpy(kCenters->x, h_kCenters.x, nKCenterBytes / 2, cudaMemcpyDeviceToHost); CHKMEMCPY_ERROR;
	cudaMemcpy(kCenters->y, h_kCenters.y, nKCenterBytes / 2, cudaMemcpyDeviceToHost); CHKMEMCPY_ERROR;

	//TODO quick test
	for (int i = 0; i < ksize; i++)
	{
		printf("%d, %f, %f\n", i, kCenters->x[i], kCenters->y[i]);
	}

	free(h_kaFlags);

Error:
	cudaFree(d_xya);
	cudaFree(d_kCenters);
	cudaFree(d_pka);
	cudaFree(d_kaFlags);

	return cudaStatus;
}