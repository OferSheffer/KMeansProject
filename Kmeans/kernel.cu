
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "Kmeans.h"

#define CHKMAL_ERROR	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
#define CHKMEMCPY_ERROR if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
#define CHKSYNC_ERROR	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize failed! Error code %d\n", cudaStatus); goto Error; }

// arrSize indices; THREADS_PER_BLOCK * NO_BLOCKS total threads;
// Each thread in charge of THREAD_BLOCK_SIZE contigeous indices
     
#define THREADS_PER_BLOCK 1000

__global__ void reClusterWithCuda(xyArrays* d_kCenters, xyArrays* d_xya, const int size, bool *kaFlags)
{
	extern __shared__ bool* d_kaFlags; // array to flag changes in point-to-cluster association

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	d_kaFlags[i] = false;
	
	// for every point: save idx where min(distance from k[idx])
	//#pragma omp parallel for reduction(|:kAssociationChangedFlag)
	
	if (i < size) {
		int prevPka = d_kCenters[i]; // save associated cluster idx
		for (int idx = 0; idx < size; idx++)
		{

		} 
		
		tmpx = data->x[i];
		float tmpy = data->y[i];
		tmpx += 10.f;
		tmpy += 20.f;
		result->x[i] = tmpx;
		result->y[i] = tmpy;
	}



		int prevPka = pka[i];  // save associated cluster idx
		getNewPointKCenterAssociation(i, size);
		if (pka[i] != prevPka)
		{
			kaFlag = true;
		}



	//c[i] = a[i] + b[i];
}

// Helper function for finding best centers for ksize clusters
cudaError_t kCentersWithCuda(xyArrays* kCenters, xyArrays* xya, long N, int ksize, int LIMIT)
{
	cudaError_t cudaStatus; 
	const int NO_BLOCKS = N / THREADS_PER_BLOCK;
	const int THREAD_BLOCK_SIZE = N / (THREADS_PER_BLOCK * NO_BLOCKS);
	if (N % (THREADS_PER_BLOCK * NO_BLOCKS) != 0) {
		fprintf(stderr, "reClusterWithCuda launch failed:\n"
			"Array size (%d) modulo Total threads (%d) != 0.\n"
			"Try changing number of threads.\n", N, (THREADS_PER_BLOCK * NO_BLOCKS));
		goto Error;
	}

	// memory init block
	{
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			goto Error;
		}

		size_t nDataBytes = sizeof(xya);
		size_t nKCenterBytes = sizeof(kCenters);

		// allocate device memory
		xyArrays *d_a, *d_k;		// data and k-centers xy information
		int* d_pka;
		
									//bool *d_kaFlags;			// array to flag changes in point-to-cluster association

		cudaMalloc((xyArrays**)&d_a, nDataBytes); CHKMAL_ERROR;
		cudaMalloc((xyArrays**)&d_k, nKCenterBytes); CHKMAL_ERROR;
		cudaMalloc((int**)&d_pka, N * sizeof(int)); CHKMAL_ERROR;
		//cudaMalloc((bool**)&d_kaFlags, N * sizeof(bool)); CHKMAL_ERROR;

		initK(ksize);				// K-centers = first points in data (on host)

									// copy data from host to device
		cudaMemcpy(d_a, xya, nDataBytes, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
		cudaMemcpy(d_k, kCenters, nKCenterBytes, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;

		//cudaStatus = cudaMemset((void*)dev_threadedHist, 0, THREADS_PER_BLOCK * NO_BLOCKS * histSize * sizeof(int));
		cudaStatus = cudaMemset((void*)d_kaFlags, 0, N * sizeof(bool));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemset failed!\n");
			goto Error;
		}
	}
	

	// *** phase 1 ***
	// One thread for every THREAD_BLOCK_SIZE elements.

	reClusterWithCuda << <NO_BLOCKS, THREADS_PER_BLOCK >> >(d_k, d_a, d_kaFlags, THREADS_PER_BLOCK, THREAD_BLOCK_SIZE);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "threadedHistKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize(); CHKSYNC_ERROR;
	

	
	

	
	int iter = 0;
	bool kAssociationChangedFlag = true;
	do {
		//printf("iter %d\n", iter + 1);
		reClusterWithCuda(ksize, kAssociationChangedFlag);
	} while (++iter < LIMIT && kAssociationChangedFlag);  // association changes: need to re-cluster

	//TODO quick test
	for (int i = 0; i < ksize; i++)
	{
		printf("%d, %f, %f\n", i, kCenters->x[i], kCenters->y[i]);
	}

	//float x = input[threadID];
	//float y = func(x);
	//output[threadID] = y;
	Error:
		cudaFree(d_a);
		cudaFree(d_k);

		return cudaStatus;
}




/**************/
//old cuda code
/*
	__global__ void addKernel(int *c, const int *a, const int *b)
	{
		int i = threadIdx.x;
		c[i] = a[i] + b[i];
	}

	// Helper function for using CUDA to add vectors in parallel.
	cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
	{
		int *dev_a = 0;
		int *dev_b = 0;
		int *dev_c = 0;
		cudaError_t cudaStatus;

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			goto Error;
		}

		// Allocate GPU buffers for three vectors (two input, one output)    .
		cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		// Launch a kernel on the GPU with one thread for each element.
		addKernel << <1, size >> >(dev_c, dev_a, dev_b);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

	Error:
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);

		return cudaStatus;
	}

	*/