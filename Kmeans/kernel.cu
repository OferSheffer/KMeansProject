#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <stdio.h>
#include "Kmeans.h"


#define CHKMAL_ERROR	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
#define CHKMEMCPY_ERROR if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
#define CHKSYNC_ERROR	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize failed! Error code %d\n", cudaStatus); goto Error; }

// arrSize indices; THREADS_PER_BLOCK * NO_BLOCKS total threads;
// Each thread in charge of THREAD_BLOCK_SIZE contigeous indices
     
#define THREADS_PER_BLOCK 1024

/*
__device__ void reduce(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>32; s >>= 1)
	{
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid < 32) //unroll warp
	{
		sdata[tid] += sdata[tid + 32];
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		sdata[tid] += sdata[tid + 1];
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
} */

/*
__device__ void reduce(bool *d_kaFlags) {
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
	d_kaFlags[tid] = d_kaFlags[i] | d_kaFlags[i + blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>32; s >>= 1)
	{
		if (tid < s)
			d_kaFlags[tid] |= d_kaFlags[tid + s];
		__syncthreads();
	}
	if (tid < 32) //unroll warp
	{
		d_kaFlags[tid] += d_kaFlags[tid + 32];
		d_kaFlags[tid] += d_kaFlags[tid + 16];
		d_kaFlags[tid] += d_kaFlags[tid + 8];
		d_kaFlags[tid] += d_kaFlags[tid + 4];
		d_kaFlags[tid] += d_kaFlags[tid + 2];
		d_kaFlags[tid] += d_kaFlags[tid + 1];
	}
} */

__global__ void reClusterWithCuda(xyArrays* d_kCenters, const int ksize, xyArrays* d_xya, int* pka, bool* d_kaFlags, const int size)
{
	__shared__ bool* dShared_kaFlags; // array to flag changes in point-to-cluster association

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// for every point: save idx where min(distance from k[idx]	
	if (tid < size)
	{
		dShared_kaFlags[tid] = false; // no changes yet
		int prevPka = pka[tid]; // save associated cluster idx
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
			d_kaFlags[tid] = true;
		}
		// reduction for d_kaFlag
		__syncthreads();
		// do reduction in shared mem
		//reduce(dShared_kaFlags);
		// each thread loads one element from global to shared mem
		unsigned int ltid = threadIdx.x;
		unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
		dShared_kaFlags[ltid] = dShared_kaFlags[i] | dShared_kaFlags[i + blockDim.x];
		__syncthreads();
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
		if (ltid == 0) d_kaFlags[ltid] = dShared_kaFlags[0];
	}
}

// Helper function for finding best centers for ksize clusters
cudaError_t kCentersWithCuda(xyArrays* kCenters, int ksize, xyArrays* xya, int* pka, long N, int LIMIT)
{
	cudaError_t cudaStatus;
	const int NO_BLOCKS = (N % THREADS_PER_BLOCK == 0)? N / THREADS_PER_BLOCK : N / THREADS_PER_BLOCK + 1;
	const int THREAD_BLOCK_SIZE = THREADS_PER_BLOCK;
	/*
	if (N % (THREADS_PER_BLOCK * NO_BLOCKS) != 0) {
		fprintf(stderr, "reClusterWithCuda launch failed:\n"
			"Array size (%d) modulo Total threads (%d) != 0.\n"
			"Try changing number of threads.\n", N, (THREADS_PER_BLOCK * NO_BLOCKS));
		goto Error;
	} */
	initK(ksize);				// K-centers = first points in data (on host)

	// memory init block
	//{
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			goto Error;
		}

		// size helpers
		size_t nDataBytes = sizeof(xya);
		size_t nKCenterBytes = sizeof(kCenters);

		// allocate host-side helpers
		bool h_kaFlag;

		// allocate device memory
		xyArrays *d_xya,
				 *d_kCenters;				// data and k-centers xy information
		int		 *d_pka;					// array to associate xya points with their closest cluster
		bool     *d_kaFlags;				// array to flags changes in point-to-cluster association							

		cudaMalloc((xyArrays**)&d_xya, nDataBytes); CHKMAL_ERROR;
		cudaMalloc((xyArrays**)&d_kCenters, nKCenterBytes); CHKMAL_ERROR;
		cudaMalloc((int**)&d_pka, N * sizeof(int)); CHKMAL_ERROR;
		cudaMalloc((bool**)&d_kaFlags, N * sizeof(bool)); CHKMAL_ERROR;

		// copy data from host to device
		cudaMemcpy(d_xya, xya, nDataBytes, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
		cudaMemcpy(d_kCenters, kCenters, nKCenterBytes, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
		cudaMemcpy(d_pka, pka, N*sizeof(int), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;

		cudaStatus = cudaMemset((void*)d_kaFlags, 0, N * sizeof(bool));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemset failed!\n");
			goto Error;
		}
	//}

	// *** phase 1 ***
	int iter = 0;
	size_t SharedMemBytes = N * sizeof(bool); // shared memory for flag work
	do {
		//KernelFunc << <DimGrid, DimBlock, SharedMemBytes >> >
		reClusterWithCuda << <NO_BLOCKS, THREADS_PER_BLOCK, SharedMemBytes >> > (d_kCenters, ksize, d_xya, d_pka, d_kaFlags, N); // THREADS_PER_BLOCK, THREAD_BLOCK_SIZE
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "reClusterWithCuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaStatus = cudaDeviceSynchronize(); CHKSYNC_ERROR;
	
		//cudaStatus = cudaMemcpy((void**)&h_kaFlag, *d_kaFlags, sizeof(bool), cudaMemcpyDeviceToHost); CHKMEMCPY_ERROR;
		
	} while (++iter < LIMIT && d_kaFlags[0]);  // association changes: need to re-cluster



	//TODO quick test
	for (int i = 0; i < ksize; i++)
	{
		printf("%d, %f, %f\n", i, kCenters->x[i], kCenters->y[i]);
	}

	Error:
		cudaFree(d_xya);
		cudaFree(d_kCenters);
		cudaFree(d_pka);
		cudaFree(d_kaFlags);

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