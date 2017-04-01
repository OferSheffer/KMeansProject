
#include "Kmeans.h"


#define CHKMAL_ERROR	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); FF; goto Error; }
#define CHKMEMCPY_ERROR if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); FF; goto Error; }
#define CHKSYNC_ERROR	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize failed! Error code %d\n", cudaStatus); FF; goto Error; }
#define EVENT_ERROR		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaEventOperation failed! Error code %d\n", cudaStatus); FF; goto Error; }

// arrSize indices; THREADS_PER_BLOCK * NO_BLOCKS total threads;
// Each thread in charge of THREAD_BLOCK_SIZE contigeous indices

#define MASTER 0
#define NEW_JOB 0
#define STOP_WORKING 1

#ifndef _WEAKGPU
#define THREADS_PER_BLOCK 1024  // replacement for THREAD_BLOCK_SIZE or blockDim.x
#else
#define THREADS_PER_BLOCK 128	// weak gpu
#endif // not _WEAKGPU

__global__ void reClusterWithCuda(xyArrays* d_kCenters, const int ksize, xyArrays* d_xya, int* pka, bool* d_kaFlags, const int size)
{
	__shared__ bool dShared_kaFlags[THREADS_PER_BLOCK]; // array to flag changes in point-to-cluster association

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ltid = threadIdx.x;
	int prevPka;
	// for every point: save idx where min(distance from k[idx]	
	if (tid < size)
	{
		dShared_kaFlags[ltid] = false; // no changes yet
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
			dShared_kaFlags[ltid] = true;
		}
		__syncthreads();
		
		// reduction in shared mem
		for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
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

// Helper for kDiamBlockWithCuda
__device__ void AtomicMax(float * const address, const float value)
{
	if (*address >= value)
	{
		return;
	}

	int * const address_as_i = (int *)address;
	int old = *address_as_i, assumed;

	do
	{
		assumed = old;
		if (__int_as_float(assumed) >= value)
		{
			break;
		}

		old = atomicCAS(address_as_i, assumed, __float_as_int(value));
	} while (assumed != old);
}


//Note: will be x2 faster with smaller blocks -- but will require (^2/numproc) runs
__global__ void kDiamBlockWithCuda(float* kDiameters, const int ksize, xyArrays* d_xya, int* pka, const int size, const int blkAIdx, const int blkBIdx)
{
	// if (blockIdx.x != blkAIdx) return;
	__shared__ float dShared_SquaredXYAB[THREADS_PER_BLOCK * 4]; // save squared values for reuse

	// local shared mem speedup - save squared values for reuse
	// diameter^2 = (XA-XB)^2 + (YA-YB)^2 = XA^2+XB^2+YA^2+YB^2  -2*XA*XB -2*YA*YB
	unsigned int tidA = blkAIdx * blockDim.x + threadIdx.x;
	if (tidA < size)
	{
		unsigned int tidB = blkBIdx * blockDim.x + threadIdx.x;
		dShared_SquaredXYAB[4 * threadIdx.x + 0] = powf(d_xya->x[tidA], 2);	// i%4==0: x^2 of blkA
		dShared_SquaredXYAB[4 * threadIdx.x + 2] = powf(d_xya->y[tidA], 2);	// i%4==2: y^2 of blkA
		if (tidB < size)
		{
			dShared_SquaredXYAB[4 * threadIdx.x + 1] = powf(d_xya->x[tidB], 2);	// i%4==1: x^2 of blkB
			dShared_SquaredXYAB[4 * threadIdx.x + 3] = powf(d_xya->y[tidB], 2);	// i%4==3: y^2 of blkB
		}
		__syncthreads();

		float max = 0;
		float cur;
		int myK = pka[tidA];

		// OPTION A
		//if (ksize <= 3)
		if (ksize <= 0)
		{
			//TODO: consider load blancing/loop unrolling of some sort
			// run kernel with a single block, use external block indices to syncronize operations
			for (int tidO, threadBRunningIdx = 0; threadBRunningIdx < blockDim.x; threadBRunningIdx++)
			{
				// prevent repeated calculations
				if (threadIdx.x < threadBRunningIdx)
				{
					tidO = blkBIdx * blockDim.x + threadBRunningIdx;

					// only calculate for points with the same k association
					if (tidO < size && myK == pka[tidO])
					{
						// XA^2+XB^2+YA^2+YB^2  -2*XA*XB -2*YA*YB
						cur = dShared_SquaredXYAB[4 * threadIdx.x + 0] + dShared_SquaredXYAB[4 * threadBRunningIdx + 1]
							+ dShared_SquaredXYAB[4 * threadIdx.x + 2] + dShared_SquaredXYAB[4 * threadBRunningIdx + 3]
							- 2 * d_xya->x[tidA] * d_xya->x[tidO] - 2 * d_xya->y[tidA] * d_xya->y[tidO];
						if (cur > max) max = cur;
					}
				}
			}
			AtomicMax(&(kDiameters[myK]), sqrtf(max));	// takes advantage of varying completion times (OptionA)
		}
		else
		{

			// OPTION B
			//int revTidA = blkAIdx * blockDim.x + blockDim.x - threadIdx.x + 1; // e.g. 1023 -> 0, 1022 -> 1

			//TODO: consider load blancing/loop unrolling of some sort
			// run kernel with a single block, use external block indices to syncronize operations
			int lastIter = THREADS_PER_BLOCK / 2;
			for (int tidO, iter = 1; iter < lastIter; iter++)
			{
				tidO = blkBIdx * blockDim.x + (threadIdx.x + iter) % THREADS_PER_BLOCK;
				if (tidO < size && myK == pka[tidO])
				{
					// XA^2+XB^2+YA^2+YB^2  -2*XA*XB -2*YA*YB
					cur = dShared_SquaredXYAB[4 * threadIdx.x + 0] + dShared_SquaredXYAB[4 * ((threadIdx.x + iter) % THREADS_PER_BLOCK) + 1]
						+ dShared_SquaredXYAB[4 * threadIdx.x + 2] + dShared_SquaredXYAB[4 * ((threadIdx.x + iter) % THREADS_PER_BLOCK) + 3]
						- 2 * d_xya->x[tidA] * d_xya->x[tidO] - 2 * d_xya->y[tidA] * d_xya->y[tidO];
					if (cur > max) max = cur;
				}

			}


			// reduction in shared mem (we finished using it and can use it for another purpose)
			//TEST for ksize=4
			switch (myK)
			{
			case 0:
				dShared_SquaredXYAB[threadIdx.x*ksize + 0] = max;
				dShared_SquaredXYAB[threadIdx.x*ksize + 1] = 0;
				dShared_SquaredXYAB[threadIdx.x*ksize + 2] = 0;
				dShared_SquaredXYAB[threadIdx.x*ksize + 3] = 0;
				break;
			case 1:
				dShared_SquaredXYAB[threadIdx.x*ksize + 0] = 0;
				dShared_SquaredXYAB[threadIdx.x*ksize + 1] = max;
				dShared_SquaredXYAB[threadIdx.x*ksize + 2] = 0;
				dShared_SquaredXYAB[threadIdx.x*ksize + 3] = 0;
				break;
			case 2:
				dShared_SquaredXYAB[threadIdx.x*ksize + 0] = 0;
				dShared_SquaredXYAB[threadIdx.x*ksize + 1] = 0;
				dShared_SquaredXYAB[threadIdx.x*ksize + 2] = max;
				dShared_SquaredXYAB[threadIdx.x*ksize + 3] = 0;
				break;
			case 3:
				dShared_SquaredXYAB[threadIdx.x*ksize + 0] = 0;
				dShared_SquaredXYAB[threadIdx.x*ksize + 1] = 0;
				dShared_SquaredXYAB[threadIdx.x*ksize + 2] = 0;
				dShared_SquaredXYAB[threadIdx.x*ksize + 3] = max;
				break;
			default:
				// NOT SUPPORTED YET!
				dShared_SquaredXYAB[threadIdx.x*ksize + 0] = 0;
				dShared_SquaredXYAB[threadIdx.x*ksize + 1] = 0;
				dShared_SquaredXYAB[threadIdx.x*ksize + 2] = 0;
				dShared_SquaredXYAB[threadIdx.x*ksize + 3] = 0;
				break;
			}
			__syncthreads();

			int tid = threadIdx.x;
			for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
			{
				if (tid < s)
				{
					if (dShared_SquaredXYAB[tid + 0] < dShared_SquaredXYAB[tid + s + 0]) dShared_SquaredXYAB[tid + 0] = dShared_SquaredXYAB[tid + s + 0];
					if (dShared_SquaredXYAB[tid + 1] < dShared_SquaredXYAB[tid + s + 1]) dShared_SquaredXYAB[tid + 1] = dShared_SquaredXYAB[tid + s + 1];
					if (dShared_SquaredXYAB[tid + 2] < dShared_SquaredXYAB[tid + s + 2]) dShared_SquaredXYAB[tid + 2] = dShared_SquaredXYAB[tid + s + 2];
					if (dShared_SquaredXYAB[tid + 3] < dShared_SquaredXYAB[tid + s + 3]) dShared_SquaredXYAB[tid + 3] = dShared_SquaredXYAB[tid + s + 3];
				}
				__syncthreads();
			}
			if (tid < 32) //unroll warp
			{
				if (dShared_SquaredXYAB[tid + 0] < dShared_SquaredXYAB[tid + 32 + 0]) dShared_SquaredXYAB[tid + 0] = dShared_SquaredXYAB[tid + 32 + 0];
				if (dShared_SquaredXYAB[tid + 1] < dShared_SquaredXYAB[tid + 32 + 1]) dShared_SquaredXYAB[tid + 1] = dShared_SquaredXYAB[tid + 32 + 1];
				if (dShared_SquaredXYAB[tid + 2] < dShared_SquaredXYAB[tid + 32 + 2]) dShared_SquaredXYAB[tid + 2] = dShared_SquaredXYAB[tid + 32 + 2];
				if (dShared_SquaredXYAB[tid + 3] < dShared_SquaredXYAB[tid + 32 + 3]) dShared_SquaredXYAB[tid + 3] = dShared_SquaredXYAB[tid + 32 + 3];
				if (dShared_SquaredXYAB[tid + 0] < dShared_SquaredXYAB[tid + 16 + 0]) dShared_SquaredXYAB[tid + 0] = dShared_SquaredXYAB[tid + 16 + 0];
				if (dShared_SquaredXYAB[tid + 1] < dShared_SquaredXYAB[tid + 16 + 1]) dShared_SquaredXYAB[tid + 1] = dShared_SquaredXYAB[tid + 16 + 1];
				if (dShared_SquaredXYAB[tid + 2] < dShared_SquaredXYAB[tid + 16 + 2]) dShared_SquaredXYAB[tid + 2] = dShared_SquaredXYAB[tid + 16 + 2];
				if (dShared_SquaredXYAB[tid + 3] < dShared_SquaredXYAB[tid + 16 + 3]) dShared_SquaredXYAB[tid + 3] = dShared_SquaredXYAB[tid + 16 + 3];
				if (dShared_SquaredXYAB[tid + 0] < dShared_SquaredXYAB[tid + 8 + 0]) dShared_SquaredXYAB[tid + 0] = dShared_SquaredXYAB[tid + 8 + 0];
				if (dShared_SquaredXYAB[tid + 1] < dShared_SquaredXYAB[tid + 8 + 1]) dShared_SquaredXYAB[tid + 1] = dShared_SquaredXYAB[tid + 8 + 1];
				if (dShared_SquaredXYAB[tid + 2] < dShared_SquaredXYAB[tid + 8 + 2]) dShared_SquaredXYAB[tid + 2] = dShared_SquaredXYAB[tid + 8 + 2];
				if (dShared_SquaredXYAB[tid + 3] < dShared_SquaredXYAB[tid + 8 + 3]) dShared_SquaredXYAB[tid + 3] = dShared_SquaredXYAB[tid + 8 + 3];
				if (dShared_SquaredXYAB[tid + 0] < dShared_SquaredXYAB[tid + 4 + 0]) dShared_SquaredXYAB[tid + 0] = dShared_SquaredXYAB[tid + 4 + 0];
				if (dShared_SquaredXYAB[tid + 1] < dShared_SquaredXYAB[tid + 4 + 1]) dShared_SquaredXYAB[tid + 1] = dShared_SquaredXYAB[tid + 4 + 1];
				if (dShared_SquaredXYAB[tid + 2] < dShared_SquaredXYAB[tid + 4 + 2]) dShared_SquaredXYAB[tid + 2] = dShared_SquaredXYAB[tid + 4 + 2];
				if (dShared_SquaredXYAB[tid + 3] < dShared_SquaredXYAB[tid + 4 + 3]) dShared_SquaredXYAB[tid + 3] = dShared_SquaredXYAB[tid + 4 + 3];
				if (dShared_SquaredXYAB[tid + 0] < dShared_SquaredXYAB[tid + 2 + 0]) dShared_SquaredXYAB[tid + 0] = dShared_SquaredXYAB[tid + 2 + 0];
				if (dShared_SquaredXYAB[tid + 1] < dShared_SquaredXYAB[tid + 2 + 1]) dShared_SquaredXYAB[tid + 1] = dShared_SquaredXYAB[tid + 2 + 1];
				if (dShared_SquaredXYAB[tid + 2] < dShared_SquaredXYAB[tid + 2 + 2]) dShared_SquaredXYAB[tid + 2] = dShared_SquaredXYAB[tid + 2 + 2];
				if (dShared_SquaredXYAB[tid + 3] < dShared_SquaredXYAB[tid + 2 + 3]) dShared_SquaredXYAB[tid + 3] = dShared_SquaredXYAB[tid + 2 + 3];
				if (dShared_SquaredXYAB[tid + 0] < dShared_SquaredXYAB[tid + 1 + 0]) dShared_SquaredXYAB[tid + 0] = dShared_SquaredXYAB[tid + 1 + 0];
				if (dShared_SquaredXYAB[tid + 1] < dShared_SquaredXYAB[tid + 1 + 1]) dShared_SquaredXYAB[tid + 1] = dShared_SquaredXYAB[tid + 1 + 1];
				if (dShared_SquaredXYAB[tid + 2] < dShared_SquaredXYAB[tid + 1 + 2]) dShared_SquaredXYAB[tid + 2] = dShared_SquaredXYAB[tid + 1 + 2];
				if (dShared_SquaredXYAB[tid + 3] < dShared_SquaredXYAB[tid + 1 + 3]) dShared_SquaredXYAB[tid + 3] = dShared_SquaredXYAB[tid + 1 + 3];

			}
			if (tid < 4)
				kDiameters[tid]= sqrtf(dShared_SquaredXYAB[tid]);	// sqrtf(max)
		}
		

	}
}





// Helper function for finding best centers for ksize clusters
cudaError_t kCentersWithCuda(xyArrays* kCenters, int ksize, xyArrays* xya, int* pka, long N, int LIMIT)
{
	cudaError_t cudaStatus;
	const int NO_BLOCKS = (N % THREADS_PER_BLOCK == 0) ? N / THREADS_PER_BLOCK : N / THREADS_PER_BLOCK + 1;

	initK(ksize);				// K-centers = first points in data (on host)
	int iter = 0;
	bool flag;
	
	// memory initializations
	size_t nDataBytes = N * 2 * sizeof(float);  // N x 2 x sizeof(float)
	size_t nKCenterBytes = ksize * 2 * sizeof(float);
	bool	 *h_kaFlags;
	int	 *d_pka;					// array to associate xya points with their closest cluster
	bool     *d_kaFlags;				// array to flags changes in point-to-cluster association	
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// allocate host-side helpers
	h_kaFlags = (bool*)malloc(NO_BLOCKS * sizeof(bool));
	
	xyArrays *d_xya,
		*d_kCenters;				// data and k-centers xy information
	xyArrays da_xya, h_kCenters;     // da_xya device anchor for copying xy-arrays data

	cudaStatus = cudaMalloc(&d_xya, sizeof(xyArrays)); CHKMAL_ERROR;
	
	cudaStatus = cudaMalloc(&(da_xya.x), nDataBytes / 2); CHKMAL_ERROR;
	cudaStatus = cudaMalloc(&(da_xya.y), nDataBytes / 2); CHKMAL_ERROR; 
	cudaStatus = cudaMemcpy(da_xya.x, xya->x, nDataBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR; 
	cudaStatus = cudaMemcpy(da_xya.y, xya->y, nDataBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
	
	cudaStatus = cudaMalloc(&d_kCenters, sizeof(xyArrays));
	// allocate device memory
	cudaStatus = cudaMalloc(&(h_kCenters.x), nKCenterBytes / 2); CHKMAL_ERROR;
	cudaStatus = cudaMalloc(&(h_kCenters.y), nKCenterBytes / 2); CHKMAL_ERROR; 
	
	cudaStatus = cudaMemcpy(d_xya, &da_xya, sizeof(xyArrays), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
	cudaStatus = cudaMemcpy(d_kCenters, &h_kCenters, sizeof(xyArrays), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;

	cudaStatus = cudaMalloc(&d_pka, N * sizeof(int)); CHKMAL_ERROR;
	cudaStatus = cudaMalloc(&d_kaFlags, N * sizeof(bool)); CHKMAL_ERROR; 

	// copy cluster association data from host to device
	cudaStatus = cudaMemcpy(d_pka, pka, N * sizeof(int), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
	
	cudaStatus = cudaMemset((void*)d_kaFlags, 0, N * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!\n"); FF;
		goto Error;
	}
	
	// *** phase 1 ***
	do {
		
		cudaStatus = cudaMemcpy(h_kCenters.x, kCenters->x, nKCenterBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
		cudaStatus = cudaMemcpy(h_kCenters.y, kCenters->y, nKCenterBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;

#ifdef _DEBUGSM
		printf("__global__ reClusterWithCuda() call with %d SharedMemBytes\n", SharedMemBytes); FF;
#endif
		//KernelFunc << <DimGrid, DimBlock, SharedMemBytes >> >n

		if (cudaGetLastError() != cudaSuccess) { printf("Failed before reCluster\n"); FF; }
		reClusterWithCuda << <NO_BLOCKS, THREADS_PER_BLOCK >> > (d_kCenters, ksize, d_xya, d_pka, d_kaFlags, N); // THREADS_PER_BLOCK, THREAD_BLOCK_SIZE
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "reClusterWithCuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
			FF;
			goto Error;
		}
		cudaStatus = cudaDeviceSynchronize(); CHKSYNC_ERROR;
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "reClusterWithCuda run failed: %s\n", cudaGetErrorString(cudaStatus));
			FF;
			goto Error;
		}


		cudaStatus = cudaMemcpy(h_kaFlags, d_kaFlags, NO_BLOCKS * sizeof(bool), cudaMemcpyDeviceToHost); CHKMEMCPY_ERROR;
		cudaStatus = cudaMemcpy(pka, d_pka, N * sizeof(int), cudaMemcpyDeviceToHost); CHKMEMCPY_ERROR;
		
		flag = ompReduceCudaFlags(h_kaFlags, NO_BLOCKS);
		
		//TODO: consider replacing with a CUDA implementation
		ompRecenterFromCuda(ksize); 
#ifdef _DEBUGV
		printf("kCentersWithCuda on %d, iter: %d\n", ksize, iter + 1); FF;
#endif
	} while (++iter < LIMIT && flag);  // association changes: need to re-cluster

	free(h_kaFlags);

Error:
	cudaFree(d_xya);
	cudaFree(da_xya.x);
	cudaFree(da_xya.y);
	cudaFree(h_kCenters.x);
	cudaFree(h_kCenters.y);
	cudaFree(d_kCenters);
	cudaFree(d_pka);
	cudaFree(d_kaFlags);

	return cudaStatus;
}

// Helper function for obtaining best candidates for kDiameters on a block x block metric
cudaError_t kDiametersWithCuda(float* kDiameters, int ksize, xyArrays* xya, int* pka, long N, int myid, int numprocs)
{
	cudaError_t cudaStatus;
	double diamKerStart, diamKerFinish;
	const int NO_BLOCKS = (N % THREADS_PER_BLOCK == 0) ? N / THREADS_PER_BLOCK : N / THREADS_PER_BLOCK + 1;
	xyArrays *d_xya;
	int	 *d_pka;
	float* d_kDiameters;
	size_t SharedMemBytes;
	MPI_Status status;

	//initialize diameters as zero
	for (int i = 0; i < ksize; i++)
	{
		kDiameters[i] = 0;
	}

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// allocate device memory
	size_t nDataBytes = N * 2 * sizeof(float);  // N x 2 x sizeof(float)
	xyArrays da_xya;  // device anchor for copying xy-arrays data
	cudaMalloc(&(da_xya.x), nDataBytes / 2); CHKMAL_ERROR;
	cudaMalloc(&(da_xya.y), nDataBytes / 2); CHKMAL_ERROR;
	cudaMemcpy(da_xya.x, xya->x, nDataBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
	cudaMemcpy(da_xya.y, xya->y, nDataBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
		
	cudaMalloc(&d_xya, sizeof(xyArrays));
	cudaMemcpy(d_xya, &da_xya, sizeof(xyArrays), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;

	//float* d_kDiameters;
	cudaMalloc(&d_kDiameters, ksize * sizeof(float));
	cudaMemcpy(d_kDiameters, kDiameters, ksize * sizeof(float), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;

	cudaMalloc(&d_pka, N * sizeof(int)); CHKMAL_ERROR;
	cudaMemcpy(d_pka, pka, N * sizeof(int), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;

	//SharedMemBytes = 4 * THREADS_PER_BLOCK * sizeof(float); // shared memory for flag work
	

	//MPI single -- working out the single BLOCK problem with CUDA
	if (myid == MASTER)
	{
#ifdef _DEBUG2
		//TEST print
		printf("%d, FirstJob, Blocks %2d, %2d\n", myid, 0, 0); fflush(stdout);
#endif
#ifdef _DEBUGSM
		printf("__global__ kDiamBlockWithCuda() call with %d SharedMemBytes\n", SharedMemBytes); FF;
#endif
#ifdef _PROF3
		diamKerStart = omp_get_wtime();
#endif
		kDiamBlockWithCuda << <1, THREADS_PER_BLOCK >> > (d_kDiameters, ksize, d_xya, d_pka, N, 0, 0);
		cudaStatus = cudaGetLastError(); 
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "id: %d, kernel kDiamBlockWithCuda launch failed: %s\n", myid, cudaGetErrorString(cudaStatus));
			FF;
			goto Error;
		}
		cudaStatus = cudaDeviceSynchronize(); CHKSYNC_ERROR;
#ifdef _PROF3
		diamKerFinish = omp_get_wtime();
		if (myid == MASTER) { printf("kDiamBlock %d run-time: %f\n", ksize, diamKerFinish - diamKerStart); FF; }
#endif
		cudaStatus = cudaMemcpy(kDiameters, d_kDiameters, ksize * sizeof(float), cudaMemcpyDeviceToHost); CHKMEMCPY_ERROR;

#ifdef _DEBUG1
		//TEST kDiameters 
		printArrTestPrint(myid, kDiameters, ksize, "kDiameters");
#endif
	}
	
	//TODO: use MASTER GPU to asynchronously run first job and poll for completion to give new jobs
	/*
	//async initializations for MASTER
	cudaEvent_t myJobIsDone;
	cudaStatus = cudaEventCreateWithFlags(&myJobIsDone, cudaEventDisableTiming); EVENT_ERROR;
	cudaEventDestroy(myJobIsDone); EVENT_ERROR;


	//cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
	//kDiamBlockWithCuda << <1, THREADS_PER_BLOCK, SharedMemBytes >> > (d_kDiameters, ksize, d_xya, d_pka, N, 0, 0);
	//cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
	//cudaEventRecord(myJobIsDone, 0);
	//
	//while (cudaEventQuery(stop) == cudaErrorNotReady) {
	//TODO:
	//non-blocking recv from slaves;
	// }
	*/

	//MASTER-SLAVES
	float* kDiametersTempAnswer, *kDiametersAnswer;
	kDiametersTempAnswer = (float*)malloc(ksize * sizeof(float));
	if (myid == MASTER)
	{
		
		int x, const NO_JOBS = (NO_BLOCKS+1)*(float)NO_BLOCKS/2;
		int* jobs = initJobArray(NO_BLOCKS, NO_JOBS);
		int resultsCounter = 1;
		
		// distribute work to SLAVES
		for (x = 1; x < numprocs && x < NO_JOBS; x++)
		{
			// send numprocs values to get the work started
			MPI_Send(&jobs[2*x], 2, MPI_INT, x, NEW_JOB, MPI_COMM_WORLD);
		}
		// dynamically allocate further jobs as results are coming in
		while (resultsCounter < NO_JOBS)
		{
			//TEST print
			//printf("x value %2d, count: %2d\n", x, resultsCounter); fflush(stdout);
			if (numprocs > 1)
				MPI_Recv(kDiametersTempAnswer, ksize, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			else
			{   // only the "MASTER" works
				kDiamBlockWithCuda << <1, THREADS_PER_BLOCK >> > (d_kDiameters, ksize, d_xya, d_pka, N, jobs[2 * x], jobs[2 * x + 1]); x++;
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) 
				{
					fprintf(stderr, "id: %d, kernel kDiamBlockWithCuda launch failed: %s\n", myid, cudaGetErrorString(cudaStatus));
					FF;
					goto Error;
				}
			}
			cudaStatus = cudaDeviceSynchronize(); CHKSYNC_ERROR;

			cudaStatus = cudaMemcpy(kDiametersTempAnswer, d_kDiameters, ksize * sizeof(float), cudaMemcpyDeviceToHost); CHKMEMCPY_ERROR;

			resultsCounter++;

			ompMaxVectors(&kDiameters, kDiametersTempAnswer, ksize);


#ifdef _DEBUG1
			//TEST kDiameters 
			printf("\nMaster values after MaxVectors with source %d !!\n", status.MPI_SOURCE);
			printArrTestPrint(myid, kDiameters, ksize, "kDiameters");
			printf("***********************************************\n\n");
#endif

			// if needed, send next job and increase x
			if (numprocs > 1)
			{
				if (x < NO_JOBS)
				{
					MPI_Send(&jobs[2 * x], 2, MPI_INT, status.MPI_SOURCE, NEW_JOB, MPI_COMM_WORLD);
					x++;
				}
				else
				{
					// notify process about work completion
					MPI_Send(&x, 1, MPI_INT, status.MPI_SOURCE, STOP_WORKING, MPI_COMM_WORLD);  // message with tag==1 from master: work complete
				}
			}
		}
		
	}
	// SLAVES
	else {  //slaves
		int masterTag = NEW_JOB;
		int jobForBlocks[2];
		while (masterTag == NEW_JOB)
		{

			MPI_Recv(jobForBlocks, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			masterTag = status.MPI_TAG;

			if (masterTag == NEW_JOB)
			{
#ifdef _DEBUG2
				//TEST print
				printf("%d, jobForBlocks %2d, %2d\n", myid, jobForBlocks[0], jobForBlocks[1]); fflush(stdout);
#endif
#ifdef _DEBUGSM
				printf("__global__ kDiamBlockWithCuda() call with %d SharedMemBytes\n", SharedMemBytes); FF;
#endif
				kDiamBlockWithCuda << <1, THREADS_PER_BLOCK >>> (d_kDiameters, ksize, d_xya, d_pka, N, jobForBlocks[0], jobForBlocks[1]);
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "id: %d, main kDiamBlockWithCuda launch failed: %s\n", myid, cudaGetErrorString(cudaStatus));
					FF;
					goto Error;
				}
				cudaStatus = cudaDeviceSynchronize(); CHKSYNC_ERROR;

				cudaStatus = cudaMemcpy(kDiameters, d_kDiameters, ksize * sizeof(float), cudaMemcpyDeviceToHost); CHKMEMCPY_ERROR;
#ifdef _DEBUG1
				//TEST kDiameters 
				printArrTestPrint(myid, kDiameters, ksize, "kDiameters");
#endif

				MPI_Send(kDiameters, ksize, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);	   // report your rank to master in tag (not necessary)
			}
			else
			{
				goto Error;
			}
		}
	}


Error:
	cudaFree(d_xya);
	cudaFree(d_kDiameters);
	cudaFree(d_pka);


	return cudaStatus;
}