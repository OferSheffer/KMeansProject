#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <omp.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "Kmeans.h"


#define CHKMAL_ERROR	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
#define CHKMEMCPY_ERROR if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
#define CHKSYNC_ERROR	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize failed! Error code %d\n", cudaStatus); goto Error; }
#define EVENT_ERROR		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaEventOperation failed! Error code %d\n", cudaStatus); goto Error; }

// arrSize indices; THREADS_PER_BLOCK * NO_BLOCKS total threads;
// Each thread in charge of THREAD_BLOCK_SIZE contigeous indices

#define MASTER 0
#define THREADS_PER_BLOCK 1024
#define NEW_JOB 0
#define STOP_WORKING 1

__global__ void reClusterWithCuda(xyArrays* d_kCenters, const int ksize, xyArrays* d_xya, int* pka, bool* d_kaFlags, const int size)
{
	__shared__ bool dShared_kaFlags[THREADS_PER_BLOCK]; // array to flag changes in point-to-cluster association

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
	__shared__ float dShared_SquaredXYAB[THREADS_PER_BLOCK * 4]; // save squared values for reuse

	// local shared mem speedup - save squared values for reuse
	// diameter^2 = (XA-XB)^2 + (YA-YB)^2 = XA^2+XB^2+YA^2+YB^2  -2*XA*XB -2*YA*YB
	unsigned int tidA = blkAIdx * blockDim.x + threadIdx.x;
	unsigned int tidB = blkBIdx * blockDim.x + threadIdx.x;
	dShared_SquaredXYAB[4 * threadIdx.x] = powf(d_xya->x[tidA], 2);	// i%4==0: x^2 of blkA
	dShared_SquaredXYAB[4 * threadIdx.x + 1] = powf(d_xya->x[tidB], 2);	// i%4==1: x^2 of blkB
	dShared_SquaredXYAB[4 * threadIdx.x + 2] = powf(d_xya->y[tidA], 2);	// i%4==2: y^2 of blkA
	dShared_SquaredXYAB[4 * threadIdx.x + 3] = powf(d_xya->y[tidB], 2);	// i%4==3: y^2 of blkB
	__syncthreads();

	float max = 0;
	float cur;

	// run kernel with a single block, use external block indices to syncronize operations
	for (int oIdx = 0; oIdx < blockDim.x; oIdx++)
	{
		// prevent repeated calculations
		if (threadIdx.x < oIdx)
		{
			tidB = blkBIdx * blockDim.x + oIdx;

			// only calculate for points with the same k association
			if (pka[tidA] == pka[tidB])
			{
				// XA^2+XB^2+YA^2+YB^2  -2*XA*XB -2*YA*YB
				cur = dShared_SquaredXYAB[4 * threadIdx.x]     + dShared_SquaredXYAB[4 * oIdx + 1]
					+ dShared_SquaredXYAB[4 * threadIdx.x + 2] + dShared_SquaredXYAB[4 * oIdx + 3]
					- 2 * d_xya->x[tidA] * d_xya->x[tidB] - 2 * d_xya->y[tidA] * d_xya->y[tidA];
				if (cur > max) max = cur;
			}
		}
	}
	//TODO: consider reduction instead
	// takes advantage of varying completion times of threads
	AtomicMax(&(kDiameters[pka[threadIdx.x]]), max);
}





// Helper function for finding best centers for ksize clusters
cudaError_t kCentersWithCuda(xyArrays* kCenters, int ksize, xyArrays* xya, int* pka, long N, int LIMIT)
{
	cudaError_t cudaStatus;
	const int NO_BLOCKS = (N % THREADS_PER_BLOCK == 0) ? N / THREADS_PER_BLOCK : N / THREADS_PER_BLOCK + 1;
	const int THREAD_BLOCK_SIZE = THREADS_PER_BLOCK;

	// data size protection code
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
	size_t nDataBytes = N * sizeof(*xya);  // N x 2 x sizeof(float)
	size_t nKCenterBytes = ksize * sizeof(*kCenters);
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

	// allocate device memory
	xyArrays *d_xya,
		*d_kCenters;				// data and k-centers xy information
	xyArrays da_xya, h_kCenters;     // da_xya device anchor for copying xy-arrays data

	cudaMalloc(&d_xya, sizeof(xyArrays)); CHKMAL_ERROR;

	cudaMalloc(&(da_xya.x), nDataBytes / 2); CHKMAL_ERROR;
	cudaMalloc(&(da_xya.y), nDataBytes / 2); CHKMAL_ERROR;
	cudaMemcpy(da_xya.x, xya->x, nDataBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
	cudaMemcpy(da_xya.y, xya->y, nDataBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;

	cudaMalloc(&d_kCenters, sizeof(xyArrays));
	cudaMalloc(&(h_kCenters.x), nKCenterBytes / 2); CHKMAL_ERROR;
	cudaMalloc(&(h_kCenters.y), nKCenterBytes / 2); CHKMAL_ERROR;
	
	cudaMemcpy(d_xya, &da_xya, sizeof(xyArrays), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
	cudaMemcpy(d_kCenters, &h_kCenters, sizeof(xyArrays), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;

	cudaMalloc(&d_pka, N * sizeof(int)); CHKMAL_ERROR;
	cudaMalloc(&d_kaFlags, N * sizeof(bool)); CHKMAL_ERROR;

	// copy cluster association data from host to device
	cudaMemcpy(d_pka, pka, N * sizeof(int), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;

	cudaStatus = cudaMemset((void*)d_kaFlags, 0, N * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!\n");
		goto Error;
	}
	//}

	// *** phase 1 ***
	do {
		
		cudaMemcpy(h_kCenters.x, kCenters->x, nKCenterBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
		cudaMemcpy(h_kCenters.y, kCenters->y, nKCenterBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
		//TEST KCenters per iteration
		/*
		for (int i = 0; i < ksize; i++)
		{
			printf("%d: k%d, %8.3f, %8.3f\n", iter+1, i, kCenters->x[i], kCenters->y[i]);
		}
		*/

		//KernelFunc << <DimGrid, DimBlock, SharedMemBytes >> >
		reClusterWithCuda << <NO_BLOCKS, THREADS_PER_BLOCK, SharedMemBytes >> > (d_kCenters, ksize, d_xya, d_pka, d_kaFlags, N); // THREADS_PER_BLOCK, THREAD_BLOCK_SIZE
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "reClusterWithCuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}
		cudaStatus = cudaDeviceSynchronize(); CHKSYNC_ERROR;


		cudaStatus = cudaMemcpy(h_kaFlags, d_kaFlags, NO_BLOCKS * sizeof(bool), cudaMemcpyDeviceToHost); CHKMEMCPY_ERROR;
		cudaStatus = cudaMemcpy(pka, d_pka, N * sizeof(int), cudaMemcpyDeviceToHost); CHKMEMCPY_ERROR;
		
		flag = ompReduceCudaFlags(h_kaFlags, NO_BLOCKS);
		
		//TODO: consider replacing with a CUDA implementation
		ompRecenterFromCuda(ksize);

	} while (++iter < LIMIT && flag);  // association changes: need to re-cluster

	//TODO: use if using CUDA to reCenter
	//cudaMemcpy(kCenters->x, h_kCenters.x, nKCenterBytes / 2, cudaMemcpyDeviceToHost); CHKMEMCPY_ERROR;
	//cudaMemcpy(kCenters->y, h_kCenters.y, nKCenterBytes / 2, cudaMemcpyDeviceToHost); CHKMEMCPY_ERROR;

	//TODO quick test
	printf("k-complete:\n");
	for (int i = 0; i < ksize; i++)
	{
		printf("%d, %8.3f, %8.3f\n", i, kCenters->x[i], kCenters->y[i]);
	}
	

	free(h_kaFlags);

Error:
	cudaFree(d_xya);
	cudaFree(d_kCenters);
	cudaFree(d_pka);
	cudaFree(d_kaFlags);

	return cudaStatus;
}

// Helper function for obtaining best candidates for kDiameters on a block x block metric
cudaError_t kDiametersWithCuda(float* kDiameters, int ksize, xyArrays* xya, int* pka, long N, int myid, int numprocs)
{
	cudaError_t cudaStatus = cudaSuccess; //TODO: rm success
	const int NO_BLOCKS = (N % THREADS_PER_BLOCK == 0) ? N / THREADS_PER_BLOCK : N / THREADS_PER_BLOCK + 1;
	const int THREAD_BLOCK_SIZE = THREADS_PER_BLOCK;
	MPI_Status status;

	for (int i = 0; i < ksize; i++)
	{
		kDiameters[i] = 0;
	}

	// allocate device memory
	size_t nDataBytes = N * sizeof(*xya);  // N x 2 x sizeof(float)
	xyArrays da_xya;  // device anchor for copying xy-arrays data
	cudaMalloc(&(da_xya.x), nDataBytes / 2); CHKMAL_ERROR;
	cudaMalloc(&(da_xya.y), nDataBytes / 2); CHKMAL_ERROR;
	cudaMemcpy(da_xya.x, xya->x, nDataBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
	cudaMemcpy(da_xya.y, xya->y, nDataBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR; 
	xyArrays *d_xya;
	cudaMalloc(&d_xya, sizeof(xyArrays));
	cudaMemcpy(d_xya, &da_xya, sizeof(xyArrays), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;




	//MPI test
	if (myid == MASTER)
		printf("id %d, %3d: %f, %d\n", myid, NO_BLOCKS, kDiameters[0], ksize); fflush(stdout);


	//MPI single -- working out the single BLOCK problem with CUDA
	if (myid == MASTER)
	{
		//TEST kDiameters 
		for (int i = 0; i < ksize; i++)
		{
			printf("kDiam%d: %8.3f\n", i, kDiameters[i]); fflush(stdout);
		}
		

		kDiamBlockWithCuda(d_kDiameters, ksize, d_xya, d_pka, size, 0, 0);

		//TEST kDiameters 
		for (int i = 0; i < ksize; i++)
		{
			printf("kDiam%d: %8.3f\n", i, kDiameters[i]); fflush(stdout);
		}


	}

	//MASTER-SLAVES
	/*
	if (myid == MASTER)
	{
		int fact = 1, x;
		for (int c = 2; c <= NO_BLOCKS; c++)
			fact *= c;
		int* jobs = initJobArray(NO_BLOCKS, fact);
		int resultsCounter = 0;
		
		//async initializations for MASTER
		cudaEvent_t myJobIsDone;
		cudaStatus = cudaEventCreateWithFlags(&myJobIsDone, cudaEventDisableTiming); EVENT_ERROR;
		cudaEventDestroy(myJobIsDone); EVENT_ERROR;

		//use MASTER GPU to asynchronously run first job
		//TODO:
		//cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
		//increment_kernel << <blocks, threads, 0, 0 >> >(d_a, value);
		//cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
		//cudaEventRecord(myJobIsDone, 0);
		//
		//while (cudaEventQuery(stop) == cudaErrorNotReady) {
			//TODO:
			//non-blocking recv from slaves;
		// }
		
		// distribute work to SLAVES
		for (x = 1; x < numprocs && x < fact; x++)
		{
			// send numprocs values to get the work started
			MPI_Send(&jobs[2*x], 2, MPI_INT, x, NEW_JOB, MPI_COMM_WORLD);
		}
		// dynamically allocate further jobs as results are coming in
		while (resultsCounter < fact)
		{
			//TEST print
			printf("x value %2d, count: %2d\n", x, resultsCounter); fflush(stdout);

			//TODO:
			MPI_Recv(&tempAnswer, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			resultsCounter++;
			//TODO:
			answer += tempAnswer;

			// if needed, send next job and increase x
			if (x < fact)
			{
				MPI_Send(&jobs[2*x], 2, MPI_INT, status.MPI_SOURCE, NEW_JOB, MPI_COMM_WORLD);
				x++;
			}
			else
			{
				// notify process about work completion
				MPI_Send(&x, 1, MPI_INT, status.MPI_SOURCE, STOP_WORKING, MPI_COMM_WORLD);  // message with tag==1 from master: work complete
			}
		}
		
		
	} */
	// SLAVES
	/*
	else {  //slaves
		int masterTag = NEW_JOB;
		int jobForBlocks[2];
		while (masterTag == NEW_JOB)
		{
			MPI_Recv(jobForBlocks, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			masterTag = status.MPI_TAG;

			if (masterTag == NEW_JOB)
			{
				answer = 0; // make sure local answer == 0
				//TODO:
				//increment_kernel << <blocks, threads, 0, 0 >> >(d_a, value);

				MPI_Send(&answer, 1, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD);	   // report your rank to master in tag (not necessary)
			}
			else
			{
				goto Error;
			}
		}
	}*/

Error:
	//cudaFree(d_xya);
	//cudaFree(d_kCenters);


	return cudaStatus;
}