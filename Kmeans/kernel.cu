#include "Kmeans.h"

#define CHKMAL_ERROR	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); FF; goto Error; }
#define CHKMEMCPY_ERROR if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); FF; goto Error; }
#define CHKSYNC_ERROR	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize failed! Error code %d\n", cudaStatus); FF; goto Error; }
#define EVENT_ERROR		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaEventOperation failed! Error code %d\n", cudaStatus); FF; goto Error; }
#define STREAMCR_ERROR	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaStreamCreate failed! Error code %d\n", cudaStatus); FF; goto Error; }

#define CHKSYNC_ERRORDDD	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize failed here!!!! Error code %d\n", cudaStatus); FF; goto Error; }


#define MASTER 0
#define NEW_JOB 0
#define STOP_WORKING 1

///////////////////////////////////////////
// CUDA functions
__global__ void reClusterWithCuda(xyArrays* d_kCenters, const int ksize, xyArrays* d_xya, int* pka, bool* d_kaFlags, const int size)
{
	extern __shared__ bool dShared_kaFlags[]; // array to flag changes in point-to-cluster association

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
			dShared_kaFlags[ltid] |= dShared_kaFlags[ltid + 32];
			dShared_kaFlags[ltid] |= dShared_kaFlags[ltid + 16];
			dShared_kaFlags[ltid] |= dShared_kaFlags[ltid + 8];
			dShared_kaFlags[ltid] |= dShared_kaFlags[ltid + 4];
			dShared_kaFlags[ltid] |= dShared_kaFlags[ltid + 2];
			dShared_kaFlags[ltid] |= dShared_kaFlags[ltid + 1];
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
	extern __shared__ float dShared_SquaredXYAB[]; // save squared values for reuse
	float XAP0, YAP0, XAP1, YAP1;

	// local shared mem speedup - save squared values for reuse
	// diameter^2 = (XA-XB)^2 + (YA-YB)^2 = XA^2+XB^2+YA^2+YB^2  -2*XA*XB -2*YA*YB
	unsigned int tidA = blkAIdx * blockDim.x + threadIdx.x;


	if (tidA < size)
	{
		XAP0 = powf(d_xya->x[tidA], 2);					// x^2 of blkA
		YAP0 = powf(d_xya->y[tidA], 2);					// y^2 of blkA
		if (tidA + blockDim.x < size)
		{
			XAP1 = powf(d_xya->x[tidA + blockDim.x], 2);	// x^2 of blkA+1
			YAP1 = powf(d_xya->y[tidA + blockDim.x], 2);	// y^2 of blkA+1
		}

		unsigned int tidB = blkBIdx * blockDim.x + threadIdx.x;
		if (tidB < size)
		{
			dShared_SquaredXYAB[2 * threadIdx.x + 0] = powf(d_xya->x[tidB], 2);	// i%2==0: x^2 of blkB
			dShared_SquaredXYAB[2 * threadIdx.x + 1] = powf(d_xya->y[tidB], 2);	// i%2==1: y^2 of blkB
		}
		__syncthreads();

		float max0 = 0;
		float max1 = 0;
		float cur;
		int myK0 = pka[tidA];
		int myK1 = (tidA + blockDim.x < size) ? pka[tidA + blockDim.x] : -1;

		//int revTidA = blkAIdx * blockDim.x + blockDim.x - threadIdx.x + 1; // e.g. 1023 -> 0, 1022 -> 1

		// run kernel with a single block, use external block indices to syncronize operations
		int lastIter = blockDim.x / 2;
		int tidO;
		for (int iter = 1; iter < lastIter; iter++)
		{
			// BlkA0 & BlkA1 compared with BlkB0
			tidO = blkBIdx * blockDim.x + (threadIdx.x + iter) % blockDim.x;
			if (tidO < size)
				if (myK0 == pka[tidO])
				{
					// XA^2+XB^2+YA^2+YB^2  -2*XA*XB -2*YA*YB
					cur = XAP0 + dShared_SquaredXYAB[2 * ((threadIdx.x + iter) % blockDim.x) + 0]
						+ YAP0 + dShared_SquaredXYAB[2 * ((threadIdx.x + iter) % blockDim.x) + 1]
						- 2 * d_xya->x[tidA] * d_xya->x[tidO] - 2 * d_xya->y[tidA] * d_xya->y[tidO];
					if (cur > max0) max0 = cur;
				}
				else if (myK1 == pka[tidO])
				{
					// XA^2+XB^2+YA^2+YB^2  -2*XA*XB -2*YA*YB
					cur = XAP0 + dShared_SquaredXYAB[2 * ((threadIdx.x + iter) % blockDim.x) + 0]
						+ YAP0 + dShared_SquaredXYAB[2 * ((threadIdx.x + iter) % blockDim.x) + 1]
						- 2 * d_xya->x[tidA] * d_xya->x[tidO] - 2 * d_xya->y[tidA] * d_xya->y[tidO];
					if (cur > max1) max1 = cur;
				}
		}
		__syncthreads();

		if (tidB + blockDim.x < size)
		{
			dShared_SquaredXYAB[2 * threadIdx.x + 0] = powf(d_xya->x[tidB + blockDim.x], 2);	// i%2==0: x^2 of blkB
			dShared_SquaredXYAB[2 * threadIdx.x + 1] = powf(d_xya->y[tidB + blockDim.x], 2);	// i%2==1: y^2 of blkB
		}
		__syncthreads();
		for (int iter = 1; iter < lastIter; iter++)
		{
			// BlkA0 & BlkA1 compared with BlkB1
			tidO = (blkBIdx + 1) * blockDim.x + (threadIdx.x + iter) % blockDim.x;
			if (tidO < size)
				if (myK0 == pka[tidO] && blkAIdx < blkBIdx)
				{
					// XA^2+XB^2+YA^2+YB^2  -2*XA*XB -2*YA*YB
					cur = XAP0 + dShared_SquaredXYAB[2 * ((threadIdx.x + iter) % blockDim.x) + 0]
						+ YAP0 + dShared_SquaredXYAB[2 * ((threadIdx.x + iter) % blockDim.x) + 1]
						- 2 * d_xya->x[tidA] * d_xya->x[tidO] - 2 * d_xya->y[tidA] * d_xya->y[tidO];
					if (cur > max0) max0 = cur;
				}
				else if (myK1 == pka[tidO])
				{
					// XA^2+XB^2+YA^2+YB^2  -2*XA*XB -2*YA*YB
					cur = XAP0 + dShared_SquaredXYAB[2 * ((threadIdx.x + iter) % blockDim.x) + 0]
						+ YAP0 + dShared_SquaredXYAB[2 * ((threadIdx.x + iter) % blockDim.x) + 1]
						- 2 * d_xya->x[tidA] * d_xya->x[tidO] - 2 * d_xya->y[tidA] * d_xya->y[tidO];
					if (cur > max1) max1 = cur;
				}
		}
		AtomicMax(&(kDiameters[myK0]), sqrtf(max0));
		AtomicMax(&(kDiameters[myK1]), sqrtf(max1));
	}
}


//////////////////////////////////////////
// Host functions

// Finding best centers for ksize clusters
cudaError_t kCentersWithCuda(xyArrays* kCenters, int ksize, xyArrays* xya, int* pka, long N, int LIMIT, int THREADS_PER_BLOCK)
{
	cudaError_t cudaStatus;
	size_t SharedMemBytes;
	//const int NO_BLOCKS = (N % THREADS_PER_BLOCK == 0) ? N / THREADS_PER_BLOCK : N / THREADS_PER_BLOCK + 1;
	const int NO_BLOCKS = ceil(N / (float)THREADS_PER_BLOCK);

	initK(ksize);				// K-centers = first points in data (on host)
	int iter = 0;
	bool flag;

	// memory initializations
	size_t nDataBytes = N * 2 * sizeof(float);  // N x 2 x sizeof(float)
	size_t nKCenterBytes = ksize * 2 * sizeof(float);
	bool *h_kaFlags;
	int	 *d_pka;					// array to associate xya points with their closest cluster
	bool *d_kaFlags;				// array to flags changes in point-to-cluster association	
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"); FF;
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
#ifdef _DEBUGV
		printf("kCentersWithCuda loop on %d, starting iter: %d\n", ksize, iter + 1); FF;
#endif

		cudaStatus = cudaMemcpy(h_kCenters.x, kCenters->x, nKCenterBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
		cudaStatus = cudaMemcpy(h_kCenters.y, kCenters->y, nKCenterBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;

		// Kernel call ** reClusterWithCuda
		SharedMemBytes = THREADS_PER_BLOCK * sizeof(bool);

		if (cudaGetLastError() != cudaSuccess) { printf("Failed before reCluster\n"); FF; }
		reClusterWithCuda << <NO_BLOCKS, THREADS_PER_BLOCK, SharedMemBytes >> > (d_kCenters, ksize, d_xya, d_pka, d_kaFlags, N); // THREADS_PER_BLOCK, THREAD_BLOCK_SIZE
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			int SharedMemBytes = THREADS_PER_BLOCK * sizeof(bool);
			printf("__global__ reClusterWithCuda() call with %d SharedMemBytes\n", SharedMemBytes); FF;
			fprintf(stderr, "reClusterWithCuda launch failed: %s\n", cudaGetErrorString(cudaStatus)); FF;
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

		ompRecenterFromCuda(ksize);

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
cudaError_t kDiametersWithCuda(float* kDiameters, int ksize, xyArrays* xya, int* pka, long N, int myid, int numprocs, int THREADS_PER_BLOCK)
{
	cudaError_t cudaStatus;
	size_t SharedMemBytes;
	double diamKerStart, diamKerFinish;
	const int NO_BLOCKS = ceil(N / (float)THREADS_PER_BLOCK);

	xyArrays *d_xya;
	int	 *d_pka;
	float* d_kDiameters;
	MPI_Status status;
	// Kernel call ** kDiamBlockWithCuda
	SharedMemBytes = THREADS_PER_BLOCK * 2 * sizeof(float);

	initArrToZeroes(&kDiameters, ksize);
	
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	
	///////////////////////////
	// allocate device memory
	{
	size_t nDataBytes = N * 2 * sizeof(float);  // N x 2 x sizeof(float)
	xyArrays da_xya;  // device anchor for copying xy-arrays data
	cudaMalloc(&(da_xya.x), nDataBytes / 2); CHKMAL_ERROR;
	cudaMalloc(&(da_xya.y), nDataBytes / 2); CHKMAL_ERROR;
	cudaMemcpy(da_xya.x, xya->x, nDataBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
	cudaMemcpy(da_xya.y, xya->y, nDataBytes / 2, cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;

	cudaMalloc(&d_xya, sizeof(xyArrays));
	cudaMemcpy(d_xya, &da_xya, sizeof(xyArrays), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;

	cudaMalloc(&d_kDiameters, ksize * sizeof(float));  //float* d_kDiameters;
	cudaMemcpy(d_kDiameters, kDiameters, ksize * sizeof(float), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;

	cudaMalloc(&d_pka, N * sizeof(int)); CHKMAL_ERROR;
	cudaMemcpy(d_pka, pka, N * sizeof(int), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
	}

	////////////////////////////
	//MPI single -- working out the single BLOCK problem with CUDA
	if (myid == MASTER)
	{

#ifdef _DEBUGV
		printf("NO_BLOCKS: %d\n", NO_BLOCKS); FF;
		printf("Proc %d, Attempting FirstJob, Blocks %2d, %2d\n", myid, 0, 0); fflush(stdout);
#endif
#ifdef _PROF_DIAM_BLOCK_KERNEL
		diamKerStart = omp_get_wtime();
#endif
		kDiamBlockWithCuda << <1, THREADS_PER_BLOCK, SharedMemBytes >> > (d_kDiameters, ksize, d_xya, d_pka, N, 0, 0);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "id: %d, kernel kDiamBlockWithCuda launch failed: %s\n", myid, cudaGetErrorString(cudaStatus)); FF;
			size_t RequestedRegistersPerBlock = THREADS_PER_BLOCK * (4 * sizeof(float) + 2 * sizeof(unsigned int) + 4 * sizeof(int));
			printf("__global__ kDiamBlockWithCuda() call error!!!\n\n"); FF;
			printf("__global__ kDiamBlockWithCuda() call with %d SharedMemBytes\n", SharedMemBytes); FF;
			printf("__global__ kDiamBlockWithCuda() call with %d RequestedRegistersPerBlock (approx.)\n", RequestedRegistersPerBlock); FF;
			printf("**********************************************\n\n", RequestedRegistersPerBlock); FF;
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize(); CHKSYNC_ERROR;
#ifdef __PROF_DIAM_BLOCK_KERNEL
		diamKerFinish = omp_get_wtime();
		if (myid == MASTER) { printf("kDiamBlock %d run-time: %f\n", ksize, diamKerFinish - diamKerStart); FF; }
#endif
		cudaStatus = cudaMemcpy(kDiameters, d_kDiameters, ksize * sizeof(float), cudaMemcpyDeviceToHost); CHKMEMCPY_ERROR;

#ifdef _DEBUG1
		//TEST kDiameters 
		printArrTestPrint(myid, kDiameters, ksize, "kDiameters");
#endif
	}

	///////////////////////////////////////////////////////////////
	//MASTER-SLAVES & streams
	float* kDiametersTempAnswer, *kDiametersAnswer;
	int streamIdx = 0;
	kDiametersTempAnswer = (float*)malloc(ksize * sizeof(float));

	//  allocate and initialize an array of NUM_STREAMS stream handles
	cudaStream_t *streams = (cudaStream_t *)malloc(NUM_CONCUR_KERNELS * sizeof(cudaStream_t));
	for (int streamIdx = 0; streamIdx < NUM_CONCUR_KERNELS; streamIdx++)
	{
		cudaStreamCreate(&(streams[streamIdx])); STREAMCR_ERROR;
	}
	int numJobsCounter = 0;

	if (myid == MASTER)
	{
		int x, const NO_JOBS = (ceil(NO_BLOCKS / 2.0f) + 1) * ceil(NO_BLOCKS / 2.0f) / 2;   // core: NO_JOBS = 1 + 2 + 3 + ... + NO_BLOCKS = (blocks+1)*blocks/2
																							// blocks adjusted (/2.0f) for a x2 kernel.
		int* jobs = initJobArray(NO_BLOCKS, NO_JOBS);
		int resultsCounter = 1;  // master already solved one job on its own.

		// distribute work to SLAVES (p < numprocs protects if numprocs==1)
		int p;	// process number
		for (p=1, x = 1; p < numprocs && x < NO_JOBS; ++p)
		{
			if (x + NUM_CONCUR_KERNELS - 1 < NO_JOBS) numJobsCounter = NUM_CONCUR_KERNELS;
			else numJobsCounter = NO_JOBS - x;
#ifdef _DEBUGV
			printf("Sending %d numJobsCounter to process %d\n", numJobsCounter, p); FF;
#endif
			MPI_Send(&numJobsCounter, 1, MPI_INT, p, NEW_JOB, MPI_COMM_WORLD); x++;
			for (int i = 0; i < numJobsCounter; i++)
			{
				MPI_Send(&jobs[2 * x], 2, MPI_INT, p, NEW_JOB, MPI_COMM_WORLD); x++;
			}
		}
		// edge case: less processors than jobs, notify slaves they don't need to work
		for (; p < numprocs; ++p)
		{
			numJobsCounter = 0;
			MPI_Send(&numJobsCounter, 1, MPI_INT, p, STOP_WORKING, MPI_COMM_WORLD);
		}
		
		// dynamically allocate further jobs as results are coming in
		while (resultsCounter < NO_JOBS)
		{
#ifdef _DEBUGV
			printf("x value %2d, results count: %2d\n", x, resultsCounter); fflush(stdout);
#endif

			if (numprocs > 1)  // MASTER-SLAVES
			{
				//TODO complete algorithm for master-slave with multi-kernels
				MPI_Recv(kDiametersTempAnswer, ksize, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				resultsCounter += status.MPI_TAG; // slaves report their numJobsCounter in the tag

				ompMaxVectors(&kDiameters, kDiametersTempAnswer, ksize);

#ifdef _DEBUG1
				//TEST kDiameters 

				//TODO rm commenting for runtime
				//printf("\nMaster values after MaxVectors with source %d !!\n", status.MPI_SOURCE); FF;
				printf("\nMaster values after MaxVectors with source %d !!\n", status.MPI_SOURCE); FF;
				printArrTestPrint(myid, kDiameters, ksize, "kDiameters");
				printf("***********************************************\n\n"); FF;
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
			else
			////////////////////////////////////////
			// only the "MASTER" works
			{ 
#ifdef _DEBUG1
				printf("Proc %d, working on jobForBlocks %2d, %2d (stream %d)\n", myid, jobs[2 * x], jobs[2 * x + 1], streamIdx); fflush(stdout);
#endif
				numJobsCounter = 0;
				kDiamBlockWithCuda << <1, THREADS_PER_BLOCK, SharedMemBytes, streams[streamIdx] >> > (d_kDiameters, ksize, d_xya, d_pka, N, jobs[2 * x], jobs[2 * x + 1]); x++;
				streamIdx = (streamIdx + 1) % NUM_CONCUR_KERNELS;
				numJobsCounter++;
				if (x < NO_JOBS)
				{
					kDiamBlockWithCuda << <1, THREADS_PER_BLOCK, SharedMemBytes, streams[streamIdx] >> > (d_kDiameters, ksize, d_xya, d_pka, N, jobs[2 * x], jobs[2 * x + 1]); x++;
					cudaStatus = cudaGetLastError();
					if (cudaStatus != cudaSuccess)
					{
						fprintf(stderr, "id: %d, kernel kDiamBlockWithCuda launch failed: %s\n", myid, cudaGetErrorString(cudaStatus)); FF;
						goto Error;
					}
					streamIdx = (streamIdx + 1) % NUM_CONCUR_KERNELS; 
					numJobsCounter++;
				}

			}
			cudaStatus = cudaDeviceSynchronize(); CHKSYNC_ERROR;

			cudaStatus = cudaMemcpy(kDiametersTempAnswer, d_kDiameters, ksize * sizeof(float), cudaMemcpyDeviceToHost); CHKMEMCPY_ERROR;

			resultsCounter += numJobsCounter;

			ompMaxVectors(&kDiameters, kDiametersTempAnswer, ksize);


#ifdef _DEBUG1
			//TEST kDiameters 

			//TODO rm commenting for runtime
			//printf("\nMaster values after MaxVectors with source %d !!\n", status.MPI_SOURCE); FF;
			printf("\nMaster values after MaxVectors with source %d !!\n", MASTER); FF;
			printArrTestPrint(myid, kDiameters, ksize, "kDiameters");
			printf("***********************************************\n\n"); FF;
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
	//////////////////////////////////////////////////////////
	// SLAVES
	else {  //slaves
		int masterTag = NEW_JOB;
		int jobForBlocks[2];
		int jobIdx = 0;
		
		/// get to work
		while (masterTag == NEW_JOB)
		{
			MPI_Recv(&numJobsCounter, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			masterTag = status.MPI_TAG;
			if (masterTag == NEW_JOB)
			{
				for (streamIdx = 0, jobIdx = 0; jobIdx < numJobsCounter; ++jobIdx)
				{
					MPI_Recv(jobForBlocks, 2, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				
#ifdef _DEBUGV
					printf("Proc %d, working on jobForBlocks %2d, %2d, streamIdx %d\n", myid, jobForBlocks[0], jobForBlocks[1], streamIdx); fflush(stdout);
#endif
					// queue nkernels in separate streams and record when they are done
					kDiamBlockWithCuda << <1, THREADS_PER_BLOCK, SharedMemBytes, streams[streamIdx] >> > (d_kDiameters, ksize, d_xya, d_pka, N, jobForBlocks[0], jobForBlocks[1]);
					cudaStatus = cudaGetLastError();
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "id: %d, kDiamBlockWithCuda<<%d, %d, %d, %d>> launch failed for data: %ld, %d, %d\n",
							myid, 1, THREADS_PER_BLOCK, SharedMemBytes, streamIdx, N, jobForBlocks[0], jobForBlocks[1]); FF;
						fprintf(stderr, "id: %d, main kDiamBlockWithCuda launch failed: %s\n", myid, cudaGetErrorString(cudaStatus)); FF;
						goto Error;
					}

					streamIdx = streamIdx + 1 % NUM_CONCUR_KERNELS;
				}
				cudaStatus = cudaDeviceSynchronize(); CHKSYNC_ERROR;
				cudaStatus = cudaMemcpy(kDiameters, d_kDiameters, ksize * sizeof(float), cudaMemcpyDeviceToHost); CHKMEMCPY_ERROR;
#ifdef _DEBUG1
				//TEST kDiameters 
				printArrTestPrint(myid, kDiameters, ksize, "kDiameters");
#endif
				MPI_Send(kDiameters, ksize, MPI_FLOAT, 0, numJobsCounter, MPI_COMM_WORLD);	   // report your numJobsCounter to master in tag
			}
			else
			/////////////////////////////
			// No more work. Exit.
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

