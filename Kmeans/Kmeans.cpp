/*
Author: Ofer Sheffer, 1 April 2017

Important note:
----------------
for weak processors, uncomment #define _WEAKGPU from the header file

*/

#include "Kmeans.h"

//NOTE: use \\ in systems were / does not work
#define FILE_NAME "C:\\Users\\MPICH\\Documents\\Visual Studio 2015\\Projects\\KMeansProject\\Kmeans\\cluster1.txt"
//#define FILE_NAME "D:\\cluster1.txt"
#define NO_OMP_THREADS 4	// OMP: 4 core laptop
#define MASTER 0

/////////// GPU controls
int THREADS_PER_BLOCK;
int _gpuReduction;
size_t SharedMemBytes;




long  N;			// number of points. e.g. 300000
int   MAX;			// maximum # of clusters to find. e.g. 300
int	  LIMIT;		// maximum # of iterations for K-means algorithm. e.g. 2000
float QM;			// quality measure to stop. e.g. 17

xyArrays *xya;		// SoA of the data
xyArrays *kCenters;	// SoA of k-mean vertices
int *pka;			// pka (points' cluster association) : array to associate xya points with their closest cluster
float kQuality;		// quality of current cluster configuration;

int numprocs, myid;

int main(int argc, char *argv[])
{
	//int res;
	double start, finish;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Status status;

	if (myid == MASTER)
	{
		//int devID;
		//cudaDeviceProp props;
		//cudaGetDevice(&devID);
		//cudaGetDeviceProperties(&props, devID);
		//printf("> Compute %d.%d CUDA device: [%s]\n", props.major, props.minor, props.name); FF;
		//printf("  Total amount of shared memory per block:       %lu bytes\n", props.sharedMemPerBlock);
		//printf("  Total number of registers available per block: %d\n\n", props.regsPerBlock);
		initializeWithGpuReduction();

		printf("\t\t*************************************\n");
		printf("\t\t*****                          ******\n");
		printf("\t\t*****     Kmeans algorithm     ******\n");
		printf("\t\t*****   Author: Ofer Sheffer   ******\n");
		printf("\t\t*****                          ******\n");
		printf("\t\t*************************************\n");
		

#ifdef _DEBUGV
		printf("\n" FILE_NAME "\n"); FF;
#endif

		readPointsFromFile();			 // init xya with data
		printf("\t\tN=%ld, Max=%d, LIMIT=%d, QM=%f\n\n\n", N, MAX, LIMIT, QM); FF;

		if (numprocs > 1)
			MPI_Bcast(&N, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Bcast(&N, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);
		mallocSoA(&xya, N);
	}
	initClusterAssociationArrays();  // no cluster (-1)

	const int NO_BLOCKS = ceil(N / (float)THREADS_PER_BLOCK);

	const int THREAD_BLOCK_SIZE = THREADS_PER_BLOCK;

#ifdef _TIME
	start = omp_get_wtime();
#endif

	if (numprocs > 1)
	{
		//send points to slaves
		MPI_Bcast(&(xya->x[0]), N, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(&(xya->y[0]), N, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
	}
	float* kDiameters;
	if (myid == MASTER)
	{

		for (long ksize = 2; ksize <= MAX; ksize++)
		{
#ifdef _DEBUGV
			printf("Full algorithm -- on ksize: %d\n***********************\n", ksize); fflush(stdout);
#endif
			if (numprocs > 1)
				MPI_Bcast(&ksize, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);
			mallocSoA(&kCenters, ksize);
			kDiameters = (float*)malloc(ksize * sizeof(float));

			// allocate host memory
			size_t nBytes = sizeof(xya);

			// *** kCentersWithCuda ***
			cudaError_t cudaStatus = kCentersWithCuda(kCenters, ksize, xya, pka, N, LIMIT, THREADS_PER_BLOCK);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "kCentersWithCuda failed!\n"); FF;
				freeSoA(xya);
				free(pka);
				free(kDiameters);
				MPI_Finalize();
				return 1;
			}

#ifdef _DEBUGV
			printf("k-complete calculated centers are:\n");
			printArrTestPrint(MASTER, kCenters->x, ksize, "ompMaster - kCentersX");
			printArrTestPrint(MASTER, kCenters->y, ksize, "ompMaster - kCentersY");
			printf("********************************************************\n\n");
#endif

			// *** kDiametersWithCuda ***
			/*
			Core concept:
			-------------
			1) Break data into blocks of size THREADS_PER_BLOCK. e.g. 100,000/1024=>10 Blocks (Bn for short)
			2) Run kernel O(Bn^2) on 2 block permutations to receive all possible distances (reduceMax to get diameters)
			Main benefit:
			Instead of O(N^2), we get O(Bn^2). Adding dynamic MPI based scheduling increases performance further.
			*/
			if (numprocs > 1)
				MPI_Bcast(pka, N, MPI_INT, MASTER, MPI_COMM_WORLD);

			cudaStatus = kDiametersWithCuda(kDiameters, ksize, xya, pka, N, myid, numprocs, THREADS_PER_BLOCK);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "kDiametersWithCuda failed!\n"); FF;
				return 1;
			}

#ifdef _DEBUG4
			//TEST kDiameters 
			printf("\nkDiameters work complete:\n"
				"------------------------\n"); FF;
			printArrTestPrint(myid, kDiameters, ksize, "kDiameters");
#endif

			//sum kQuality for every (center_i,center_j) combo (i < j): (d_i+d_j)/distance(i,j)
			float kQuality = 0;
#pragma omp parallel for reduction(+:kQuality)
			for (int i = 0; i < ksize; i++)
			{
				for (int j = 0; j < ksize; j++)
				{
					if (i != j)
						kQuality += kDiameters[i] / sqrtf((powf(kCenters->x[i] - kCenters->x[j], 2)) + (powf(kCenters->y[i] - kCenters->y[j], 2)));
				}
			}

#ifdef _DEBUG5
			//TEST Quality
			printf("\nQuality work complete:   ***  %f  *** Smaller than %f?  * %d *\n\n*********************************************\n", kQuality, QM, (kQuality <= QM)); FF;
#endif


#ifdef _TIMEK
			finish = omp_get_wtime();
			if (myid == MASTER) { printf("run-time: %f\n", finish - start); FF; }
#endif




			if (kQuality <= QM)
			{
#ifdef _TIME
				finish = omp_get_wtime();
#endif
				//print to file
				printArrTestPrint(MASTER, kCenters->x, ksize, "kCentersX");
				printArrTestPrint(MASTER, kCenters->y, ksize, "kCentersY");
				printf("********************************************************\n\n"); FF;
				//let the slaves know the job is done
				ksize = 0;
				if (numprocs > 1)
					MPI_Bcast(&ksize, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);
				printf("\nQuality work complete:   ***  %f  *** Smaller than %f?  * %d *\n\n*********************************************\n", kQuality, QM, (kQuality <= QM)); FF;

				cudaStatus = cudaDeviceReset();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceReset failed!\n"); FF;
					return 1;
				}

				break;
			}


			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceReset failed!\n"); FF;
				return 1;
			}
		}
	}
	else
	{
		// slaves
		long ksize = 1;
		while (true)
		{
			cudaError_t cudaStatus;
			MPI_Bcast(&ksize, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);
			if (ksize == 0)
			{
				cudaStatus = cudaDeviceReset();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceReset failed!");
					return 1;
				}
				//Slave stops working
				break;
			}

			mallocSoA(&kCenters, ksize);
			kDiameters = (float*)malloc(ksize * sizeof(float));
			MPI_Bcast(pka, N, MPI_INT, MASTER, MPI_COMM_WORLD);


			//getKQuality();
			cudaStatus = kDiametersWithCuda(kDiameters, ksize, xya, pka, N, myid, numprocs, THREADS_PER_BLOCK);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "kCentersWithCuda failed!\n"); FF;
				return 1;
			}


			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceReset failed!\n"); FF;
				return 1;
			}

		}
	}

	freeSoA(kCenters);

#ifdef _DEBUGV
	printf("Process %d signing off.\n", myid); FF;
#endif

#ifdef _TIME
	if (myid == MASTER) { printf("run-time: %f\n", finish - start); FF; }
#endif
	freeSoA(xya);
	free(pka);
	free(kDiameters);
	MPI_Finalize();
	return 0;
}


void readPointsFromFile()
{
	FILE *fp;
	fp = fopen(FILE_NAME, "r");
	if (fp == NULL) {
		fprintf(stderr, "Can't open input file\n"); fflush(stderr);
		MPI_Finalize();
		exit(1);
	}
	fscanf(fp, "%ld %d %d %f", &N, &MAX, &LIMIT, &QM);	// obtain core info from first line	
	populateSoA(fp);										// populate data into xya
	fclose(fp);
}

void initClusterAssociationArrays()
{
	/* pka -> pointClusterAssociation */
	pka = (int*)malloc(N * sizeof(int));

	for (long i = 0; i < N; i++)
	{
		pka[i] = -1; // no cluster
	}
}






void populateSoA(FILE* fp)
{
	mallocSoA(&xya, N);

	for (long i = 0; i < N; i++)
	{
		fscanf(fp, "%d %f %f", &i, &(xya->x[i]), &(xya->y[i]));
	}

}



bool ompReduceCudaFlags(bool* flags, int size)
{
	bool flag = false;
#pragma omp parallel for reduction(|:flag)
	for (int i = 0; i < size; i++)
	{
		flag |= flags[i];
	}
	return flag;
}

void ompRecenterFromCuda(int ksize)
{
	// re-calculate cluster centers:
	// *****************************
	/*
	each thread given pseudo-private 1D arrays of size:  threads# x k#
	e.g. ksize = 3, threads = 4:
	step 1:
	indices 0,4,8 will be accessed by thread 0: e.g. [4] sum x that belongs to cluster 1
	indices 1,5,9 will be accessed by thread 1

	step 2: sum each row (e.g. row 0) to get totals of k[0]
	i.e. x_tot, y_tot, count_tot
	step 3: divide x_tot,y_tot by count_tot for new k[0] center value.
	*/

	float* ompSumXArr = (float*)calloc(NO_OMP_THREADS * ksize, sizeof(float));
	float* ompSumYArr = (float*)calloc(NO_OMP_THREADS * ksize, sizeof(float));
	int*   ompCntPArr = (int*)calloc(NO_OMP_THREADS * ksize, sizeof(int));

	// step 1:
#pragma omp parallel for num_threads(NO_OMP_THREADS)
	//for (long i = 0; i < N; i++)
	for (int i = 0; i < N; i++)
	{
		int ompArrIdx = pka[i] * NO_OMP_THREADS + omp_get_thread_num();  // row: pka[i]*NO_OMP_THREADS, col: thread id
		ompCntPArr[ompArrIdx]++;
		ompSumXArr[ompArrIdx] += xya->x[i];
		ompSumYArr[ompArrIdx] += xya->y[i];
	}

	// steps 2+3:
	//TODO: test prepK erases only what it should erase and keeps values it should not touch (clusters without points)
	prepK(ompCntPArr, ksize);



#pragma omp parallel for
	for (int idx = 0; idx < ksize; idx++)
	{
		//TODO: gotta decide where in kCenters to change the value....

		long count = 0;
		for (int i = idx*NO_OMP_THREADS; i < idx*NO_OMP_THREADS + NO_OMP_THREADS; i++)
		{
			kCenters->x[idx] += ompSumXArr[i];
			kCenters->y[idx] += ompSumYArr[i];
			count += ompCntPArr[i];
		}
		//complete center calculation
		kCenters->x[idx] /= count;
		kCenters->y[idx] /= count;
	}
#ifndef _RUNAFEKA
	free(ompSumXArr);
	free(ompSumYArr);
	free(ompCntPArr);
#endif

}

void ompMaxVectors(float** kDiameters, float* kDiametersTempAnswer, int ksize)
{
#pragma omp parallel for
	for (int i = 0; i < ksize; i++)
	{
		if ((*kDiameters)[i] < kDiametersTempAnswer[i])
			(*kDiameters)[i] = kDiametersTempAnswer[i];
	}
}

void mallocSoA(xyArrays** soa, long size)
{
	*soa = (xyArrays*)malloc(sizeof(xyArrays));
	(*soa)->x = (float*)malloc(size * sizeof(float));
	(*soa)->y = (float*)malloc(size * sizeof(float));
}

void freeSoA(xyArrays* soa)
{
	free(soa->x);
	free(soa->y);
	free(soa);
}

void initK(long ksize)
{
	for (long i = 0; i < ksize; i++)
	{
		kCenters->x[i] = xya->x[i];
		kCenters->y[i] = xya->y[i];
	}
}

void prepK(int* ompCntPArr, long ksize)
{
	// reset Kxy to 0 where there are points in the new assignments
#pragma omp parallel for
	for (int idx = 0; idx < ksize; idx++)
	{
		for (int i = idx*NO_OMP_THREADS; i < idx*NO_OMP_THREADS + NO_OMP_THREADS; i++)
		{
			if (ompCntPArr[i]>0)
			{
				kCenters->x[idx] = 0;
				kCenters->y[idx] = 0;
				break;
			}
		}
	}
}

void getNewPointKCenterAssociation(long i, int ksize)
{

	float minSquareDist = INFINITY;

	float curSquareDist;
	for (long idx = 0; idx < ksize; idx++)
	{
		curSquareDist = powf(xya->x[i] - kCenters->x[idx], 2) + powf(xya->y[i] - kCenters->y[idx], 2);
		if (curSquareDist < minSquareDist)
		{
			minSquareDist = curSquareDist;
			pka[i] = idx;
		}
	}
}

int* initJobArray(int NO_BLOCKS, int fact)
{
	int* jobs = (int*)malloc(2 * fact * sizeof(int));
	int jidx = 0;
	for (int i = 0; i < NO_BLOCKS; i++)
		for (int j = 0; j < NO_BLOCKS; j++)
		{
			if (i <= j)
			{
				jobs[jidx++] = i;
				jobs[jidx++] = j;
			}

		}
	return jobs;
}

void printArrTestPrint(int myid, float* arr, int size, const char* arrName)
{
	char* user = (myid == MASTER) ? "Master" : "slave";
	switch (size)
	{
	case 2:
		printf("%6s %d - %s%d-%d: %f, %f\n", user, myid, arrName, 0, 1, arr[0], arr[1], arr[2], arr[3], arr[4]); fflush(stdout);
		break;
	case 3:
		printf("%6s %d - %s%d-%d: %f, %f, %f\n", user, myid, arrName, 0, 2, arr[0], arr[1], arr[2], arr[3], arr[4]); fflush(stdout);
		break;
	case 4:
		printf("%6s %d - %s%d-%d: %f, %f, %f, %f\n", user, myid, arrName, 0, 3, arr[0], arr[1], arr[2], arr[3], arr[4]); fflush(stdout);
		break;
	case 5:
		printf("%6s %d - %s%d-%d: %f, %f, %f, %f, %f\n", user, myid, arrName, 0, 4, arr[0], arr[1], arr[2], arr[3], arr[4]); fflush(stdout);
		break;
	case 6:
		printf("%6s %d - %s%d-%d: %f, %f, %f, %f, %f, %f\n", user, myid, arrName, 0, 5, arr[0], arr[1], arr[2], arr[3], arr[4]); fflush(stdout);
		break;
	case 7:
		printf("%6s %d - %s%d-%d: %f, %f, %f, %f, %f, %f, %f\n", user, myid, arrName, 0, 6, arr[0], arr[1], arr[2], arr[3], arr[4]); fflush(stdout);
		break;
	case 8:
		printf("%6s %d - %s%d-%d: %f, %f, %f, %f, %f, %f, %f, %f\n", user, myid, arrName, 0, 7, arr[0], arr[1], arr[2], arr[3], arr[4]); fflush(stdout);
		break;
	default:
		for (int i = 0; i < size; i++) { printf("%6s %d - %s%d: %f\n", user, myid, arrName, i, arr[i]); fflush(stdout); }
		break;
	}
}


void initializeWithGpuReduction()
{
	/* replacing
		#ifndef _WEAKGPU
		#define THREADS_PER_BLOCK 1024  // replacement for THREAD_BLOCK_SIZE or blockDim.x
		#else
		#ifndef _WEAKGPU2
		#define THREADS_PER_BLOCK 512	// weak gpu
		#endif //_WEAKGPU2
		#define THREADS_PER_BLOCK 128	// very weak gpu
		#endif // not _WEAKGPU
	*/
	int devID;
	cudaDeviceProp props;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&props, devID);

	///*** Current GPU Memory requirements  (unprofiled -- my estimations) ***/
	while (true)
	{
		THREADS_PER_BLOCK = BASE_THREADS_PER_BLOCK / pow(2, _gpuReduction);

		//// ****   SharedMemBytes   ****
		/********************************/

		//// ** reClusterWithCuda  -----		THREADS_PER_BLOCK * sizeof(bool);
		//// ** kDiamBlockWithCuda -----		THREADS_PER_BLOCK * 2 * sizeof(float);  // MAX
		size_t maxRequestedSharedMemBytes = 2 * THREADS_PER_BLOCK * sizeof(float);


		//// ****   RegistersPerBlock   ****
		/***********************************/

		//// ** reClusterWithCuda  -----		THREADS_PER_BLOCK * (1 * sizeof(float) + 3 * sizeof(unsigned int) + 2 * sizeof(int));
		//// ** kDiamBlockWithCuda -----		THREADS_PER_BLOCK * (4 * sizeof(float) + 2 * sizeof(unsigned int) + 4 * sizeof(int));  // MAX
		size_t RequestedRegistersPerBlock = THREADS_PER_BLOCK * (4 * sizeof(float) + 2 * sizeof(unsigned int) + 4 * sizeof(int));

		/*printf("  Total amount of shared memory per block:       %lu bytes\n", props.sharedMemPerBlock); FF;
		printf("  Total number of registers available per block: %d\n\n", props.regsPerBlock); FF;*/

		if (props.sharedMemPerBlock < maxRequestedSharedMemBytes || 
			props.regsPerBlock < RequestedRegistersPerBlock)
		{
			_gpuReduction += 1;
		}
		else if (props.sharedMemPerBlock > 2*maxRequestedSharedMemBytes &&
				 props.regsPerBlock > 2*RequestedRegistersPerBlock)
		{
			_gpuReduction -= 1;
		}
		else
		{
			printf("> Compute %d.%d CUDA device: [%s]\n", props.major, props.minor, props.name); FF;
			printf("  Kernel/Gpu run with 2^%d reduction factor\n"
				"  THREADS_PER_BLOCK:                         %7d /%7d\n"
				"  Per block Shared memory usage:             %7lu /%7lu bytes\n"
				"  Per block register usage (estimated):      %7d /%7d\n\n", _gpuReduction, THREADS_PER_BLOCK, props.maxThreadsPerBlock,
																	maxRequestedSharedMemBytes, props.sharedMemPerBlock, 
																	RequestedRegistersPerBlock, props.regsPerBlock); FF;
			break;
		}
	}	// while
}
