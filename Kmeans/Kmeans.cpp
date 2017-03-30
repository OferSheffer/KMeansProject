
#include "Kmeans.h"

//NOTE: use \\ in systems were / does not work
#define FILE_NAME "C:\\Users\\MPICH\\Documents\\Visual Studio 2015\\Projects\\KMeansProject\\Kmeans\\cluster1.txt"
//#define FILE_NAME "D:\\cluster1.txt"
#define NO_OMP_THREADS 4	// OMP: 4 core laptop
#define MASTER 0
#define THREADS_PER_BLOCK 1024

//TODO: review different ways to get better messages: currently works up to 40000 on my laptop. 
//TODO: note: different error messages between 50000 and 100000

long  N;			// number of points. e.g. 300000
int   MAX;			// maximum # of clusters to find. e.g. 300
int	  LIMIT;		// maximum # of iterations for K-means algorithm. e.g. 2000
float QM;			// quality measure to stop. e.g. 17

xyArrays *xya;		// SoA of the data
xyArrays *kCenters;	// SoA of k-mean vertices
int *pka;			// array to associate xya points with their closest cluster
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
#ifdef _DEBUGV
		printf(FILE_NAME"\n"); FF;
#endif
		readPointsFromFile();			 // init xya with data
		printf("N=%ld, Max=%d, LIMIT=%d, QM=%f\n\n", N, MAX,LIMIT,QM); FF;
		MPI_Bcast(&N, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Bcast(&N, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);
		mallocSoA(&xya, N);
	}
	initClusterAssociationArrays();  // no cluster (-1)
	
	const int NO_BLOCKS = (N % THREADS_PER_BLOCK == 0) ? N / THREADS_PER_BLOCK : N / THREADS_PER_BLOCK + 1;
	const int THREAD_BLOCK_SIZE = THREADS_PER_BLOCK;

#ifdef _DEBUGT
	start = omp_get_wtime();
#endif
	
	//send points to slaves
	MPI_Bcast(&(xya->x[0]), N, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&(xya->y[0]), N, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
	
	float* kDiameters;
	if (myid == MASTER)
	{
				
		//for (long ksize = 2; ksize <= MAX; ksize++)
		for (long ksize = 2; ksize <= MAX; ksize++)
		{
#ifdef _DEBUG3
			//ompGoTest(int initSize, int maxSize)
			ompGoTest(ksize, ksize);
#endif
#ifdef _DEBUGV
			printf("Full algorithm -- on ksize: %d\n***********************\n", ksize); fflush(stdout);
#endif
			MPI_Bcast(&ksize, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);
			mallocSoA(&kCenters, ksize);
			kDiameters = (float*)malloc(ksize * sizeof(float));

			// allocate host memory
			size_t nBytes = sizeof(xya);

			// *** kCentersWithCuda ***
			//cudaError_t cudaStatus = kCentersWithCuda(hist, &(myLargeArr[MY_ARR_SIZE / 2]), MY_ARR_SIZE / 2, VALUES_RANGE);
			cudaError_t cudaStatus = kCentersWithCuda(kCenters, ksize, xya, pka, N, LIMIT);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "kCentersWithCuda failed!");
				return 1;
			}
#ifdef _DEBUGV
			printf("k-complete calculated centers are:\n");
			printArrTestPrint(MASTER, kCenters->x, ksize, "ompMaster - kCentersX");
			printArrTestPrint(MASTER, kCenters->y, ksize, "ompMaster - kCentersY");
			printf("********************************************************\n\n");
#endif
			//getKQuality();
			//step1: for every cluster - get diameter
			//concept: 1) Master sends point arrays as well as pka array to slaves
			//master and slaves use a cuda kernel (maxSquareDistances) with a block of 1024 threads (less if it takes too long)
			//kernel receives two sections of the array and each thread of the 1024 registers its maximum
			//squared distance from those in the other 1024 who belong to the same cluster.
			//master dynamically sends computers which block to compare with which block
			//slaves send back the indices that gathe data to master to combine all 


			// *** kDiametersWithCuda ***
			MPI_Bcast(pka, N, MPI_INT, MASTER, MPI_COMM_WORLD);
			cudaStatus = kDiametersWithCuda(kDiameters, ksize, xya, pka, N, myid, numprocs);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "kCentersWithCuda failed!");
				return 1;
			}
#ifdef _DEBUG4
			//TEST kDiameters 
			printf("\nkDiameters work complete:\n"
				"------------------------\n"); fflush(stdout);
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

			if (kQuality <= QM)
			{
#ifdef _DEBUGT
				finish = omp_get_wtime();
#endif
				//print to file
				printArrTestPrint(MASTER, kCenters->x, ksize, "kCentersX");
				printArrTestPrint(MASTER, kCenters->y, ksize, "kCentersY");
				printf("********************************************************\n\n"); FF;
				//let the slaves know the job is done
				ksize = 0;
				MPI_Bcast(&ksize, 1, MPI_LONG, MASTER, MPI_COMM_WORLD);
				printf("\nQuality work complete:   ***  %f  *** Smaller than %f?  * %d *\n\n*********************************************\n", kQuality, QM, (kQuality <= QM)); FF;

				cudaStatus = cudaDeviceReset();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaDeviceReset failed!");
					return 1;
				}
				
				break;
			}


			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceReset failed!");
				return 1;
			}
		}
	}
	else 
	{
		// slaves
		long ksize=1;
		while(true)
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
			cudaStatus = kDiametersWithCuda(kDiameters, ksize, xya, pka, N, myid, numprocs);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "kCentersWithCuda failed!");
				return 1;
			}

		
			cudaStatus = cudaDeviceReset();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceReset failed!");
				return 1;
			}

		}
	}
	
	freeSoA(kCenters);

#ifdef _DEBUGV
	printf("Process %d signing off.\n", myid); FF;
#endif

#ifdef _DEBUGT
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
	//TODO: consider renaming pka -> pointClusterAssociation
	pka = (int*)malloc(N * sizeof(int));

	for (long i = 0; i < N; i++)
	{
		pka[i] = -1; // no cluster
	}
}

void ompGoTest(int initSize, int maxSize)
{
	//for (long ksize = 2; ksize <= MAX; ksize++)
	for (long ksize = initSize; ksize <= maxSize; ksize++)
	{
		printf("omp centers test -- ksize: %d\n", ksize); FF;
		int iter = 0;
		mallocSoA(&kCenters, ksize);
		initK(ksize);				// K-centers = first points in data 
		bool kAssociationChangedFlag = true;
		do {
			kAssociationChangedFlag = reCluster(ksize);
		} while (++iter < LIMIT && kAssociationChangedFlag);  // association changes: need to re-cluster

		printArrTestPrint(MASTER, kCenters->x, ksize, "ompMaster - kCentersX");
		printArrTestPrint(MASTER, kCenters->y, ksize, "ompMaster - kCentersY");
		printf("No QM test for ompGoTest\n\n"); FF;

	}

	freeSoA(kCenters);
}








void populateSoA(FILE* fp)
{
	mallocSoA(&xya, N);

	for (long i = 0; i < N; i++)
	{
		fscanf(fp, "%d %f %f", &i, &(xya->x[i]), &(xya->y[i]));
	}
		
}


bool reCluster(int ksize)
{
	bool kAssociationChangedFlag = false;

	// for every point: save idx where min(distance from k[idx])
	#pragma omp parallel for reduction(|:kAssociationChangedFlag)
	for (long i = 0; i < N; i++)
	{
		int prevPka = pka[i];  // save associated cluster idx
		getNewPointKCenterAssociation(i, ksize);
		if (pka[i] != prevPka)
		{
			kAssociationChangedFlag = true;
		}
	}

	// reCenter
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
	int*   ompCntPArr = (int*)  calloc(NO_OMP_THREADS * ksize, sizeof(int));

	// step 1:
	#pragma omp parallel for num_threads(NO_OMP_THREADS) // TODO: scheduling?
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

	return kAssociationChangedFlag;
}

bool ompReduceCudaFlags(bool* flags, int size)
{
	bool flag = false;
#pragma omp parallel for reduction(|:flag)
	for (int i = 0; i < size; i++)
	{
		//TODO: quicktest
		//printf("%d, %d\n", omp_get_thread_num(), flags[i]);
		flag |= flags[i];
	}
	//TODO: quicktest
	//printf("flag: %d!\n", flag);
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


//TODO quick test
//for (int i = 0; i < 10; i++)
//{
//	printf("%d: %6.3f, %6.3f Closest to K-idx: %d\n", i, xya->x[i], xya->y[i], pka[i]);
//}

//TODO quick omp thread check
//printf("id: %d, running i: %d\n", omp_get_thread_num(), i);

//MPI_TEST
//printf("id %d, %3d: %8.3f, %8.3f\n", myid, 0, (xya->x)[0], (xya->y)[0]); fflush(stdout);
//printf("id %d, %3d: %8.3f, %8.3f\n", myid, 1, (xya->x)[1], (xya->y)[1]); fflush(stdout);
//printf("id %d, %3d: %8.3f, %8.3f\n", myid, 126, (xya->x)[126], (xya->y)[126]); fflush(stdout);
//printf("id %d, %3d: %8.3f, %8.3f\n", myid, 127, (xya->x)[127], (xya->y)[127]); fflush(stdout);
//printf("id %d, %3d: %8.3f, %8.3f\n", myid, 510, (xya->x)[510], (xya->y)[510]); fflush(stdout);
//printf("id %d, %3d: %8.3f, %8.3f\n", myid, 511, (xya->x)[511], (xya->y)[511]); fflush(stdout);
//printf("id %d, %3d: %8.3f, %8.3f\n", myid, 999, (xya->x)[999], (xya->y)[999]); fflush(stdout);
//printf("id %d, %3d: %8.3f, %8.3f\n", myid, N - 1, (xya->x)[N - 1], (xya->y)[N - 1]); fflush(stdout);


// Considerations:
// - SoA helps increase load/store productivity
// - GPU Fine grain SIMD, low latency floating point computation, Streaming throughput of large data
// - CUDA - deducated super-threaded
// - GPU - lightweight threads, 1000s of threads for efficiency
// - CUDA reduce - decompose into multiple kernels: negligible HW overhead, low SW overhead (bandwidth-optimal)
// - GPU performance: Choose the right metric:
//		GFLOP / s: for compute - bound kernels
//		Bandwidth : for memory - bound kernels

