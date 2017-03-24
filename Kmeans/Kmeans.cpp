#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "Kmeans.h"

#define NO_OMP_THREADS 4	// OMP: 4 core laptop


//TODO: consider adding a delimiter (testfile: ',' -- presentation: ' ')

long  N;			// number of points. e.g. 300000
int   MAX;			// maximum # of clusters to find. e.g. 300
int	  LIMIT;		// maximum # of iterations for K-means algorithm. e.g. 2000
float QM;			// quality measure to stop. e.g. 17

xyArrays *xya;		// SoA of the data
xyArrays *karray;	// SoA of k-mean vertices
int *pka;			// array to associate xya points with their closest cluster
float kQuality;		// quality of current cluster configuration;


//TODO: use single/double precision in the CUDA computations (half?)

int main()
{
	double start, finish;
	readPointsFromFile();
	initClusterAssociationArrays();

	start = omp_get_wtime();

	//for (long ksize = 2; ksize <= MAX; ksize++)
	for (long ksize = 20; ksize <= 20; ksize++)
	{
		//TODO quick test
		//printf("ksize: %d\n", ksize);
		int iter = 0;
		mallocSoA(&karray, ksize);
		initK(ksize);
		bool kAssociationChangedFlag = true;
		do {
			//printf("iter %d\n", iter + 1);
			kAssociationChangedFlag = reCluster(ksize);
		} while (++iter < LIMIT && kAssociationChangedFlag);  // association changes: need to re-cluster


		//TODO: getKQuality();
		//TODO1: for every cluster - get diameter



		//TODO2: sum for every (center_i,center_j) combo (i < j): (d_i+d_j)/distance(i,j)


		//if (kQuality < QM)
		if (true)
		{
			//quicktest
			printf("karray:\n");
			for (int i = 0; i < ksize; i++)
				printf("%d, %6.3f, %6.3f\n", i, karray->x[i], karray->y[i]);
			//TODO: print to file
			break;
		}
			

	}
	
	freeSoA(karray);


	finish = omp_get_wtime();
	printf("run-time: %f\n", finish - start);


	//old cuda code
	{
	//const int arraySize = 5;
	//const int a[arraySize] = { 1, 2, 3, 4, 5 };
	//const int b[arraySize] = { 10, 20, 30, 40, 50 };
	//int c[arraySize] = { 0 };



	//// Add vectors in parallel.
	//cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "addWithCuda failed!");
	//	return 1;
	//}

	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
	//	c[0], c[1], c[2], c[3], c[4]);

	//// cudaDeviceReset must be called before exiting in order for profiling and
	//// tracing tools such as Nsight and Visual Profiler to show complete traces.
	//cudaStatus = cudaDeviceReset();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceReset failed!");
	//	return 1;
	//}
	}

	freeSoA(xya);
	free(pka);
	return 0;
}


void readPointsFromFile()
{
	FILE *fp;
	fp = fopen("large_testinput_guy.txt", "r");
	if (fp == NULL) {
		fprintf(stderr, "Can't open input file\n");
		exit(1);
	}

	fscanf(fp, "%ld, %d, %d, %f", &N, &MAX, &LIMIT, &QM);	// obtain core info from first line	
	populateSoA(fp);										// populate data into xya

	fclose(fp);
}

void populateSoA(FILE* fp)
{
	mallocSoA(&xya, N);

	for (long i = 0; i < N; i++)
		fscanf(fp, "%d, %f, %f", &i, &(xya->x[i]), &(xya->y)[i]);
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

	//reCenter
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
		//TODO: gotta decide where in karray to change the value....

		long count = 0;
		for (int i = idx*NO_OMP_THREADS; i < idx*NO_OMP_THREADS + NO_OMP_THREADS; i++)
		{
			karray->x[idx] += ompSumXArr[i];
			karray->y[idx] += ompSumYArr[i];
			count += ompCntPArr[i];
		}
		//complete center calculation
		karray->x[idx] /= count;
		karray->y[idx] /= count;
	}

	free(ompSumXArr);
	free(ompSumYArr);
	free(ompCntPArr);


	return kAssociationChangedFlag;
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
		karray->x[i] = xya->x[i];
		karray->y[i] = xya->y[i];
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
				karray->x[idx] = 0;
				karray->y[idx] = 0;
				break;
			}
		}
	}
}

void initClusterAssociationArrays()
{
	pka = (int*)malloc(N * sizeof(int));

	for (long i = 0; i < N; i++)
	{
		pka[i] = -1; // no cluster
	}
}

void getNewPointKCenterAssociation(long i, int ksize)
{
	float minSquareDist = INFINITY;
	float curSquareDist;
	for (long idx = 0; idx < ksize; idx++)
	{
		curSquareDist = powf(xya->x[i] - karray->x[idx], 2) + powf(xya->y[i] - karray->y[idx], 2);
		if (curSquareDist < minSquareDist)
		{
			minSquareDist = curSquareDist;
			pka[i] = idx;
		}
	}
}


//TODO quick test
//for (int i = 0; i < ksize; i++)
//{
//	printf("%d, %f, %f\n", i, karray->x[i], karray->y[i]);
//}

//TODO quick test
//for (int i = 0; i < 10; i++)
//{
//	printf("%d: %6.3f, %6.3f Closest to K-idx: %d\n", i, xya->x[i], xya->y[i], pka[i]);
//}

//TODO quick omp thread check
//printf("id: %d, running i: %d\n", omp_get_thread_num(), i);