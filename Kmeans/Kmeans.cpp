#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "Kmeans.h"

//TODO: consider adding a delimiter (testfile: ',' -- presentation: ' ')

long  N;			// number of points. e.g. 300000
int   MAX;			// maximum # of clusters to find. e.g. 300
int	  LIMIT;		// maximum # of iterations for K-means algorithm. e.g. 2000
float QM;			// quality measure to stop. e.g. 17

xyArrays *xya;		// SoA of the data
xyArrays *karray;	// SoA of k-mean vertices
int *pka;			// array to associate xya points with their closest cluster
float *pkx;			// array to collect data to re-calculate cluster centers
float *pky;			// array to collect data to re-calculate cluster centers


//TODO: use single/double precision in the CUDA computations (half?)

int main()
{
	double start, finish;
	readPointsFromFile();
	initClusterAssociationArrays();

	start = omp_get_wtime();

	//for (long ksize = 2; ksize <= MAX; ksize++)
	for (long ksize = 2; ksize <= 4; ksize++)
	{
		//TODO quick test
		//printf("ksize: %d\n", ksize);

		mallocSoA(&karray, ksize);
		initK(ksize);

		//************************************************************/
		// for every point: save idx where min(distance from k[idx])
		//************************************************************/

		// omp stinks
		{
				bool kAssociationChangedFlag = false;
				//for (long i = 0; i < N; i++)
				#pragma omp parallel for reduction(|:kAssociationChangedFlag)
				for (long i = 0; i < 20; i++)
				{
					printf("id: %d, running i: %d\n", omp_get_thread_num(), i);
					int prevPka = pka[i];  // save associated cluster idx

					//option A:		
					getNewPointKCenterAssociation(i, ksize);


					// TODO: save data for re-calculation of cluster centers 


					if (pka[i] != prevPka)
					{
						kAssociationChangedFlag = true;
					}
				}

				//TODO quick test
				//printf("karray:\n");
				//for (int i = 0; i < ksize; i++)
				//	printf("%d, %6.3f, %6.3f\n", i, karray->x[i], karray->y[i]);
				//for (int i = 0; i < 10; i++)
				//{
				//	printf("%d: %6.3f, %6.3f Closest to K-idx: %d\n", i, xya->x[i], xya->y[i], pka[i]);
				//}
		}




		//TODO: if kAssociationChangedFlag, recalculate cluster centers... yada yada

		//TODO: if breakcondition: break
		freeSoA(karray);
	}

	finish = omp_get_wtime();
	printf("option a: %f\n", finish - start);


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