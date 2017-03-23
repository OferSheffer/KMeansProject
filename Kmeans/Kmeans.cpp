#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
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

int main()
{
	readPointsFromFile();
	pka = (int*)malloc(N * sizeof(int));

	//for (long ksize = 2; ksize <= MAX; ksize++)
	for (long ksize = 2; ksize <= 2; ksize++)
	{
		mallocSoA(&karray, ksize);
		initK(ksize);

		//************************************************************/
		// for every point: save idx where min(distance from k[idx])
		//************************************************************/

		// omp version:
		//for (long i = 0; i < N; i++)
		for (long i = 0; i < 5; i++)
		{
			float minSquareDist = INFINITY;
			float curSquareDist = 0;
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
		printf("karray:\n");
		printf("%d, %6.3f, %6.3f\n", 0, karray->x[0], karray->y[0]);
		printf("%d, %6.3f, %6.3f\n", 1, karray->x[1], karray->y[1]);
		for (int i = 0; i < 5; i++)
		{
			printf("%d: %6.3f, %6.3f Closest to K-idx: %d\n", i, xya->x[i], xya->y[i], pka[i]);
		}



		//TODO: if breakcondition: break
		freeSoA(karray);
	}



	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };



	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
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

	fscanf(fp, "%ld, %d, %d, %f", &N, &MAX, &LIMIT, &QM);
	mallocSoA(&xya, N);
	
	// populate data points:
	for (long i = 0; i < N; i++)
	{
		fscanf(fp, "%d, %f, %f", &i, &(xya->x[i]), &(xya->y)[i]);
	}
	fclose(fp);
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


//TODO quick test
//for (int i = 0; i < ksize; i++)
//{
//	printf("%d, %f, %f\n", i, karray->x[i], karray->y[i]);
//}