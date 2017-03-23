#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "Kmeans.h"

//TODO: consider adding a delimiter (testfile: ',' -- presentation: ' ')

// SoA: reduce load/store operations
typedef struct _xyArrays {
	float *x;
	float *y;
} xyArrays;


int main()
{

	long  N;			// number of points. e.g. 300000
	int   MAX;			// maximum # of clusters to find. e.g. 300
	int	  LIMIT;		// maximum # of iterations for K-means algorithm. e.g. 2000
	float QM;			// quality measure to stop. e.g. 17

	FILE *fp;
	fp = fopen("large_testinput_guy.txt", "r");
	if (fp == NULL) {
		fprintf(stderr, "Can't open input file\n");
		exit(1);
	}

	fscanf(fp, "%ld, %d, %d, %f", &N, &MAX, &LIMIT, &QM);
	
	xyArrays *xya = (xyArrays*)malloc(sizeof(xyArrays));
	xya->x = (float*)malloc(N * sizeof(float));
	xya->y = (float*)malloc(N * sizeof(float));

	// populate data points:
	for (long i = 0; i < N; i++)
	{
		fscanf(fp, "%d, %f, %f", &i, &((xya->x)[i]), &((xya->y)[i]));
	}
	fclose(fp);

	for (int i = 0; i < 5; i++)
	{
		printf("%d, %f, %f\n", i, xya->x[i], xya->y[i]);
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

	return 0;
}
