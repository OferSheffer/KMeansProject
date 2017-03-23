#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "Kmeans.h"

#define N		300000		// number of points
#define MAX		300			// maximum # of clusters to find
#define LIMIT	2000		// maximum # of iterations for K-means algorithm
#define QM		17			// quality measure to stop


int main()
{
	FILE *fp;
	fp = fopen("", "r");
	if (fp == NULL) {
		fprintf(stderr, "Can't open input file");
		exit(1);
	}

	// SoA: reduce load/store operations
	struct xyArrays {
		float x[N];
		float y[N];
	};

	struct xyArrays xya;

	// populate data points:
	for (long i = 0; i < N; i++)
	{
		fscanf(fp, "%d %s %d", username, &score)
	}

	fclose(fp);


	//const int arraySize = 5;
	//const int a[arraySize] = { 1, 2, 3, 4, 5 };
	//const int b[arraySize] = { 10, 20, 30, 40, 50 };
	//int c[arraySize] = { 0 };



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
