# KMeansProject

TODO:
cudaStatus = kDiametersWithCuda(); 
	fprintf(stderr, "kCentersWithCuda failed!\n"); FF;
--> fprintf(stderr, "kDiametersWithCuda failed!\n"); FF;

slave kernel launch config: (add sharedMembytes)
kDiamBlockWithCuda << <1, THREADS_PER_BLOCK, SharedMemBytes  >> >
