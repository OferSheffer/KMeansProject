# KMeansProject

TODO:
cudaStatus = kDiametersWithCuda(); 
	fprintf(stderr, "kCentersWithCuda failed!\n"); FF;
--> fprintf(stderr, "kDiametersWithCuda failed!\n"); FF;

slave kernel launch config: (add sharedMembytes)
kDiamBlockWithCuda << <1, THREADS_PER_BLOCK, SharedMemBytes  >> >

id: 1, kDiamBlockWithCuda<<1, 0, 0>> launch failed for data: 100000, 0, 2
fix so that slaves receive no.threads+size of sharedmem
