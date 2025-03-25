#include<stdio.h>
#include<curand_kernel.h>
#define MAX_BATCH_SIZE 16384
double calculate_percentage_delta(double calculated_pi) {
    return fabs((calculated_pi - M_PI) / M_PI) * 100.0;
}
__device__ int indexFinder(){
    long int globalId;
    int gdY=gridDim.y;
    int gdZ=gridDim.z;

    int bdX=blockDim.x;
    int bdY=blockDim.y;
    int bdZ=blockDim.z;

    int bx=threadIdx.x;
    int by=threadIdx.y;
    int bz=threadIdx.z;

    int gx=blockIdx.x;
    int gy=blockIdx.y;
    int gz=blockIdx.z;
    globalId=gx*gdY*gdZ*bdX*bdY*bdZ + gy*gdZ*bdX*bdY*bdZ + gz*bdX*bdY*bdZ + bx*bdY*bdZ + by*bdZ + bz;
    return globalId;
}

__device__ int simulate(long int seed, double needleLength){
    curandState state;
    curand_init(seed+clock64(), seed+clock64(), 0, &state);
    double xCordNeedleCenter=curand_uniform(&state)*needleLength;
    double needleAngle=curand_uniform(&state)*M_PI / 2.0f;
    double xRightTip=xCordNeedleCenter + (needleLength / 2) * sin(needleAngle) ;
    double xLeftTip=xCordNeedleCenter - (needleLength / 2) * sin(needleAngle);
    return xRightTip>(2*needleLength) || xLeftTip<0.0f;
}
__global__ void parallelNeedleSimulation(double needleLength,int *gpuResultArray, long int experiments, long int batchOffset){
    
    int threadId=indexFinder();
    bool isActive=threadId<experiments;

    if(isActive){
        if(simulate(threadId+batchOffset, needleLength)){
            gpuResultArray[threadId]=1;
        }else{
            gpuResultArray[threadId]=0;
        }
    }
}

__global__ void serialSum(int *arrayAddress, long int *localSum, long int limit){
    for(long int i=0;i<limit;i++){
        *localSum+=arrayAddress[i];
    }
}
void compute6DLaunchParameters(long int Y, dim3 *gridDim, dim3 *blockDim, int blockDimx, int blockDimy, int blockDimz) {
    blockDim->x = blockDimx;  
    blockDim->y = blockDimy;
    blockDim->z = blockDimz;
    
    int threadsPerBlock = blockDim->x * blockDim->y * blockDim->z;
    int totalBlocks = ceil((float)Y / threadsPerBlock);
    gridDim->y = gridDim->z = gridDim->x = ceil(cbrt(totalBlocks));
}
long int parallelSimulationRunner(long int simulationCount, int bX, int bY, int bZ){
    long int currentBatchSize, needleCrossCount=0,batchOffset=0,localSimulationSum;
    dim3 DimGrid, DimBlock;
    int *gpuResultArray;
    long int * localDSum;
    cudaMalloc(&gpuResultArray, MAX_BATCH_SIZE*sizeof(int));
    cudaMalloc(&localDSum, sizeof(long int));
    double needleLength=1.0f;
    while(simulationCount){
        if(simulationCount>=MAX_BATCH_SIZE){
            currentBatchSize=MAX_BATCH_SIZE;
        }
        else{
            currentBatchSize=simulationCount;
        }
        cudaMemset(gpuResultArray, 0, currentBatchSize*sizeof(int));
        cudaMemset(localDSum, 0, sizeof(long int));
        compute6DLaunchParameters(currentBatchSize, &DimGrid, &DimBlock, bX, bY, bZ);
        parallelNeedleSimulation<<<DimGrid,DimBlock>>>(needleLength, gpuResultArray, currentBatchSize, batchOffset);
        cudaDeviceSynchronize();
        serialSum<<<1,1>>>(gpuResultArray, localDSum,currentBatchSize);
        cudaDeviceSynchronize();
        cudaMemcpy(&localSimulationSum, localDSum, sizeof(long int), cudaMemcpyDeviceToHost);
        needleCrossCount+=localSimulationSum;
        batchOffset+=currentBatchSize;
        simulationCount-=currentBatchSize;
    }
    return needleCrossCount;
}

void compareBlockSizes(long int experiments){
    	cudaEvent_t start, stop;
	long long int time_start, time_end,parallel_time, crosses;
    float parallelMS;
	int blockSizes[][3] = {
        	{1024, 1, 1}, {512, 2, 1}, {256, 4, 1}, 
        	{128, 8, 1}, {64, 16, 1}, {32, 32, 1},
        	{16, 64, 1}, {8, 8, 16}, {16, 4, 16},
        	{4, 4, 64}
    	};
    	int numConfigs = sizeof(blockSizes) / sizeof(blockSizes[0]);
	for(int i=0;i<numConfigs;i++){
        printf("Testing %d %d %d\n", blockSizes[i][0], blockSizes[i][1], blockSizes[i][2]);
		time_start=clock();
    		cudaEventCreate(&start);
    		cudaEventCreate(&stop);
    		cudaEventRecord(start);
    		crosses=parallelSimulationRunner(experiments, blockSizes[i][0], blockSizes[i][1], blockSizes[i][2]);
    		time_end=clock();
    		cudaEventRecord(stop);
    		cudaEventSynchronize(stop);
    		cudaEventElapsedTime(&parallelMS, start, stop);
    		cudaEventDestroy(start);
    		cudaEventDestroy(stop);
    		parallel_time=time_end-time_start;
    		printf("%lf %lf %lf %lf\n", parallel_time/(double)CLOCKS_PER_SEC, parallelMS/1000, experiments/(double)crosses, calculate_percentage_delta(experiments/(double)crosses));
	}
}
int main(int argc, char **argv){
    if(argc<2 || argc>3){
        printf("Incorrect parameters %d\n",argc);
        return;
    }
    long int experiments=atol(argv[1]);
    compareBlockSizes(atol(argv[1]));
}

