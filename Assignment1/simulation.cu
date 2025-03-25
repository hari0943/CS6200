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
__global__ void serialNeedleSimulation(double needleLength,long int *gpuNeedleCrossCount, long int experiments, long int offset){
    for(long int i=0;i<experiments;i++){
        if(simulate(i+offset, needleLength)){
            *gpuNeedleCrossCount+=1;
        }
    }
}
__global__ void serialSum(int *arrayAddress, long int *localSum, long int limit){
    for(long int i=0;i<limit;i++){
        *localSum+=arrayAddress[i];
    }
}
void compute6DLaunchParameters(long int Y, dim3 *gridDim, dim3 *blockDim) {
    blockDim->x = 8;  
    blockDim->y = 8;
    blockDim->z = 16;
    
    int threadsPerBlock = blockDim->x * blockDim->y * blockDim->z;
    int totalBlocks = ceil((float)Y / threadsPerBlock);
    gridDim->y = gridDim->z = gridDim->x = ceil(cbrt(totalBlocks));
}
long int parallelSimulationRunner(long int simulationCount){
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
        compute6DLaunchParameters(currentBatchSize, &DimGrid, &DimBlock);
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
long int serialSimulationRunner(long int simulationCount){
    long int currentBatchSize, needleCrossCount=0,batchOffset=0, localSimulationSum;
    long int *gpuResult;
    cudaMalloc(&gpuResult, sizeof(long int));
    double needleLength=1.0f;
    while(simulationCount){
        if(simulationCount>=MAX_BATCH_SIZE){
            currentBatchSize=MAX_BATCH_SIZE;
        }
        else{
            currentBatchSize=simulationCount;
        }
        cudaMemset(gpuResult, 0, sizeof(long int));
        serialNeedleSimulation<<<1,1>>>(needleLength, gpuResult, currentBatchSize, batchOffset);
        cudaDeviceSynchronize();
        cudaMemcpy(&localSimulationSum, gpuResult, sizeof(long int), cudaMemcpyDeviceToHost);
        needleCrossCount+=localSimulationSum;
        batchOffset+=currentBatchSize;
        simulationCount-=currentBatchSize;
    }
    return needleCrossCount;
}
int main(int argc, char **argv){
    long int experiments=atol(argv[1]), time_start, time_end, parallel_time, serial_time, crosses;
    float serialMS=0,parallelMS=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    time_start=clock();
    crosses=serialSimulationRunner(experiments);
    time_end=clock();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&serialMS, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    serial_time=time_end-time_start;
    printf("%lf %lf %lf %lf\n", serial_time/(double)CLOCKS_PER_SEC, serialMS/1000, experiments/(double)crosses, calculate_percentage_delta(experiments/(double)crosses));
    time_start=clock();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    crosses=parallelSimulationRunner(experiments);
    time_end=clock();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&parallelMS, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    parallel_time=time_end-time_start;
    printf("%lf %lf %lf %lf\n", parallel_time/(double)CLOCKS_PER_SEC, parallelMS/1000, experiments/(double)crosses, calculate_percentage_delta(experiments/(double)crosses));
}
