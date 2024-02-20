
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>

#include "helpFunction.h"
#include "constant.h"
#include "ising.h"
#include "in_out_functions.h"


//Gpu kernel for metropolis function => Almost same as sequential version of metropolis just perform tasks in parallel.
__global__ void metropolisKernel(double* dev_H, double* dev_DelH, int* dev_DelH_sign, int* dev_Y, int* dev_Selected_index, int dev_lenY, double* dev_E, int* dev_bestSpinModel, double* best_energy, int exchange_attempts, double T)
{
    int tid = threadIdx.x;
    
    //extern __shared__ int sdata[];
    curandState state;

    if (tid < dev_lenY) {
        for (int step = 1; step < exchange_attempts; step++) {
            
            //compute Delata energy
            double deltaE = -1 * (1 - 2* dev_Y[tid]) * dev_H[tid];
            //printf("Thread %d: Local variable value = %f\n", threadIdx.x, deltaE);
            

            // Make decision that a bit flip can be accepted
            curand_init(clock64(), tid, clock64(), &state);
            
            if ((deltaE < 0) || (curand_uniform_double(&state) < exp(-deltaE / T))) {
                dev_Selected_index[tid] = tid;
            }
            else {
                dev_Selected_index[tid] = -1;
            }            
            //__syncthreads();

            // select which bit accepted            
            for (int s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if (tid<s)
                {
                    if (dev_Selected_index[tid] != -1 && dev_Selected_index[tid + s] != -1)
                    {
                        // Generate a random integer (0 or 1)

                        if ((curand(&state) % 2) == 1)
                            dev_Selected_index[tid] = dev_Selected_index[tid + s];
                    }
                    else if (dev_Selected_index[tid] == -1 && dev_Selected_index[tid + s] != -1) {
                        dev_Selected_index[tid] = dev_Selected_index[tid + s];
                    }
                }
                __syncthreads();
            }
            
            __syncthreads();


            // based on the flipped bit j update parameters
            int j = dev_Selected_index[0];
            if (tid == j) {
                dev_Y[tid] = 1 - dev_Y[tid];
                dev_E[step] = dev_E[step - 1] + deltaE;
                if (dev_E[step] < *best_energy) {
                    *best_energy = dev_E[step];
                    dev_bestSpinModel[tid] = dev_Y[tid];
                }
                //printf("Thread %d: flipped bit = %d, En: %f -> %f (delE: %f) bestE: %f\n", tid, j, dev_E[step-1], dev_E[step], deltaE, *best_energy);
                //printf("\t\t step %d [%d, %d, %d, %d, %d]\n", step, dev_Y[0], dev_Y[1], dev_Y[2], dev_Y[3], dev_Y[4]);
            }
            __syncthreads();

            if (j != -1) {
                //Update H
                dev_H[tid] = dev_H[tid] + dev_DelH[tid + j * dev_lenY];                
            }
            else {
                // Log the Energy when there is not any bit to flip
                dev_E[step] = dev_E[step - 1];
            }
            
            // Update delta_H
            if (tid == j) {
                for (int i = 0; i < dev_lenY; i++) {
                    dev_DelH[i + j * dev_lenY] *= -1;                    
                }
            }            
        }
    }
}

//dev_H, dev_DelH, dev_DelH_sign, dev_Y, dev_Selected_index, lenY, dev_E, dev_bestSpinModel, dev_bestenergy, numberOfIteration
__global__ void full_mode_metropolisKernel(double* dev_H, double* dev_DelH, int* dev_DelH_sign, int* dev_Y, int* dev_Selected_index, const int select_index_size, double* dev_E, int* dev_bestSpinModel, double* dev_best_energy, int numberOfIteration, int exchange_attempts, double* dev_Temprature)
{   
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int tid = blockId * blockDim.x + threadId;
    int temprature_index = blockId; 
    int temp_index_direction = (blockId+1) % 2;
    bool stop_flag = false;

    int index_base = select_index_size* blockId;
    
    double bE = dev_best_energy[blockId];
    //for (int i = 0; i < 101; i++) {
    //    printf("dev_DelH[%d] = %lf\n", i, dev_DelH[i]);
    //}

    if (temp_index_direction==0) {
        temp_index_direction = -1;
        //printf("\tblockId %d temp_index_direction %d \n", blockId, temp_index_direction);
    }
    
    //if (threadId == 0) {
    //    printf("blockId %d, tempIndex %d, tempDirection %d \n", blockId, temprature_index, temp_index_direction);
    //}
    
    if (blockId == gridDim.x - 1 && blockId % 2 == 0) {
        stop_flag = true;
        temp_index_direction = -1;
        printf("blockId %d the stop_flag is on \n", blockId);
    }

    //extern __shared__ int sdata[];
    curandState state;

    //printf("\tblockId: %d \t threadId: %d \t index_base + threadId: %d\n", blockId, threadId, index_base + threadId);
     
    
    //if (tid < gridDim.x * blockDim.x) {

        for (int step = 1; step < numberOfIteration; step++) {
            
            //compute Delata energy
            double deltaE = -1 * (1 - 2 * dev_Y[tid]) * dev_H[tid];
            //printf("Thread %d: Local variable value = %f\n", threadIdx.x, deltaE);

            

            // Make decision that a bit flip can be accepted
            curand_init(clock64(), tid, clock64(), &state);

            if ((deltaE < 0) || (curand_uniform_double(&state) < exp(-deltaE / dev_Temprature[temprature_index]))) {
                dev_Selected_index[index_base+threadId] = threadId;
            }
            else {
                dev_Selected_index[index_base + threadId] = -1;
            }
            //__syncthreads();

            // select which bit accepted            
            for (int s = select_index_size / 2; s > 0; s >>= 1)
            {
                if (threadId < s)
                {
                    //printf("\tblockId: %d \t threadId: %d \t index_base + threadId: %d \t index_base + threadId+s: %d\n", blockId, threadId, index_base + threadId, index_base + threadId + s);
                    if (dev_Selected_index[index_base + threadId] != -1 && dev_Selected_index[index_base + threadId + s] != -1)
                    {
                        // Generate a random integer (0 or 1)

                        if ((curand(&state) & 1) == 1) //find the least significant bit
                            dev_Selected_index[index_base + threadId] = dev_Selected_index[index_base + threadId + s];
                    }
                    else if (dev_Selected_index[index_base + threadId] == -1 && dev_Selected_index[index_base + threadId + s] != -1) {
                        dev_Selected_index[index_base + threadId] = dev_Selected_index[index_base + threadId + s];
                    }
                }
                __syncthreads();
            }

            __syncthreads();

            //if (step == 1 && blockId <= 3)
            //    printf("\tblockId: %d \t threadId: %d \t dev_Selected_index %d \n", blockId, threadId, dev_Selected_index[index_base + threadId]);

            // based on the flipped bit j update parameters
            int j = dev_Selected_index[index_base];

            //if (threadId == j && step <= 2) {
                //printf("step: %d, tid: %d, blockId: %d, threadId: %d == %d :j, E: %f, deltaE: %f, NewE: %f \n", step, tid, blockId, threadId, j, dev_E[blockId * numberOfIteration + step-1], deltaE, dev_E[blockId * numberOfIteration + step - 1] + deltaE);
            //}

            if (threadId == j) {

                dev_Y[tid] = 1 - dev_Y[tid];
                dev_E[blockId * numberOfIteration + step] = dev_E[blockId * numberOfIteration + step - 1] + deltaE;
                if (dev_E[blockId * numberOfIteration + step] < dev_best_energy[blockId]) {
                    dev_best_energy[blockId] = dev_E[blockId * numberOfIteration + step];
                    dev_bestSpinModel[tid] = dev_Y[tid];
                }
                int Y_index=blockId* blockDim.x;
                if (blockId == 3 && DEBUGFLAG) {
                    printf("\tstep %d :: blockId: %d threadId %d: flipped bit = %d, En: %f -> %f (delE: %f) bestE: %f\n", step, blockId, threadId, j, dev_E[blockId * numberOfIteration + step - 1], dev_E[blockId * numberOfIteration + step], deltaE, dev_best_energy[blockId]);
                    printf("\t\t step %d :: blockId %d [%d, %d, %d, %d, %d]\n", step, blockId, dev_Y[Y_index + 0], dev_Y[Y_index + 1], dev_Y[Y_index + 2], dev_Y[Y_index + 3], dev_Y[Y_index + 4]);
                }
            }
            __syncthreads();

            if (j != -1) {
                //Update H                  ( dev_DelH : replica * lenY * lenY)
                if (step <=-1 && DEBUGFLAG)
                    printf("step: %d, [index: %d] , tid: %d, blockId: %d, threadId: %d, j: %d, dev_H: %f, dev_DelH: %f, new dev_H: %f \n", 
                    step,
                    blockId * blockDim.x * blockDim.x + threadId * blockDim.x + j,
                    tid,
                    blockId,
                    threadId,
                    j,
                    dev_H[tid],
                    dev_DelH[blockId * blockDim.x * blockDim.x + threadId * blockDim.x + j],
                    dev_H[tid]);

                double oldValue = dev_H[tid];

                dev_H[tid] = dev_H[tid] + dev_DelH[blockId * blockDim.x * blockDim.x + threadId * blockDim.x + j];
                //dev_H[tid] = dev_H[tid] + dev_DelH[tid + j * dev_lenY];
                //VectorDelH[r * num_replicas * lenY + i * lenY + i] = DelH[r][i][j];
                if (blockId == 3 && DEBUGFLAG) {
                    printf("Update dev_H: %lf to %lf \n", oldValue, dev_H[tid]);
                }

            }
            else {
                // Log the Energy when there is not any bit to flip
                dev_E[blockId * numberOfIteration + step] = dev_E[blockId * numberOfIteration + step - 1];
            }

            // Update delta_H
            if (threadId == j) {
                for (int i = 0; i < blockDim.x; i++) {
                    //double oldValue= dev_DelH[blockId * blockDim.x * blockDim.x + i * blockDim.x + j];
                    dev_DelH[blockId * blockDim.x * blockDim.x + i * blockDim.x + j] *= -1;
                    
                }
            }
            
            __syncthreads();

            if (step % exchange_attempts == 0) {
                if (stop_flag == false) {
                    
                    temprature_index += temp_index_direction;

                    if (temprature_index == 0 || temprature_index == gridDim.x-1) {
                        stop_flag = true;
                        temp_index_direction = temp_index_direction * -1;
                    }
                }
                else {
                    stop_flag = false;
                }                   
            }
            if (threadId == 0 && bE != dev_best_energy[blockId]) {
                printf("S %d bE in block %d is %lf \n", step, blockId, dev_best_energy[blockId]);
                bE = dev_best_energy[blockId];
            }
        }
    //}
}


// Show Error message
void checkErrorCuda(cudaError_t cudaStatus, string message) {
    if (cudaStatus != cudaSuccess) {
        cout << message << " : " << cudaGetErrorString(cudaStatus) << endl;
        //fprintf(stderr, "%s : %s \n",  message, cudaGetErrorString(cudaStatus));
    }
}

cudaError_t allocateMemory(int lenY, int block_size, int exchange_attempts, double** dev_H, double** dev_DelH, double** dev_W, double** dev_B, double** dev_E, int** dev_bestSpinModel, int** dev_Y, int** dev_Selected_index, int** dev_lenY, int** dev_DelH_sign, double** dev_bestenergy) {

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system. (!For future!)
    cudaStatus = cudaSetDevice(0);
    checkErrorCuda(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

    // Allocate GPU buffers for inputs and outputs.
    cudaStatus = cudaMalloc((void**)dev_H, lenY * sizeof(double));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_H");
    cudaStatus = cudaMalloc((void**)dev_DelH, lenY * lenY * sizeof(double));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_DelH");
    cudaStatus = cudaMalloc((void**)dev_DelH_sign, lenY * sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_DelH_sign");
    cudaStatus = cudaMalloc((void**)dev_W, lenY * lenY * sizeof(double));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_W");


    cudaMalloc((void**)&dev_B, lenY * sizeof(double));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_B");
    cudaStatus = cudaMalloc((void**)dev_Y, lenY * sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_Y");
    cudaStatus = cudaMalloc((void**)dev_Selected_index, block_size * sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_Selected_index");
    cudaStatus = cudaMalloc((void**)dev_E, (exchange_attempts + 1) * sizeof(double));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_E");
    cudaStatus = cudaMalloc((void**)dev_bestSpinModel, lenY * sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_bestSpinModel");
    //cudaMalloc((void**)&dev_Flag, lenY * sizeof(int));
    cudaStatus = cudaMalloc((void**)dev_bestenergy, sizeof(double));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_bestenergy");
    cudaStatus = cudaMalloc((void**)dev_lenY, sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_lenY");

    return cudaStatus;

}

cudaError_t allocateMemory_with_size(double** dev_H, double** dev_DelH, double** dev_W, double** dev_B, double** dev_E, int** dev_bestSpinModel, int** dev_Y, int** dev_Selected_index, int** dev_lenY, int** dev_DelH_sign, double** dev_bestenergy, double** dev_Temprature, int* sizeArray) {

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system. (!For future!)
    cudaStatus = cudaSetDevice(0);
    checkErrorCuda(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

    // Allocate GPU buffers for inputs and outputs.
    cudaStatus = cudaMalloc((void**)dev_H, sizeArray[0] * sizeof(double));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_H");
    cudaStatus = cudaMalloc((void**)dev_DelH, sizeArray[1] * sizeof(double));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_DelH");
    cudaStatus = cudaMalloc((void**)dev_DelH_sign, sizeArray[2] * sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_DelH_sign");
    cudaStatus = cudaMalloc((void**)dev_W, sizeArray[3] * sizeof(double));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_W");


    cudaMalloc((void**)&dev_B, sizeArray[4] * sizeof(double));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_B");


    cudaStatus = cudaMalloc((void**)dev_Y, sizeArray[5] * sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_Y");
    cudaStatus = cudaMalloc((void**)dev_Selected_index, sizeArray[6] * sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_Selected_index");
    cudaStatus = cudaMalloc((void**)dev_E, sizeArray[7] * sizeof(double));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_E");
    cudaStatus = cudaMalloc((void**)dev_bestSpinModel, sizeArray[9] * sizeArray[8] * sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_bestSpinModel");
    //cudaMalloc((void**)&dev_Flag, lenY * sizeof(int));
    cudaStatus = cudaMalloc((void**)dev_bestenergy, sizeArray[9] * sizeof(double));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_bestenergy");

    cudaStatus = cudaMalloc((void**)dev_Temprature, sizeArray[10] * sizeof(double));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_Temprature");

    cudaStatus = cudaMalloc((void**)dev_lenY, sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_lenY");

    return cudaStatus;

}

cudaError_t copyMemoryFromHostToDevice_with_size(double* H, double* dev_H, double* DelH, double* dev_DelH, int* DelH_sign, int* dev_DelH_sign, double* WGpu, double* dev_W, int* Y, int* dev_Y, double* E, double* dev_E, double bestEnergy, double* dev_bestenergy, int* bestSpinModel, int* dev_bestSpinModel, int* dev_Selected_index, double* dev_Temprature, double* Temprature, int* sizeArray) {

    cudaError_t cudaStatus;
    // Copy input vectors from host memory to GPU buffers.

    cudaStatus = cudaMemcpy(dev_H, H, sizeArray[0] * sizeof(double), cudaMemcpyHostToDevice);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_H");
    cudaStatus = cudaMemcpy(dev_DelH, DelH, sizeArray[1] * sizeof(double), cudaMemcpyHostToDevice);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_DelH");


    cudaStatus = cudaMemcpy(dev_W, WGpu, sizeArray[3] * sizeof(double), cudaMemcpyHostToDevice);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_W");

  
    cudaStatus = cudaMemcpy(dev_Y, Y, sizeArray[5] * sizeof(int), cudaMemcpyHostToDevice);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_Y");
    cudaStatus = cudaMemset(dev_Selected_index, -1, sizeArray[6] * sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMemset failed! dev_Selected_index");
    cudaStatus = cudaMemcpy(dev_E, E, sizeArray[7] * sizeof(double), cudaMemcpyHostToDevice);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_E");

    double* extended_bestSpinModel = new double[sizeArray[9] * sizeArray[8]];
    for(int i=0;i< sizeArray[9];i++)
        memcpy(extended_bestSpinModel + i * sizeArray[8], bestSpinModel, sizeof(int) * sizeArray[8]);
    cudaStatus = cudaMemcpy(dev_bestSpinModel, extended_bestSpinModel, sizeArray[9] * sizeArray[8] * sizeof(int), cudaMemcpyHostToDevice);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_bestSpinModel");

    double* bestEnergyArray = new double[sizeArray[9]];
    fill1Darray(bestEnergyArray, bestEnergy, sizeArray[9]);
    cudaStatus = cudaMemcpy(dev_bestenergy, bestEnergyArray, sizeArray[9] * sizeof(double), cudaMemcpyHostToDevice);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_bestenergy");

    cudaStatus = cudaMemcpy(dev_Temprature, Temprature, sizeArray[10] * sizeof(double), cudaMemcpyHostToDevice);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_Temprature");

    //cudaMemset(dev_bestenergy, bestEnergy, sizeof(double));
    



    //cudaStatus = cudaMemcpy(dev_DelH_sign, DelH_sign, lenY * sizeof(int), cudaMemcpyHostToDevice);
    //checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_DelH_sign");
    return cudaStatus;


}


cudaError_t copyMemoryFromHostToDevice(double* H, double* dev_H, double* DelHGpu, double* dev_DelH, int* DelH_sign, int* dev_DelH_sign, double* WGpu, double* dev_W, double* B, double* dev_B, int* Y, int* dev_Y, int lenY, double* E, double* dev_E, int step, double bestEnergy, double* dev_bestenergy, int* bestSpinModel, int* dev_bestSpinModel, int replica, int block_size, int* dev_Selected_index) {
                    
    cudaError_t cudaStatus;
    // Copy input vectors from host memory to GPU buffers.

    cudaStatus = cudaMemcpy(dev_H, H, lenY * sizeof(double), cudaMemcpyHostToDevice);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_H");
    cudaStatus = cudaMemcpy(dev_DelH, DelHGpu, lenY * lenY * sizeof(double), cudaMemcpyHostToDevice);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_DelH");
    cudaStatus = cudaMemcpy(dev_W, WGpu, lenY * lenY * sizeof(double), cudaMemcpyHostToDevice);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_W");
    //cudaMemcpy(dev_B, B, lenY * sizeof(double), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_Y, Y, lenY * sizeof(int), cudaMemcpyHostToDevice);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_Y");
    cudaStatus = cudaMemcpy(dev_bestSpinModel, bestSpinModel, lenY * sizeof(int), cudaMemcpyHostToDevice);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_bestSpinModel");
    cudaStatus = cudaMemcpy(dev_bestenergy, &bestEnergy, sizeof(double), cudaMemcpyHostToDevice);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_bestenergy");
    //cudaMemset(dev_bestenergy, bestEnergy, sizeof(double));
    cudaStatus = cudaMemcpy(dev_E, E + step - 1, sizeof(double), cudaMemcpyHostToDevice);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_E");
    //cudaStatus = cudaMemset(dev_lenY, lenY, sizeof(int));
    //checkErrorCuda(cudaStatus, "cudaMemset failed! dev_lenY");
    cudaStatus = cudaMemset(dev_Selected_index, -1, block_size * sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMemset failed! dev_Selected_index");


    cudaStatus = cudaMemcpy(dev_DelH_sign, DelH_sign, lenY * sizeof(int), cudaMemcpyHostToDevice);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_DelH_sign");
    return cudaStatus;


}

void copyMemoryFromDeviceToHost_with_size(int* vector_Y, int* dev_Y, int* bestSpinModel, int* dev_bestSpinModel, int& bestEnergy, double* dev_bestenergy, double* vector_E, double* dev_E, int* memory_size)
{
    cudaError_t cudaStatus;

    //return cudaStatus;
}


cudaError_t copyMemoryFromDeviceToHost(int lenY, double* H, double* dev_H, double* DelHGpu, double* dev_DelH, int* Y, int* dev_Y, int* bestSpinModel, int* dev_bestSpinModel, double& bestEnergy, double* dev_bestenergy, double* E, double* dev_E, int* DelH_sign, int* dev_DelH_sign, int step, int exchange_attempts) {
    cudaError_t cudaStatus;

    cudaStatus = cudaMemcpy(&bestEnergy, dev_bestenergy, sizeof(double), cudaMemcpyDeviceToHost);
    printf("Value of bestEnergy: %lf\n", bestEnergy);
    
    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(H, dev_H, lenY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(DelHGpu, dev_DelH, lenY * lenY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Y, dev_Y, lenY * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(bestSpinModel, dev_bestSpinModel, lenY * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(E + step, dev_E, exchange_attempts * sizeof(double), cudaMemcpyDeviceToHost);


    cudaStatus = cudaMemcpy(DelH_sign, dev_DelH_sign, lenY * sizeof(int), cudaMemcpyDeviceToHost);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_DelH_sign -> DelH_sign");

    

    return cudaStatus;
}

void FreeMemoryDevice(double* dev_H, double* dev_DelH, double* dev_W, double* dev_B, double* dev_E, int* dev_bestSpinModel, int* dev_Y, int* dev_Selected_index, int* dev_lenY, int* dev_DelH_sign, double* dev_bestenergy) {
    cudaFree(dev_H);
    cudaFree(dev_DelH);
    cudaFree(dev_DelH_sign);
    cudaFree(dev_W);
    cudaFree(dev_B);
    cudaFree(dev_Y);
    cudaFree(dev_E);
    cudaFree(dev_Selected_index);
    cudaFree(dev_bestSpinModel);
    cudaFree(dev_bestenergy);
}


//prepare_full_MetropolisKernel(vector_H, vector_DelH, DelH_sign, WGpu, B, vector_Y, lenY, vector_E, Temperature, exchange_attempts, bestEnergy, bestSpinModel);
// prepare memory to call Gpu Kernel
cudaError_t prepare_full_MetropolisKernel(double* vector_H, double* vector_DelH, int* vector_DelH_sign, double* WGpu, double* B, int* vector_Y, int lenY, double* vector_E, double* Temperature, int exchange_attempts, double& bestEnergy, int* bestSpinModel, int replica, int numberOfIteration) {

    double* dev_H = 0;
    double* dev_DelH = 0;
    double* dev_W = 0;
    double* dev_B = 0;
    double* dev_E = 0;

    int* dev_bestSpinModel = 0;
    int* dev_Y = 0;
    int* dev_Selected_index = 0;
    int* dev_lenY = 0;
    int* dev_DelH_sign = 0;
    double* dev_bestenergy = 0;
    double* dev_Temprature = 0;

    

    int select_index_size = nextPowerOf2(lenY);

    int memory_size[] = {
        replica*lenY,                   //  0: dev_H
        replica* lenY*lenY,             //  1: dev_DelH
        replica,                        //  2: dev_DelH_sign
        lenY*lenY,                      //  3: dev_W
        lenY,                           //  4: dev_B
        replica* lenY,                  //  5: dev_Y
        replica* select_index_size,            //  6: dev_selected_index
        replica* numberOfIteration,     //  7: dev_E
        lenY,                           //  8: dev_bestSpinModel, actaual size of dev_bestSpinModel is replica * lenY
        replica,                        //  9: dev_bestenergy
        replica,                        // 10: dev_Temprature
        1                               // 11: dev_lenY
    };

    cudaError_t cudaStatus;
    // Allocate GPU buffers for inputs and outputs.
    cudaStatus = allocateMemory_with_size(&dev_H, &dev_DelH, &dev_W, &dev_B, &dev_E, &dev_bestSpinModel, &dev_Y, &dev_Selected_index, &dev_lenY, &dev_DelH_sign, &dev_bestenergy, &dev_Temprature, memory_size);

    cudaStatus = copyMemoryFromHostToDevice_with_size(vector_H, dev_H, vector_DelH, dev_DelH, vector_DelH_sign, dev_DelH_sign, WGpu, dev_W, vector_Y, dev_Y, vector_E, dev_E, bestEnergy, dev_bestenergy, bestSpinModel, dev_bestSpinModel, dev_Selected_index, dev_Temprature, Temperature, memory_size);
    cout << "prepare_full_MetropolisKernel, after memory copy"<< endl;
    // Launch a kernel on the GPU with one thread for each element.
    printX(vector_Y + 3 * lenY, lenY, "X for replica 3:");
    printH(vector_H + 3*lenY, lenY, "H[3]: ");
    printH(vector_DelH + 3 * lenY * lenY, lenY * lenY, "DelH for replica 3: ");

    full_mode_metropolisKernel << <replica, lenY >> > (dev_H, dev_DelH, dev_DelH_sign, dev_Y, dev_Selected_index, select_index_size, dev_E, dev_bestSpinModel, dev_bestenergy, numberOfIteration, exchange_attempts, dev_Temprature);
    
    cout << "after full_mode_metropolisKernel" << endl;
    
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    checkErrorCuda(cudaStatus, "prepareMetropolisKernel launch failed : !");

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching metropolisKernel!\n", cudaStatus);
        cout << cudaGetErrorString(cudaStatus) << endl;;
        goto Error;
    }



    // Copy output vector from GPU buffer to host memory.

    //copyMemoryFromDeviceToHost_with_size(vector_Y, dev_Y, bestSpinModel, dev_bestSpinModel, bestEnergy, dev_bestenergy, vector_E, dev_E, memory_size);

    //cout << " \t\t\t Best Energy: " << bestEnergy << endl;

Error:
    FreeMemoryDevice(dev_H, dev_DelH, dev_W, dev_B, dev_E, dev_bestSpinModel, dev_Y, dev_Selected_index, dev_lenY, dev_DelH_sign, dev_bestenergy);

    return cudaStatus;
}



// prepare memory to call Gpu Kernel
cudaError_t prepareMetropolisKernel(double* H, double* DelHGpu, int* DelH_sign, double* WGpu, double* B, int* Y, int lenY, double* M, double* E, double T, int step, int exchange_attempts, double& bestEnergy, int* bestSpinModel, int replica) {
    
    double* dev_H = 0;
    double* dev_DelH = 0;
    double* dev_W = 0;
    double* dev_B = 0;
    double* dev_E = 0;

    int* dev_bestSpinModel = 0;
    int* dev_Y = 0;
    int* dev_Selected_index=0;
    int* dev_lenY = 0;
    int* dev_DelH_sign = 0;
    double* dev_bestenergy = 0;

    
    int block_size = nextPowerOf2(lenY);
    
    
        
    cudaError_t cudaStatus;
    // Allocate GPU buffers for inputs and outputs.
    cudaStatus = allocateMemory(lenY, block_size, exchange_attempts, &dev_H, &dev_DelH, &dev_W, &dev_B, &dev_E, &dev_bestSpinModel, &dev_Y, &dev_Selected_index, &dev_lenY, &dev_DelH_sign, &dev_bestenergy);

    cudaStatus = copyMemoryFromHostToDevice(H, dev_H, DelHGpu, dev_DelH, DelH_sign, dev_DelH_sign, WGpu, dev_W, B, dev_B, Y, dev_Y, lenY, E, dev_E, step, bestEnergy, dev_bestenergy, bestSpinModel, dev_bestSpinModel, replica, block_size, dev_Selected_index);

    // Launch a kernel on the GPU with one thread for each element.
    metropolisKernel << <1, block_size >> > (dev_H, dev_DelH, dev_DelH_sign, dev_Y, dev_Selected_index, lenY, dev_E, dev_bestSpinModel, dev_bestenergy, exchange_attempts, T);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    checkErrorCuda(cudaStatus, "prepareMetropolisKernel launch failed : !");

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching metropolisKernel!\n", cudaStatus);
        cout << cudaGetErrorString(cudaStatus) << endl;;
        goto Error;
    }


    
    // Copy output vector from GPU buffer to host memory.

    copyMemoryFromDeviceToHost(lenY, H, dev_H, DelHGpu, dev_DelH, Y, dev_Y, bestSpinModel, dev_bestSpinModel, bestEnergy, dev_bestenergy, E, dev_E, DelH_sign, dev_DelH_sign, step, exchange_attempts);
    
    cout << " \t\t\t Best Energy: " << bestEnergy << endl;

Error:
    FreeMemoryDevice(dev_H, dev_DelH, dev_W, dev_B, dev_E, dev_bestSpinModel, dev_Y, dev_Selected_index, dev_lenY, dev_DelH_sign, dev_bestenergy);

    return cudaStatus;
}


/* ***************************************************************** */




// One step in metropolis algorithm
double metropolis(int ExecuteMode, double** W, double* B, double* H, double** DelH, int* X, int lenX, double OldE, double T, int step, int replica) {
    double E = 0;
    int i = rand() % lenX;

    double deltaE;
    if (ExecuteMode == IsingMode)    deltaE = deltaEnergyIsing(ExecuteMode, W, B, X, lenX, i);
    else if (ExecuteMode == QUBOMode) deltaE = deltaEnergyQUBO(W, X, i, H);

    if ((deltaE < 0) || ((rand() / static_cast<double>(RAND_MAX)) < exp(-deltaE / T))) {
        if (ExecuteMode == IsingMode) {
            X[i] *= -1;
            E = energy(W, B, X, lenX);
        }
        else if (ExecuteMode == QUBOMode) {
            X[i] = 1 - X[i];
            E = Energy_based_Delta(OldE, deltaE);
            updateH(H, DelH, lenX, i);
            update_delta_H(DelH, i, lenX);
        }
        return E;
    }
    return OldE;
}

//initialize the bestEnergy, Temperature array range, Energy (E), Magnet (M), and bestSpinModel

double initEnergyAndMagnet(int num_replicas, double** W, double* B, int** Y, int lenY, double** M, double** E, int* bestSpinModel){
    double bestEnergy = 0;
    if (num_replicas != 1) {
        for (int r = 0; r < num_replicas; r++) {
            E[r][0] = energy(W, B, Y[r], lenY);

            M[r][0] = magnetization(Y[r], lenY);
            if (r == 0 || bestEnergy > E[r][0]) {
                bestEnergy = E[r][0];
                memcpy(bestSpinModel, Y[r], sizeof(int) * lenY);
            }
        }
    }
    else {
        E[0][0] = energy(W, B, Y[0], lenY);
        M[0][0] = magnetization(Y[0], lenY);
        bestEnergy = E[0][0];
        memcpy(bestSpinModel, Y[0], sizeof(int) * lenY);
    }
    return bestEnergy;
}


void intitTemperature(int num_replicas, double minTemp, double maxTemp, double* Temperature){
    if (num_replicas != 1) {
        for (int r = 0; r < num_replicas; r++) {
            Temperature[r] = minTemp + r * (maxTemp - minTemp) / (num_replicas - 1);
            cout << "Temperature: " << Temperature[r] << endl;
        }
    }
    else {
        Temperature[0] = minTemp;
        cout << "Temperature: " << Temperature[0] << endl;
    }
}


double full_GPU_Mode(int num_replicas, double** W, double* B, int** Y, int lenY, double** M, double** E, int* bestSpinModel, int numberOfIteration, int exchange_attempts, double minTemp, double maxTemp) {
    double bestEnergy;
    //double* Temperature = new double[num_replicas];
    double** H = ComputeH_forAllReplica(num_replicas, W, B, Y, lenY);
    double*** DelH = ComputeDelH_forAllReplica(num_replicas, W, Y, lenY);
    int* DelH_sign = new int[num_replicas* lenY];
    //fill2DarrayInt(DelH_sign, 1, num_replicas, lenY);
    double* vector_DelH;
    double* vector_H;
    double* vector_E;
    int* vector_Y;
    double* vector_W = convert2Dto1D(W, lenY, lenY);;

    double* Temperature = new double[num_replicas];
    intitTemperature(num_replicas, minTemp, maxTemp, Temperature);
    bestEnergy = initEnergyAndMagnet(num_replicas, W, B, Y, lenY, M, E, bestSpinModel);
    cout << bestEnergy << endl;

    vector_DelH = VectorizedDelH(DelH, num_replicas, lenY);
    vector_H = convert2Dto1D(H, num_replicas, lenY);
    //printAllH(H, num_replicas, lenY);
    //printH(vector_H, num_replicas * lenY, "vector_H");
    vector_Y = convert_int_2Dto1D(Y, num_replicas, lenY);
    vector_E = convert2Dto1D(E, num_replicas, numberOfIteration);

    // Preperation of replica exchange parameters
    //int exchangeFlag = 0;   // Flag to enable the exchange between neighbour replicas
    // Perform the Metropolis function numberOfIteration times for each replica 
    //for (int step = 1; step < numberOfIteration; step++) {
        //cout << "step: " << step << endl;
        //cout << "******************* before prepareMetropolisKernel calling/ step: " << step << endl;
        
        prepare_full_MetropolisKernel(vector_H, vector_DelH, DelH_sign, vector_W, B, vector_Y, lenY, vector_E, Temperature, exchange_attempts, bestEnergy, bestSpinModel, num_replicas, numberOfIteration);

        
      
        // Replica exchange attempts
        
        //step = step + exchange_attempts - 1;
        //replicaExchangeGpu(Temperature, num_replicas, Y, E, step, H, DelHGpu, DelH_sign, exchangeFlag);

        //cout << "after prepareMetropolisKernel calling/ step: " << step << endl;
        

    //}

    return bestEnergy;
}

/* Execute the optimization function based on replica exchange MCMC
* ExecuteMode : Ising or QUBO
* W: interconnection of spins
* B: bias
* Y: spins 2D array sizeof (num_replicas \times lenY), lenY: number of spins
* M: record of Magnet in each iteration per replica
* E: record of Energy in each iteration per replica
*/
void ising(int ExecuteMode, double** W, double* B, int** Y, int lenY, double** M, double** E, double T, int num_replicas, int numberOfIteration, int exchange_attempts, int* bestSpinModel, double minTemp, double maxTemp) {

    double bestEnergy;
    double* Temperature = new double[num_replicas];
    double** H = ComputeH_forAllReplica(num_replicas, W, B, Y, lenY);
    double*** DelH = ComputeDelH_forAllReplica(num_replicas, W, Y, lenY);
    int** DelH_sign = Declare2D_ArrayInt(num_replicas, lenY);
    fill2DarrayInt(DelH_sign, 1, num_replicas, lenY);
    double** DelHGpu;
    double* WGpu;
    
    

    
    //initialize the bestEnergy, Temperature array range, Energy (E), Magnet (M), and bestSpinModel
    /*
    if (num_replicas != 1) {
        for (int r = 0; r < num_replicas; r++) {
            Temperature[r] = minTemp + r * (maxTemp - minTemp) / (num_replicas - 1);
            cout << "Temperature: " << Temperature[r] << endl;
            E[r][0] = energy(W, B, Y[r], lenY);

            M[r][0] = magnetization(Y[r], lenY);
            if (r == 0 || bestEnergy > E[r][0]) {
                bestEnergy = E[r][0];
                memcpy(bestSpinModel, Y[r], sizeof(int) * lenY);
            }
        }
    }
    else {
        Temperature[0] = minTemp;
        cout << "Temperature: " << Temperature[0] << endl;
        E[0][0] = energy(W, B, Y[0], lenY);
        M[0][0] = magnetization(Y[0], lenY);
        bestEnergy = E[0][0];
        memcpy(bestSpinModel, Y[0], sizeof(int) * lenY);
    }

    cout << bestEnergy<< endl;
    */
    intitTemperature(num_replicas, minTemp, maxTemp, Temperature);
    bestEnergy = initEnergyAndMagnet(num_replicas, W, B, Y, lenY, M, E, bestSpinModel);

    cout << bestEnergy << endl;

    // Vectorization of parammeters for Gpu
    if (ExecuteMode == QUBOGPU) {
        DelHGpu = convertDelHtoGpuDelH(DelH, num_replicas, lenY);
        WGpu = convert2Dto1D(W, lenY, lenY);
        
    }
    else if (ExecuteMode == QUBOGPUFULL) {
        //prepare_Full_GPU_Mode();
        full_GPU_Mode(num_replicas, W, B, Y, lenY, M, E, bestSpinModel, numberOfIteration, exchange_attempts, minTemp, maxTemp);
        return;
    }    

    // Preperation of replica exchange parameters
    int exchangeFlag = 0;   // Flag to enable the exchange between neighbour replicas
    // Perform the Metropolis function numberOfIteration times for each replica 
    for (int step = 1; step < numberOfIteration; step++) {
        //cout << "step: " << step << endl;
        //cout << "******************* before prepareMetropolisKernel calling/ step: " << step << endl;
        for (int r = 0; r < num_replicas; r++) {
            T = Temperature[r];
            double previousE = E[r][step - 1];
            //for (int spin = 0; spin < lenY/25; spin++) 
            
            if (ExecuteMode == QUBOGPU) {                
                //printX(Y[r], lenY, "Y["+to_string(r)+"]");
                //printH(H[r], lenY, "H");
                //print2D_arr_double(DelH[r], lenY, lenY);
                prepareMetropolisKernel(H[r], DelHGpu[r], DelH_sign[r], WGpu, B, Y[r], lenY, M[r], E[r], T, step, exchange_attempts, bestEnergy, bestSpinModel, r);
                    
            }
            else {
                previousE = metropolis(ExecuteMode, W, B, H[r], DelH[r], Y[r], lenY, previousE, T, step, r);
                E[r][step] = previousE;
                M[r][step] = magnetization(Y[r], lenY);
                if (bestEnergy > E[r][step]) {
                    memcpy(bestSpinModel, Y[r], sizeof(int) * lenY);
                    bestEnergy = E[r][step];
                }
            }            
        }
        // Replica exchange attempts
        if (ExecuteMode != QUBOGPU && step % exchange_attempts == 0) {
            replicaExchange(Temperature, num_replicas, Y, M, E, step, H, DelH, exchangeFlag);
        }
        else if (ExecuteMode == QUBOGPU) {
            step = step + exchange_attempts-1;
            replicaExchangeGpu(Temperature, num_replicas, Y, E, step, H, DelHGpu, DelH_sign, exchangeFlag);
            
            cout << "after prepareMetropolisKernel calling/ step: " << step << endl;
        }

    }
}

/**********************************************************************************************/





int main()
{
    


    int L = 0;                      // Number of spins for each replica
    int Lsqrt = 0;                  /* This parameter is just used for graphical representation of 2-D Ising model
                                     and it is not important for other cases Lsqrt * Lsqrt = L         */
    double T = 0.0;                 // Temperature ---> when 
    int num_replicas = 1;           // Number of replica in the replica exchange MCMC
    double minTemp = 0;             // Min temperature of replica exchange MCMC
    double maxTemp = 0;             // Max temperature of replica exchange MCMC
    int numberOfIteration = 1;      // Number of Iteration in for MCMC
    int exchange_attempts = 0;      // After how many iteration, an exchange should be applied

    /*      E = -\sum_{i,j} A_{i,j} s_i s_j - \sum_i B_i s_i   */
    string Afile = "";              //      file path for A matrix in Ising (QUBO) model 
    string Bfile = "";              //      file path for Bias (B) in Ising (QUBO) model 
    string outputPath = "";          //     The path to the directory to save the output 
    int ExecuteMode = QUBOMode;     //      Execution mode: IsingMode or QUBOMode

    /*      Read the setting file and initialize the parameters    */
    readSetting(L, Lsqrt, Afile, Bfile, outputPath, ExecuteMode, num_replicas, numberOfIteration, exchange_attempts, minTemp, maxTemp);

    //cout << "L: " << L << " Lsqrt: " << Lsqrt << " Afile: " << Afile << " Bfile: " << Bfile << " ExecuteMode: " << ExecuteMode << " num_replicas: " << num_replicas << " numberOfIteration: " << numberOfIteration << " exchange_attempts: " << exchange_attempts << endl;
    //cout << "minTemp: " << minTemp << " maxTemp: " << maxTemp << endl;


    // Create the output folder
    createFolder(outputPath);

    //Initialize the spins for each replica
    int** X = createVector(ExecuteMode, L, num_replicas);
    int* bestSpinModel = new int[L];            // Best found solution

    double** M = new double* [num_replicas];     //  Magnetization for different replica (This parameter is usefull for Ising model)
    double** E = new double* [num_replicas];     //  Energy for different replica
    for (int r = 0; r < num_replicas; r++) {
        M[r] = new double[numberOfIteration];
        E[r] = new double[numberOfIteration];
    }

    //  Initialize the bias    
    double* B;
    B = initB(L, ReadDataFromFile, Bfile);

    //Initalize the interconnection of spins
    double** A;
    A = initW(L, Lsqrt, ReadDataFromFile, Afile);
    //writeMatrixToFile(outputPath + "\\WInit.csv", A, L);

    // Log all the initial spin states
    writeSpinsInFile(num_replicas, X, L, Lsqrt, outputPath, "Initlattice");

    // Optimization Function
    ising(ExecuteMode, A, B, X, L, M, E, T, num_replicas, numberOfIteration, exchange_attempts, bestSpinModel, minTemp, maxTemp);

    //  Record Logs: Magnet, Energy, final spin states, and best spin model
    recordLogs(outputPath, M, E, numberOfIteration, num_replicas, L, Lsqrt, X, bestSpinModel);

    return 0;
}



/*****************************************************************************************/