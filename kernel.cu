
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
            __syncthreads();

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

    /*
    // Choose which GPU to run on, change this on a multi-GPU system. (!For future!)
    cudaStatus = cudaSetDevice(0);
    checkErrorCuda(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    
    // Allocate GPU buffers for inputs and outputs.
    cudaStatus = cudaMalloc((void**)&dev_H, lenY * sizeof(double));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_H");
    cudaStatus = cudaMalloc((void**)&dev_DelH, lenY * lenY * sizeof(double));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_DelH");
    cudaStatus = cudaMalloc((void**)&dev_DelH_sign, lenY * sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_DelH_sign");
    cudaStatus = cudaMalloc((void**)&dev_W, lenY * lenY * sizeof(double));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_W");
    //cudaMalloc((void**)&dev_B, lenY * sizeof(double));
    cudaStatus = cudaMalloc((void**)&dev_Y, lenY * sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_Y");
    cudaStatus = cudaMalloc((void**)&dev_Selected_index, block_size * sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_Selected_index");
    cudaStatus = cudaMalloc((void**)&dev_E, (exchange_attempts + 1) * sizeof(double));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_E");
    cudaStatus = cudaMalloc((void**)&dev_bestSpinModel, lenY * sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_bestSpinModel");
    //cudaMalloc((void**)&dev_Flag, lenY * sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_bestenergy, sizeof(double));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_bestenergy");
    cudaStatus = cudaMalloc((void**)&dev_lenY, sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMalloc failed! dev_lenY");
    */


    /*
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
    cudaStatus = cudaMemset(dev_lenY, lenY, sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMemset failed! dev_lenY");
    cudaStatus = cudaMemset(dev_Selected_index, -1, block_size * sizeof(int));
    checkErrorCuda(cudaStatus, "cudaMemset failed! dev_Selected_index");
    
    
    cudaStatus = cudaMemcpy(dev_DelH_sign, DelH_sign, lenY * sizeof(int), cudaMemcpyHostToDevice);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_DelH_sign");
    */

    // Launch a kernel on the GPU with one thread for each element.
    //metropolisKernel <<<1, block_size >>> (dev_H, dev_DelH, dev_DelH_sign, dev_Y, dev_Selected_index, lenY, dev_E, dev_bestSpinModel, dev_bestenergy, exchange_attempts, T);

    
    
    /*
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "prepareMetropolisKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }*/

    /*
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching metropolisKernel!\n", cudaStatus);
        cout << cudaGetErrorString(cudaStatus) << endl;;
        goto Error;
    }


    
    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(H, dev_H, lenY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(DelHGpu, dev_DelH, lenY * lenY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Y, dev_Y, lenY * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(bestSpinModel, dev_bestSpinModel, lenY * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&bestEnergy, dev_bestenergy, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(E + step, dev_E, exchange_attempts * sizeof(double), cudaMemcpyDeviceToHost);
    

    cudaStatus = cudaMemcpy(DelH_sign, dev_DelH_sign, lenY * sizeof(int), cudaMemcpyDeviceToHost);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! dev_DelH_sign -> DelH_sign");
    */

    
    /*cudaFree(dev_H);
    cudaFree(dev_DelH);
    cudaFree(dev_DelH_sign);
    cudaFree(dev_W);
    cudaFree(dev_B);
    cudaFree(dev_Y);
    cudaFree(dev_E);
    cudaFree(dev_Selected_index);
    cudaFree(dev_bestSpinModel);
    cudaFree(dev_bestenergy);
    

    return cudaStatus;
    */

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

    // Vectorization of parammeters for Gpu
    if (ExecuteMode == QUBOGPU) {
        DelHGpu = convertDelHtoGpuDelH(DelH, num_replicas, lenY);
        WGpu = convert2Dto1D(W, lenY, lenY);
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