
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>

#include "helpFunction.h"
#include "constant.h"
#include "ising.h"
#include "in_out_functions.h"

cudaError_t performWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t performTestWithCuda(double* c, const double* a, const double* b, unsigned int size);

__global__ void addKernel(double *c, const double*a, const double*b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


__global__ void metropolisKernelTest(double* dev_H, double** dev_DelH, double* dev_DelH_sign, double* dev_B, int* dev_Y, int lenY, double* dev_E, double OldE, int* dev_bestSpinModel, int best_energy, int exchange_attempts, double T)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.x;
    curandState state;

    //int* dev_Selected_index
    extern __shared__ int dev_Selected_index[];

    for (int step = 0; step < exchange_attempts ; step++) {
        //compute Delata energy
        double deltaE = -1 * (1 - dev_Y[i]) * dev_H[i];

        // Make decision that a bit flip can be accepted
        curand_init(0, tid, 0, &state);
        //curand_init(seed, tid, 0, &state);
        //double pr = curand_uniform_double(&state)
        if ((deltaE < 0) || (curand_uniform_double(&state) < exp(-deltaE / T))) {
            dev_Selected_index[tid] = tid;
        }
        else {
            dev_Selected_index[tid] = -1;
        }

        // select which bit accepted
        for (int s = lenY; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                if (dev_Selected_index[tid] != -1 && dev_Selected_index[tid + s] != -1)
                {
                    if ((rand() % 2) == 1)
                        dev_Selected_index[tid] = tid + s;
                }
                else if (dev_Selected_index[tid] == -1 && dev_Selected_index[tid + s] != -1) {
                    dev_Selected_index[tid] = tid + s;
                }
            }
        }

        // based on flipped bit do some calculation
        if (tid == dev_Selected_index[0]) {
            dev_Y[tid] = 1 - dev_Y[tid];
            dev_E[step] = OldE + deltaE;
            OldE = dev_E[step];
            if (dev_E[step] < best_energy) {
                best_energy = dev_E[step];
                dev_bestSpinModel[tid] = dev_Y[tid];
            }
        }
        //__syncthreads();


        dev_H[tid] = dev_H[tid] + dev_DelH[tid][dev_Selected_index[0]];
        //dev_H[tid] = dev_H[tid] + dev_DelH[tid][dev_Selected_index[0]] * dev_DelH_sign[tid];


        if (tid == dev_Selected_index[0]) {
            for (int i = 0; i < lenY; i++) {
                dev_DelH[tid][dev_Selected_index[0]] *= -1;
            }
        }

    
            //dev_bestSpinModel[i] *= (1 - dev_Y[i]);
            //dev_E[counter] = exchange_attemps - counter;
    }




}
  
cudaError_t prepareMetropolisKernel(double* H, double* DelHGpu, double* WGpu, double* B, int* Y, int lenY, double* M, double* E, double T, int step, int exchange_attempts, double bestEnergy, int* bestSpinModel) {
    
       
    double* dev_H = 0;
    double* dev_DelH = 0;
    double* dev_DelH_sign;
    double* dev_W = 0;
    double* dev_B = 0;
    int* dev_Y = 0;


    int* dev_Flag;

    double* dev_E = 0;
    //double* dev_M = 0;
    int* dev_bestSpinModel=0;
    
    cudaError_t cudaStatus;


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for inputs and outputs.
    cudaMalloc((void**)&dev_H, lenY * sizeof(double));
    cudaMalloc((void**)&dev_DelH, lenY * lenY * sizeof(double));
    cudaMalloc((void**)&dev_DelH_sign, lenY * lenY * sizeof(int));
    cudaMalloc((void**)&dev_W, lenY * lenY * sizeof(double));
    cudaMalloc((void**)&dev_B, lenY * sizeof(double));
    cudaMalloc((void**)&dev_Y, lenY * sizeof(int));
    cudaMalloc((void**)&dev_E, exchange_attempts * sizeof(double));
    //cudaMalloc((void**)&dev_M, exchange_attempts * sizeof(double));
    cudaMalloc((void**)&dev_bestSpinModel, lenY * sizeof(int));
    cudaMalloc((void**)&dev_Flag, lenY * sizeof(int));
    


    
    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_H, H, lenY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_DelH, DelHGpu, lenY * lenY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_W, WGpu, lenY * lenY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, lenY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Y, Y, lenY * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bestSpinModel, bestSpinModel, lenY * sizeof(int), cudaMemcpyHostToDevice);
    



    // Launch a kernel on the GPU with one thread for each element.
    metropolisKernelTest <<<1, lenY >>> (dev_H, dev_DelH, dev_DelH_sign, dev_B, dev_Y,lenY, dev_E, dev_bestSpinModel, exchange_attempts, T);
    //metropolisKernelTest(double* dev_H, double** dev_DelH, double* dev_DelH_sign, double* dev_B, int* dev_Y, int lenY, double* dev_E, double OldE, int* dev_bestSpinModel, int best_energy, int* dev_Selected_index, int exchange_attempts, double T)


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.

    //cudaMemcpy(H, dev_H, lenY * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(DelH, dev_DelH, lenY * lenY * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(Y, dev_Y, lenY * sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(bestSpinModel, dev_bestSpinModel, lenY * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(E + step, dev_E, exchange_attempts * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(M + step, dev_M, exchange_attempts * sizeof(double), cudaMemcpyDeviceToHost);
       


Error:
    cudaFree(dev_H);
    cudaFree(dev_DelH);
    cudaFree(dev_W);
    cudaFree(dev_B);
    cudaFree(dev_Y);
    cudaFree(dev_E);
    //cudaFree(dev_M);
    cudaFree(dev_Flag);
    cudaFree(dev_bestSpinModel);

    return cudaStatus;


}

/* ***************************************************************** */


__global__ void metropolisKernel(double* dev_H, double* dev_DelH, double* dev_W, double* dev_B, int* dev_Y, double* dev_E, double* dev_M, int* dev_bestSpinModel)
{
    int i = threadIdx.x;
    
}





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

    double** DelHGpu;
    double* WGpu;
    if (ExecuteMode == QUBOGPU) {
        DelHGpu = convertDelHtoGpuDelH(DelH, num_replicas, lenY);
        WGpu = convert2Dto1D(W, lenY, lenY);
    }

    //testHamiltonianPreparation(W, B, lenY);
    //int tempValue;
    //cin >> tempValue;

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



    // Preperation of replica exchange parameters
    int exchangeFlag = 0;   // Flag to enable the exchange between neighbour replicas
    

    // Perform the Metropolis function numberOfIteration times for each replica 
    for (int step = 1; step < numberOfIteration; step++) {
        //cout << "step: " << step << endl;
        cout << "before prepareMetropolisKernel calling/ step: " << step << endl;
        for (int r = 0; r < num_replicas; r++) {
            T = Temperature[r];
            double previousE = E[r][step - 1];
            //for (int spin = 0; spin < lenY/25; spin++) 
            
            if (ExecuteMode == QUBOGPU) {                
                prepareMetropolisKernel(H[r], DelHGpu[r], WGpu, B, Y[r], lenY, M[r], E[r], T, step, exchange_attempts, bestEnergy, bestSpinModel);
                
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


// Helper function for using CUDA to add vectors in parallel.
cudaError_t performTestWithCuda(double* c, const double* a, const double* b, unsigned int size)
{
    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    cudaError_t cudaStatus;

    
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);

    return cudaStatus;

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel <<<1, size >>> (dev_a, dev_b, dev_c);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

/*****************************************************************************************/

// Helper function for using CUDA to add vectors in parallel.
cudaError_t performWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

