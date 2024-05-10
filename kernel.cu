
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>

#include "helpFunction.h"
#include "constant.h"
#include "ising.h"
#include "in_out_functions.h"


__device__ void printLog(const char* action, int step, int temprature_index, int blockId, int threadId, double previousEnergy, double deltaE, double newEnergy, double H, double del_H, int flipped_bit, int index_del_h) {
    // Use printf for printing within device functions
    printf("%d, %d, %d, %d, %s, %lf, %lf, %lf, %lf, %lf, %d, %d\n", step, blockId, temprature_index, threadId, action, previousEnergy, deltaE, newEnergy, H, del_H, flipped_bit, index_del_h);
}

//Gpu kernel for metropolis function => Almost same as sequential version of metropolis just perform tasks in parallel.
__global__ void metropolisKernel(double* dev_H, double* dev_DelH, int* dev_DelH_sign, int* dev_Y, int* dev_Selected_index, int dev_lenY, double* dev_E, int* dev_bestSpinModel, double* best_energy, int exchange_attempts, double T)
{
    int tid = threadIdx.x;
    
    //extern __shared__ int sdata[];
    curandState state;

    curand_init(clock64(), tid, clock64(), &state);
    if (tid < dev_lenY) {
        for (int step = 1; step < exchange_attempts; step++) {
            
            //compute Delata energy
            double deltaE = -1 * (1 - 2* dev_Y[tid]) * dev_H[tid];
            //printf("Thread %d: Local variable value = %f\n", threadIdx.x, deltaE);
            

            // Make decision that a bit flip can be accepted
            
            
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


__device__ unsigned int generate_random_int_v02(unsigned int* seed) {
    unsigned int a = 1103515245;
    unsigned int c = 12345;
    //unsigned int m = 2147483648;
    *seed = (*seed * a + c);
    return (*seed / 10);

    //return (*seed) % m;
}
__device__ unsigned int generate_random_int_v01(unsigned int* seed) {
    // LCG parameters
    unsigned int a = 1664525;
    unsigned int c = 1013904223;

    // Update seed using LCG formula
    *seed = (*seed * a + c) * UINT_MAX;
    *seed = (*seed >> 1);
    //printf("*seed: %u\n", *seed);
    // Return the random number
    return *seed;
}

__device__ double generateRandomDouble(double minVal, double maxVal, unsigned int* seed) {

    // Generate random doubles
    unsigned int a = 1664525;
    unsigned int c = 1013904223;

    // Update seed using LCG formula
    *seed = (*seed * a + c);
    // Generate random unsigned integer using LCG


    // Map the random integer to a double value within the specified range
    double randomDouble = minVal + ((maxVal - minVal) * (*seed)) / (double)(UINT_MAX);


    return randomDouble;
}

__device__ void choise_between_two_bits(int threadId, int s, int* Shared_selected_index, unsigned int random_int) {
    int b_flag;
    int d_tid = Shared_selected_index[threadId]; int d_tid_s = Shared_selected_index[threadId + s];
    if (d_tid == -1) {
        Shared_selected_index[threadId] = d_tid_s;
    } else if (d_tid_s != -1)
    {
        // Generate a random integer (0 or 1)
        b_flag = (random_int & 1);
        Shared_selected_index[threadId] = (b_flag * d_tid_s + (1 - b_flag) * d_tid);
        /* the upper line is equivalent with the following code: */
        /*if ((random_int & 1) == 1)//find the least significant bit
            Shared_selected_index[threadId] = Shared_selected_index[threadId + s];*/
    }
}


__device__ void warpReduce_to_select_a_bit(int* Shared_selected_index, unsigned int tid, int array_size, unsigned int random_int) {
    //int random_int = *p_random_int;
    switch (array_size) {
    case 64:
        random_int = (random_int >> 1); choise_between_two_bits(tid, 32, Shared_selected_index, random_int);
    case 32:
        if (tid < 16) {random_int = (random_int >> 1); choise_between_two_bits(tid, 16, Shared_selected_index, random_int); }
    case 16:
        if (tid < 8) {
            random_int = (random_int >> 1); choise_between_two_bits(tid, 8, Shared_selected_index, random_int); }
    case 8:
        if (tid < 4) {
            random_int = (random_int >> 1); choise_between_two_bits(tid, 4, Shared_selected_index, random_int); }
    case 4:
        if (tid < 2) {
            random_int = (random_int >> 1); choise_between_two_bits(tid, 2, Shared_selected_index, random_int); }
    case 2:
        if (tid < 1) {
            random_int = (random_int >> 1); choise_between_two_bits(tid, 1, Shared_selected_index, random_int); }
    }

    /*
    if (array_size >= 64) { 
        random_int = (random_int >> 1); choise_between_two_bits(tid, 32, Shared_selected_index, random_int); }
    if (array_size >= 32) if (tid < 16) {
        random_int = (random_int >> 1); choise_between_two_bits(tid, 16, Shared_selected_index, random_int); }
    if (array_size >= 16)if (tid < 8) {
        random_int = (random_int >> 1); choise_between_two_bits(tid, 8, Shared_selected_index, random_int);  }
    if (array_size >= 8) if (tid < 4) {
        random_int = (random_int >> 1); choise_between_two_bits(tid, 4, Shared_selected_index, random_int);  }
    if (array_size >= 4) if (tid < 2) {
        random_int = (random_int >> 1); choise_between_two_bits(tid, 2, Shared_selected_index, random_int);  }
    if (array_size >= 2) if (tid < 1) {
        random_int = (random_int >> 1); choise_between_two_bits(tid, 1, Shared_selected_index, random_int);  }*/
    //*p_random_int = random_int;
}


// reduce_to_select_a_bit is another version of function "select_flipping_bit"
__device__ void reduce_to_select_a_bit(int* Shared_selected_index, unsigned int tid, int array_size, unsigned int random_int) {
    //int random_int = *p_random_int;
    int warp_array_size = array_size;
    //if (array_size >= 2048) { if (tid < 1024) {random_int = (random_int >> 1); flip_a_bit_between_pair(tid, 1024, Shared_selected_index, random_int); } __syncthreads(); }
    if (array_size >= 1024) { if (tid < 512) { random_int = (random_int >> 1); choise_between_two_bits(tid, 512, Shared_selected_index, random_int); } __syncthreads(); }
    if (array_size >= 512) { if (tid < 256) { random_int = (random_int >> 1); choise_between_two_bits(tid, 256, Shared_selected_index, random_int); } __syncthreads(); }
    if (array_size >= 256) { if (tid < 128) { random_int = (random_int >> 1); choise_between_two_bits(tid, 128, Shared_selected_index, random_int); } __syncthreads(); }
    if (array_size >= 128) { warp_array_size = 64; if (tid < 64) { random_int = (random_int >> 1); choise_between_two_bits(tid, 64, Shared_selected_index, random_int); } __syncthreads(); }
    if (tid < 32) warpReduce_to_select_a_bit(Shared_selected_index, tid, warp_array_size, random_int);
    //*p_random_int = random_int;
    __syncthreads();
}

__device__ void reduce_to_select_a_bit_v2(int* Shared_selected_index, unsigned int tid, int array_size, unsigned int random_int) {
    int warp_array_size = array_size;
    switch (array_size) {
    case 2048:
        if (tid < 1024) { random_int = (random_int >> 1); choise_between_two_bits(tid, 1024, Shared_selected_index, random_int); } __syncthreads();
    case 1024:
        if (tid < 512) { random_int = (random_int >> 1); choise_between_two_bits(tid, 512, Shared_selected_index, random_int); } __syncthreads();
    case 512:
        if (tid < 256) { random_int = (random_int >> 1); choise_between_two_bits(tid, 256, Shared_selected_index, random_int); } __syncthreads();
    case 256:
        if (tid < 128) { random_int = (random_int >> 1); choise_between_two_bits(tid, 128, Shared_selected_index, random_int); } __syncthreads();
    case 128:
        if (tid < 64) { random_int = (random_int >> 1); choise_between_two_bits(tid, 64, Shared_selected_index, random_int); } __syncthreads(); warp_array_size = 64;

    
    default:
        
        
        switch (warp_array_size) {
        case 64:
            if (tid < 32) { random_int = (random_int >> 1); choise_between_two_bits(tid, 32, Shared_selected_index, random_int); }
        case 32:
            if (tid < 16) { random_int = (random_int >> 1); choise_between_two_bits(tid, 16, Shared_selected_index, random_int); }
        case 16:
            if (tid < 8) { random_int = (random_int >> 1); choise_between_two_bits(tid, 8, Shared_selected_index, random_int); }
        case 8:
            if (tid < 4) { random_int = (random_int >> 1); choise_between_two_bits(tid, 4, Shared_selected_index, random_int); }
        case 4:
            if (tid < 2) { random_int = (random_int >> 1); choise_between_two_bits(tid, 2, Shared_selected_index, random_int); }
        case 2:
            if (tid < 1) { random_int = (random_int >> 1); choise_between_two_bits(tid, 1, Shared_selected_index, random_int); }
        }
    
    }
    __syncthreads();
}

__device__  void select_flipping_bit_v2(int select_index_size, int threadId, int* Shared_selected_index, unsigned int random_int) {
    int s = select_index_size >> 1;
    int warp_array_size = select_index_size;
    for (; s > 32; s >>= 1)
    {
        if (threadId < s)
        {
            int b_flag;
            int d_tid = Shared_selected_index[threadId]; int d_tid_s = Shared_selected_index[threadId + s];
            if (d_tid == -1) {
                Shared_selected_index[threadId] = d_tid_s;
            }
            else if (d_tid_s != -1)
            {   // Generate a random integer (0 or 1)
                random_int = random_int >> 1;
                b_flag = (random_int & 1);
                Shared_selected_index[threadId] = (b_flag * d_tid_s + (1 - b_flag) * d_tid);

            }
        }
        __syncthreads();
        warp_array_size = 64;
    }
    if (threadId < 32) {
        switch (warp_array_size) {
        case 64:
            random_int = (random_int >> 1); choise_between_two_bits(threadId, 32, Shared_selected_index, random_int);
        case 32:
            if (threadId < 16) { random_int = (random_int >> 1); choise_between_two_bits(threadId, 16, Shared_selected_index, random_int); }
        case 16:
            if (threadId < 8) { random_int = (random_int >> 1); choise_between_two_bits(threadId, 8, Shared_selected_index, random_int); }
        case 8:
            if (threadId < 4) { random_int = (random_int >> 1); choise_between_two_bits(threadId, 4, Shared_selected_index, random_int); }
        case 4:
            if (threadId < 2) { random_int = (random_int >> 1); choise_between_two_bits(threadId, 2, Shared_selected_index, random_int); }
        case 2:
            if (threadId < 1) { random_int = (random_int >> 1); choise_between_two_bits(threadId, 1, Shared_selected_index, random_int); }
        }
    }
    __syncthreads();
}

__device__  void select_flipping_bit_v4(int select_index_size, int threadId, int* Shared_selected_index, unsigned int random_int) {
    int s = select_index_size >> 1;
    int warp_array_size = select_index_size;
    char b_flag1, b_flag2;
    int d_tid;
    int d_tid_s;
    for (; s > 32; s >>= 1)
    {
        if (threadId < s)
        {
            d_tid = Shared_selected_index[threadId]; d_tid_s = Shared_selected_index[threadId + s];
            b_flag1 = !(d_tid ^ -1);
            random_int = random_int >> 1;
            b_flag2 = (random_int & 1);
            Shared_selected_index[threadId] = (1 - b_flag1) * (b_flag2 * d_tid_s + (1 - b_flag2) * d_tid) + (b_flag1 * d_tid_s);

            //d_tid = Shared_selected_index[threadId]; d_tid_s = Shared_selected_index[threadId + s];
            //if (d_tid == -1) {
            //    Shared_selected_index[threadId] = d_tid_s;
            //}
            //else if (d_tid_s != -1)
            //{   // Generate a random integer (0 or 1)
            //    random_int = random_int >> 1;
            //    b_flag = (random_int & 1);
            //    Shared_selected_index[threadId] = (b_flag * d_tid_s + (1 - b_flag) * d_tid);
            //}
        }
        __syncthreads();
        warp_array_size = 64;
    }
    if (threadId < 32) {
        s = warp_array_size >> 1;
        d_tid = Shared_selected_index[threadId];
        for (; s > 0; s >>= 1)
        {
            if (threadId < s)
            {                
                d_tid_s = Shared_selected_index[threadId + s];
                b_flag1 = !(d_tid ^ -1);
                random_int = random_int >> 1;
                b_flag2 = (random_int & 1);
                d_tid = (1 - b_flag1) * (b_flag2 * d_tid_s + (1 - b_flag2) * d_tid) + (b_flag1 * d_tid_s);
                Shared_selected_index[threadId] = d_tid;
                //if (d_tid == -1) {
                //    Shared_selected_index[threadId] = d_tid_s;
                //}
                //else if (d_tid_s != -1)
                //{   // Generate a random integer (0 or 1)
                //    random_int = random_int >> 1;
                 //   b_flag = (random_int & 1);
                 //   Shared_selected_index[threadId] = (b_flag * d_tid_s + (1 - b_flag) * d_tid);

               // }
            }
        }
    }
    __syncthreads();
}

// This version should be completed. 
__device__  void select_flipping_bit_v03(int select_index_size, int threadId, int* Shared_selected_index, int* Shared_selected_index2, int* Shared_selected_index_level2, unsigned int random_int) {
    //select_index_size >> 1;
    if (threadId< (select_index_size>>1)){
        int offset = (threadId<<1) & DATA_OFFSET_MASK;
        int data_id=threadId & DATA_ID_MASK;
        int b_flag;
        int data_index = offset + data_id;
        int d_tid = Shared_selected_index[data_index];
        int d_tid_s;
        for (int s = 32; s > 0; s >>= 1)
        {
            d_tid_s = Shared_selected_index[data_index + s];
            if (d_tid == -1) {
                d_tid = d_tid_s;
                Shared_selected_index2[data_index] = d_tid_s;
            }
            else if (d_tid_s != -1)
            {   // Generate a random integer (0 or 1)
                random_int = random_int >> 1;
                b_flag = (random_int & 1);
                d_tid = (b_flag * d_tid_s + (1 - b_flag) * d_tid);
                Shared_selected_index2[data_index] = d_tid;
            }
            s >>= 1;
            d_tid_s = Shared_selected_index2[data_index + s];
            if (d_tid == -1) {
                d_tid = d_tid_s;
                Shared_selected_index[data_index] = d_tid_s;
            }
            else if (d_tid_s != -1)
            {   // Generate a random integer (0 or 1)
                random_int = random_int >> 1;
                b_flag = (random_int & 1);
                d_tid = (b_flag * d_tid_s + (1 - b_flag) * d_tid);
                Shared_selected_index[data_index] = d_tid;
            }
        }
        if (data_id == 0) {
            offset = offset >> 6;
            Shared_selected_index_level2[offset] = d_tid;
        }
        __syncthreads();
    }
}


__device__  void select_flipping_bit(int select_index_size, int threadId, int* Shared_selected_index, unsigned int random_int) {
    int s = select_index_size >> 1;
    int b_flag;
    for (; s > 0; s >>= 1)
    {
        if (threadId < s)
        {            
            int d_tid = Shared_selected_index[threadId]; int d_tid_s = Shared_selected_index[threadId + s];
            if (d_tid == -1) {
                Shared_selected_index[threadId] = d_tid_s;
            }
            else if (d_tid_s != -1)
            {   // Generate a random integer (0 or 1)
                random_int = random_int >> 1;
                b_flag = (random_int & 1);
                Shared_selected_index[threadId] = (b_flag * d_tid_s + (1 - b_flag) * d_tid);
            }
            /*
            if (Shared_selected_index[threadId] != -1 && Shared_selected_index[threadId + s] != -1)
            {
                // Generate a random integer (0 or 1)
                random_int = random_int >> 1;
                if ((random_int & 1) == 1)//find the least significant bit
                    Shared_selected_index[threadId] = Shared_selected_index[threadId + s];
            }
            else if (Shared_selected_index[threadId] == -1 && Shared_selected_index[threadId + s] != -1) {
                Shared_selected_index[threadId] = Shared_selected_index[threadId + s];
            }*/
        }
        __syncthreads();
    }
}


__device__ int fill_Shared_selected_index(int threadId, int* Shared_selected_index, int blockSize, double random_double) {
    Shared_selected_index[threadId] = -1;
    int next_valid_index = 0;
    if (random_double < HIGH_TEMP_FAST_FLIP_PRO) {
        Shared_selected_index[threadId] = threadId;
        //printf("%d\n", threadId);
    }
    __syncthreads();
    if (threadId == 0) {
        for (int i = 0; i < blockSize; i++) {
            if (Shared_selected_index[i] != -1) {
                Shared_selected_index[next_valid_index] = Shared_selected_index[i];
                if (next_valid_index != i)
                    Shared_selected_index[i] = -1;
                next_valid_index++;
            }
        }
    }
    __syncthreads();
    return next_valid_index;
}


//dev_H, dev_DelH, dev_DelH_sign, dev_Y, dev_Selected_index, lenY, dev_E, dev_bestSpinModel, dev_bestenergy, numberOfIteration
__global__ void full_mode_metropolisKernel(double* dev_H, double* dev_DelH, int* dev_DelH_sign, int* dev_Y, int* dev_Selected_index, const int select_index_size, double* dev_E, int* dev_bestSpinModel, double* dev_best_energy, int numberOfIteration, int exchange_attempts, int extend_exchange_step, double* dev_Temprature, int number_of_temp, int debug_mode, int program_config)
{   
    extern __shared__ char sharedMemory[];

    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int tid = blockId * blockDim.x + threadId;
    int temprature_index = blockId % number_of_temp;
    char temp_index_direction = (blockId+1) % 2;
    int second_previous_j = -1, previous_j = -1;
    bool stop_flag = false;
    char Y;
    char best_config_Y;
    unsigned int seed =  (tid + 1) * SEED_COEF * clock64();
    
    //double best_energy = dev_best_energy[blockId];

    

    // Define pointers to different shared memory segments
    //int* Shared_Y = (int*)sharedMemory;
    //int* Shared_bestSpinModel = (int*)(sharedMemory + (blockDim.x) * sizeof(int));
    
    //int* Shared_selected_index = (int*)(sharedMemory + 2 * (blockDim.x) * sizeof(int));
    int* Shared_selected_index = (int*)sharedMemory ;

    //double* Shared_H = (double*)(sharedMemory + (select_index_size +2 * (blockDim.x)) * sizeof(int)); //
    double* Shared_H = (double*)(sharedMemory + select_index_size  * sizeof(int));

    //double* Shared_Temprature = (double*)(sharedMemory + (select_index_size + 2 * (blockDim.x)) * sizeof(int)+ (blockDim.x) * sizeof(double));
    double* Shared_Temprature = (double*)(sharedMemory + (select_index_size * sizeof(int)) + (blockDim.x) * sizeof(double));

    //double* shared_best_energy = (double*)(sharedMemory + (select_index_size + 2 * (blockDim.x)) * sizeof(int) + (blockDim.x + gridDim.x) * sizeof(double));
    double* shared_best_energy = (double*)(sharedMemory + (select_index_size * sizeof(int)) + (blockDim.x + number_of_temp) * sizeof(double));
    *shared_best_energy = dev_best_energy[blockId];
    //double* Shared_DelH = (double*)(sharedMemory+blockDim.x*sizeof(double));
    //double* Shared_E = (double*)(sharedMemory + (blockDim.x+1) * blockDim.x * sizeof(double));
    
    double* shared_previous_step_energy = (double*)(sharedMemory + (select_index_size * sizeof(int)) + (blockDim.x + number_of_temp + 1) * sizeof(double));
    if (threadId == 0) {
        *shared_previous_step_energy = dev_best_energy[blockId];
    }


    Y = dev_Y[tid];
    //Shared_Y[threadId] = Y;
    //Shared_bestSpinModel[threadId] = Y;
    best_config_Y = Y;
    char No_new_suggestion = extend_exchange_step;
    Shared_selected_index[threadId] = -1;
    if (threadId + blockDim.x < select_index_size) {
        Shared_selected_index[threadId + blockDim.x] = -1;
    }

    if (threadId < number_of_temp) {
        Shared_Temprature[threadId] = dev_Temprature[threadId];
        
    }

    Shared_H[threadId] = dev_H[tid];

    
    __syncthreads();
     


    int index_base = select_index_size* blockId;
    
    if (temp_index_direction==0) {
        temp_index_direction = -1;
    }
    
    
    //if (blockId == gridDim.x - 1 && blockId % 2 == 0) {
    if ((temprature_index == number_of_temp - 1 && temp_index_direction == 1)|| (temprature_index == 0 && temp_index_direction == -1)) {
        stop_flag = true;
        temp_index_direction *= -1;
        //printf("blockId %d the stop_flag is on \n", blockId);
    }

    double deltaE;
    double random_double;
    unsigned int random_int;
    int next_exchange = exchange_attempts;
    bool high_temp_first_time = true;
    int next_valid_index=-1;
    //point 1: time: 0.63 ms
    //curandState state;
    // point 2: This part is time consuming, we should change it!!!> 162 ms
    //curand_init(clock64(), tid, clock64(), &state);
    
    if (threadId == 0) {
        if((*shared_previous_step_energy) != dev_E[blockId * numberOfIteration]) printf("blockId: %d, bestEnergy %lf %lf , %d\n", blockId, *shared_previous_step_energy, dev_E[blockId * numberOfIteration], (*shared_previous_step_energy) == dev_E[blockId * numberOfIteration]);
        //printf("blockId: %d   temprature %d : %lf \n", blockId, temprature_index, Shared_Temprature[temprature_index]);
    }

    for (int step = 1; step < numberOfIteration; step++) {
        //point3:  time : 754 ms


        // Generate a random double nubmer
        random_double = generateRandomDouble(0, 1, &seed);
        random_int = seed;


        if ((temprature_index == number_of_temp - 1) && (program_config & HIGH_TEMP_FAST_FLIP) == HIGH_TEMP_FAST_FLIP) {
            if (high_temp_first_time == true) {
                next_valid_index = fill_Shared_selected_index(threadId, Shared_selected_index, blockDim.x, random_double);
                //if (tid == 0)printf("\t step: %d next_valid_index : %d\n", step, next_valid_index);
                high_temp_first_time = false;
            }
            else if(threadId==0 && next_valid_index>1)
            {
                next_valid_index--;
                CURRENT_FLIP= Shared_selected_index[next_valid_index];

            }
            else if (threadId == 0 && next_valid_index == 1)
            {
                CURRENT_FLIP = -1;
                
            }
            __syncthreads();
            //if (tid == 0)printf("step:%d CURRENT_FLIP %d, dev_E %lf\n ",step, CURRENT_FLIP, dev_E[step-1]);
            if(CURRENT_FLIP==-1) {
                next_exchange = step;
                //high_temp_first_time = true;
            }
            if (threadId == CURRENT_FLIP) {
                deltaE = -1 * (1 - 2 * Y) * Shared_H[threadId];
            }
            high_temp_first_time = false;
        }
        else {
            //compute Delata energy
            deltaE = -1 * (1 - 2 * Y) * Shared_H[threadId];



            // Make decision that a bit flip can be accepted

            Shared_selected_index[threadId] = -1;
            if (deltaE < 0) {
                //dev_Selected_index[index_base+threadId] = threadId;
                Shared_selected_index[threadId] = threadId;
                //if ((debug_mode & DEBUG_DELTA_FLIP) != 0) printLog("delta candiate", step, temprature_index, blockId, threadId, dev_E[blockId * numberOfIteration + step - 1], deltaE, 0, Shared_H[threadId], 0, -1, -1);
            }
            else if (random_double < expf(-deltaE / Shared_Temprature[temprature_index])) {
                Shared_selected_index[threadId] = threadId;
                //if ((debug_mode & DEBUG_RANDOM_FLIP) != 0) printLog("random candiate", step, temprature_index, blockId, threadId, dev_E[blockId * numberOfIteration + step - 1], deltaE, 0, Shared_H[threadId], 0, -1, -1);
                //printf("\titeration:%d BId: %d, tId: %d, flip suggestion by randomness. DeltaE: %lf \n", step, blockId, threadId, deltaE);
            }

            //point 6: time: 30,517

            // This syncthread add around 3 seconds to execution time.       
            __syncthreads();
            // point 7: time:  33,878



            // select which bit accepted 
            select_flipping_bit(select_index_size, threadId, Shared_selected_index, random_int);
            //select_flipping_bit_v2(select_index_size, threadId, Shared_selected_index, random_int);

            //select_flipping_bit_v4(select_index_size, threadId, Shared_selected_index, random_int); // for scenario select_index >=64

            //reduce_to_select_a_bit(Shared_selected_index, threadId, select_index_size, random_int);
            //reduce_to_select_a_bit_v2(Shared_selected_index, threadId, select_index_size, random_int);
            //point 8: time : 333,611




            //__syncthreads();
            // point 9: time: 338,722    338,964
        }
            
            // based on the flipped bit j update parameters
            int j = CURRENT_FLIP; //Shared_selected_index[0];
            
            if (j == -1) {
                No_new_suggestion -= 1;
                if (threadId == 0) {
                    // Log the Energy when there is not any bit to flip
                    if ((debug_mode & DEBUG_ENERGY_RECORD_LOG) == DEBUG_ENERGY_RECORD_LOG) {
                        dev_E[blockId * numberOfIteration + step] = dev_E[blockId * numberOfIteration + step - 1];
                    }
                    //if ((debug_mode & DEBUG_SELECTED_FLIP) != 0) printLog("Nothing", step, temprature_index, blockId, threadId, dev_E[blockId * numberOfIteration + step - 1], 0, dev_E[blockId * numberOfIteration + step - 1], Shared_H[threadId], 0, j, -1);
                }
            }
            else {
                
                if (threadId == j) {
                    Y = 1 - Y;

                    //if ((debug_mode & DEBUG_ENERGY_RECORD_LOG) == DEBUG_ENERGY_RECORD_LOG)
                    *shared_previous_step_energy = *shared_previous_step_energy + deltaE;
                    if ((debug_mode & DEBUG_ENERGY_RECORD_LOG) == DEBUG_ENERGY_RECORD_LOG) {
                        dev_E[blockId * numberOfIteration + step] = *shared_previous_step_energy;
                    }
                    //dev_E[blockId * numberOfIteration + step] = dev_E[blockId * numberOfIteration + step - 1] + deltaE;

                    //if ((debug_mode & DEBUG_SELECTED_FLIP) != 0) printLog("flipped", step, temprature_index, blockId, threadId, dev_E[blockId * numberOfIteration + step - 1], deltaE, dev_E[blockId * numberOfIteration + step - 1] + deltaE, Shared_H[threadId], 0, j, -1);
                }
                if(j==previous_j || j == second_previous_j)
                    No_new_suggestion -= 1;
                else {
                    No_new_suggestion = extend_exchange_step;
                    second_previous_j = previous_j;
                    previous_j = j;
                }
                //if (tid == 0)
                  //  printf("step: flip,%d,%d\n",step, j);
            }
            

            /*
            if (threadId == j) {
                Y = 1 - Y;
                dev_E[blockId * numberOfIteration + step] = dev_E[blockId * numberOfIteration + step - 1] + deltaE;
                if ((debug_mode & DEBUG_SELECTED_FLIP) != 0) printLog("flipped", step, temprature_index, blockId, threadId, dev_E[blockId * numberOfIteration + step - 1], deltaE, dev_E[blockId * numberOfIteration + step - 1] + deltaE, Shared_H[threadId], 0, j, -1);
            }else if(j==-1 && threadId==0){
                // Log the Energy when there is not any bit to flip
                dev_E[blockId * numberOfIteration + step] = dev_E[blockId * numberOfIteration + step - 1];
                if ((debug_mode & DEBUG_SELECTED_FLIP) != 0) printLog("Nothing", step, temprature_index, blockId, threadId, dev_E[blockId * numberOfIteration + step - 1], 0, dev_E[blockId * numberOfIteration + step-1], Shared_H[threadId], 0, j, -1);
            }*/
            __syncthreads();
            // point 10: tiem: 128,444 or 129,322, 129,240  or  129,439 
            
            
            if (*shared_previous_step_energy < *shared_best_energy) {
            //if (dev_E[blockId * numberOfIteration + step] < *shared_best_energy) {
                //if (threadId == 0) *shared_best_energy = dev_E[blockId * numberOfIteration + step];
                if (threadId == 0) *shared_best_energy = *shared_previous_step_energy;
                best_config_Y = Y;
                if ((program_config & FAST_CONVERGE) == FAST_CONVERGE) {
                    //if (tid == 0 && temprature_index != 0)printf("\t\t\t step: %d drop temp to zero level\n", step);
                    //if (tid == 0)printf("** \t step: %d, temp_index: %d drop temp\n", step, temprature_index);
                    temprature_index = 0;
                    temp_index_direction = 1;
                    if (next_exchange < step + extend_exchange_step)
                        next_exchange = step + extend_exchange_step;                        
                }

                //if(threadId==0) printf("\treplica: %d, bestEnergy %lf \n", blockId, *shared_best_energy);
                //if ((debug_mode & DEBUG_FIND_BEST_ENERGY) != 0 && threadId == 0) printLog("best energy", step, temprature_index, blockId, threadId, dev_E[blockId * numberOfIteration + step - 1], 0, dev_E[blockId * numberOfIteration + step], 0, 0, j, -1);
            }
            
            //time: almost same as previous part

            if (j != -1) {
                //Update H                  (dev_DelH : replica * lenY * lenY)
                Shared_H[threadId] = Shared_H[threadId] + dev_DelH[blockId * blockDim.x * blockDim.x + threadId * blockDim.x + j];
                
                // Update delta_H
                dev_DelH[blockId * blockDim.x * blockDim.x + threadId * blockDim.x + j] *= -1;
                //if ((debug_mode & DEBUG_UPDATE_H) != 0) printLog("Update H", step, temprature_index, blockId, threadId, 0, 0, 0, Shared_H[threadId], dev_DelH[blockId * blockDim.x * blockDim.x + threadId * blockDim.x + j], j, blockId * blockDim.x * blockDim.x + threadId * blockDim.x + j);

                /*if (step > 2 && dev_E[blockId * numberOfIteration + step] == dev_E[blockId * numberOfIteration + step - 2])
                    No_new_suggestion -= 1;
                else
                    No_new_suggestion = extend_exchange_step;
                */

            }
            if (No_new_suggestion == 0) {

                //if (tid == 0)printf("\tstep:%d temp index %d No_new_suggestion is zero\n", step, temprature_index);
                No_new_suggestion = extend_exchange_step;
                if ((program_config & FAST_CONVERGE) == FAST_CONVERGE) {
                    temp_index_direction = 1;
                    //if (tid == 0)printf("** \t step: %d, temp_index: %d increase No_new_suggestion\n", step, temprature_index);
                    temprature_index += temp_index_direction;
                    stop_flag = false;
                    next_exchange = step + exchange_attempts;
                }
            }
            __syncthreads();
            
            //point 11: time: 361,359   or   356,756   or   
            

            
            
            if (step == next_exchange) {
                //if (tid == 0)printf("** \t step: %d, temp_index: %d\n", step, temprature_index);
                No_new_suggestion = extend_exchange_step;
                if ((program_config & TEMPERATURE_CIRCULAR) == TEMPERATURE_CIRCULAR) {
                    if (temprature_index==0) temprature_index = number_of_temp-1;
                    else temprature_index--;
                }else if (stop_flag == false) {
                    temprature_index += temp_index_direction;
                    high_temp_first_time = true;
                    if (temprature_index == 0 || temprature_index == number_of_temp - 1) {
                        stop_flag = true;
                        temp_index_direction = temp_index_direction * -1;
                    }
                    if ((debug_mode & DEBUG_EXCHANGE) != 0 && threadId == 0) printLog("Exchange", step, temprature_index, blockId, threadId, 0, 0, 0, 0, 0, -1, -1);
                }
                else {
                    stop_flag = false;
                }
                //if (tid == 0)
                //    printf("\t\tstep: %d, index: %d, temp: %lf\n", step, temprature_index, Shared_Temprature[temprature_index]);
                next_exchange = step + exchange_attempts;
            }

            

            //if (threadId == 0 && bE != dev_best_energy[blockId]) {
            //    printf("S %d bE in block %d is %lf \n", step, blockId, dev_best_energy[blockId]);
            //    bE = dev_best_energy[blockId];
            //}
        }
    //}
        //return;
    //point 12: time: 348,522     347624
        
    if ((debug_mode & DEBUG_SAVE_DEVICE_RESULT) != 0 && threadId == 0) printLog("Final", numberOfIteration, temprature_index, blockId, threadId, dev_E[(blockId+1) * numberOfIteration - 1], 0, 0, 0, 0, -1, -1);
    dev_best_energy[blockId] = *shared_best_energy;
    //if ((debug_mode & DEBUG_ENERGY_RECORD_LOG) == DEBUG_ENERGY_RECORD_LOG) dev_Y[tid] = Shared_Y[threadId];
    if ((debug_mode & DEBUG_ENERGY_RECORD_LOG) == DEBUG_ENERGY_RECORD_LOG) dev_Y[tid] = Y;
    //dev_bestSpinModel[tid] = Shared_bestSpinModel[threadId];
    dev_bestSpinModel[tid] = best_config_Y;

    //point 13: time: 348315
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

cudaError_t allocateMemory_with_size(double** dev_H, double** dev_DelH, double** dev_W, double** dev_B, double** dev_E, int** dev_bestSpinModel, int** dev_Y, int** dev_Selected_index, int** dev_lenY, int** dev_DelH_sign, double** dev_bestenergy, double** dev_Temprature, int* sizeArray, int debug_mode) {

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

cudaError_t copyMemoryFromHostToDevice_with_size(double* H, double* dev_H, double* DelH, double* dev_DelH, int* DelH_sign, int* dev_DelH_sign, double* WGpu, double* dev_W, int* Y, int* dev_Y, double* E, double* dev_E, double bestEnergy, double* bestEnergyArray, double* dev_bestenergy, int* bestSpinModel, int* dev_bestSpinModel, int* dev_Selected_index, double* dev_Temprature, double* Temprature, int* sizeArray, int debug_mode) {

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

    //double* bestEnergyArray = new double[sizeArray[9]];
    //fill1Darray(bestEnergyArray, bestEnergy, sizeArray[9]);
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

void copyMemoryFromDeviceToHost_with_size(int* vector_Y, int* dev_Y, int* bestSpinModel, int* vector_best_all_config, int* dev_bestSpinModel, double* bestEnergy, double* dev_bestenergy, double* vector_E, double* dev_E, int* memory_size, int debug_mode)
{
    cudaError_t cudaStatus;

    double* bestEnergyArray = new double[memory_size[9]];
    cudaStatus = cudaMemcpy(bestEnergyArray, dev_bestenergy, memory_size[9] * sizeof(double), cudaMemcpyDeviceToHost);
    int bestEnergyIndex = findMinIndex(bestEnergyArray, memory_size[9]);

    *bestEnergy = bestEnergyArray[bestEnergyIndex];
    printf("Value of bestEnergy: %lf\n", *bestEnergy);

    // Copy output vector from GPU buffer to host memory.
    //cudaMemcpy(H, dev_H, lenY * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(DelHGpu, dev_DelH, lenY * lenY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vector_Y, dev_Y, memory_size[5] * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vector_best_all_config, dev_bestSpinModel, memory_size[5] * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(bestSpinModel, dev_bestSpinModel + bestEnergyIndex * memory_size[8], memory_size[8] * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(vector_E , dev_E, memory_size[7] * sizeof(double), cudaMemcpyDeviceToHost);


    //cudaStatus = cudaMemcpy(DelH_sign, dev_DelH_sign, lenY * sizeof(int), cudaMemcpyDeviceToHost);
    checkErrorCuda(cudaStatus, "cudaMemcpy failed! Copy form Device to Host");


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

void FreeMemoryDevice(double* dev_H, double* dev_DelH, double* dev_W, double* dev_B, double* dev_E, int* dev_bestSpinModel, int* dev_Y, int* dev_Selected_index, int* dev_lenY, int* dev_DelH_sign, double* dev_bestenergy, int debug_mode) {
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
cudaError_t prepare_full_MetropolisKernel(double* vector_H, double* vector_DelH, int* vector_DelH_sign, double* WGpu, double* B, int* vector_Y , int* vector_best_all_config, int lenY, double* vector_E, double* Temperature, int number_of_temp, int exchange_attempts, int extend_exchange_mode, double*  bestEnergy, double* bestEnergyArray, int* bestSpinModel, int replica, int numberOfIteration, int debug_mode, int program_config) {

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

    float milliseconds = 0;

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
    cudaStatus = allocateMemory_with_size(&dev_H, &dev_DelH, &dev_W, &dev_B, &dev_E, &dev_bestSpinModel, &dev_Y, &dev_Selected_index, &dev_lenY, &dev_DelH_sign, &dev_bestenergy, &dev_Temprature, memory_size, debug_mode);

    cudaStatus = copyMemoryFromHostToDevice_with_size(vector_H, dev_H, vector_DelH, dev_DelH, vector_DelH_sign, dev_DelH_sign, WGpu, dev_W, vector_Y, dev_Y, vector_E, dev_E, *bestEnergy, bestEnergyArray, dev_bestenergy, bestSpinModel, dev_bestSpinModel, dev_Selected_index, dev_Temprature, Temperature, memory_size, debug_mode);

    
    
    //int size_of_shared_Memory = (2 * lenY + select_index_size)* sizeof(int) + (lenY+ replica + 1) * sizeof(double);
    int size_of_shared_Memory = ((select_index_size) * sizeof(int)) + (lenY + number_of_temp + 2) * sizeof(double);
    //int size_of_shared_Memory = (2 * (lenY)+select_index_size) * sizeof(int) + (replica + ((lenY + 1) * lenY) + numberOfIteration) * sizeof(double);
    printf("size_of_shared_Memory : %d \n", size_of_shared_Memory);
    
    // Start recording time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch a kernel on the GPU with one thread for each element.
    full_mode_metropolisKernel << <replica, lenY , size_of_shared_Memory >> > (dev_H, dev_DelH, dev_DelH_sign, dev_Y, dev_Selected_index, select_index_size, dev_E, dev_bestSpinModel, dev_bestenergy, numberOfIteration, exchange_attempts, extend_exchange_mode, dev_Temprature, number_of_temp, debug_mode, program_config);
    

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    checkErrorCuda(cudaStatus, "prepare_full_MetropolisKernel launch failed : !");

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching metropolisKernel!\n", cudaStatus);
        cout << cudaGetErrorString(cudaStatus) << endl;;
        goto Error;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate and print the elapsed time
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time elapsed: %f ms\n", milliseconds);

    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);



    // Copy output vector from GPU buffer to host memory.
    
    copyMemoryFromDeviceToHost_with_size(vector_Y, dev_Y, bestSpinModel, vector_best_all_config, dev_bestSpinModel, bestEnergy, dev_bestenergy, vector_E, dev_E, memory_size, debug_mode);

    //cout << " \t\t\t Best Energy: " << bestEnergy << endl;

Error:
    FreeMemoryDevice(dev_H, dev_DelH, dev_W, dev_B, dev_E, dev_bestSpinModel, dev_Y, dev_Selected_index, dev_lenY, dev_DelH_sign, dev_bestenergy, debug_mode);

    return cudaStatus;
}



// prepare memory to call Gpu Kernel
cudaError_t prepareMetropolisKernel(double* H, double* DelHGpu, int* DelH_sign, double* WGpu, double* B, int* Y, int lenY, double* E, double T, int step, int exchange_attempts, double& bestEnergy, int* bestSpinModel, int replica, int debug_mode) {
    
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
    FreeMemoryDevice(dev_H, dev_DelH, dev_W, dev_B, dev_E, dev_bestSpinModel, dev_Y, dev_Selected_index, dev_lenY, dev_DelH_sign, dev_bestenergy,debug_mode);

    return cudaStatus;
}


/* ***************************************************************** */




// One step in metropolis algorithm
double metropolis(int ExecuteMode, double** W, double* B, double* H, double** DelH, int* X, int lenX, double OldE, double T, int step, int replica, int debug_mode) {
    double E = 0;
    int i = rand() % lenX;

    double deltaE;
    if (ExecuteMode == IsingMode)    deltaE = deltaEnergyIsing(ExecuteMode, W, B, X, lenX, i);
    else if (ExecuteMode == QUBOMode) deltaE = deltaEnergyQUBO(W, X, i, H);

    if ((deltaE < 0) || ((rand() / static_cast<double>(RAND_MAX)) < exp(-deltaE / T))) {
        if (ExecuteMode == IsingMode) {
            X[i] *= -1;
            E = energy(W, B, X, lenX, debug_mode);
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




double full_GPU_Mode(int num_replicas, double** W, double* B, int** Y,int** best_all_config, int lenY, double** E, int* bestSpinModel, double bestEnergy, int numberOfIteration, int exchange_attempts, int extend_exchange_mode, int number_of_temp, double minTemp, double maxTemp, int debug_mode, int program_config) {
    double** H = ComputeH_forAllReplica(num_replicas, W, B, Y, lenY);
    double*** DelH = ComputeDelH_forAllReplica(num_replicas, W, Y, lenY);
    int* DelH_sign = new int[num_replicas* lenY];
    //fill2DarrayInt(DelH_sign, 1, num_replicas, lenY);
    double* vector_DelH;
    double* vector_H;
    double* vector_E;
    int* vector_Y;
    int* vector_best_all_config; 
    double* vector_W = convert2Dto1D(W, lenY, lenY);;
    double* bestEnergyArray = new double[num_replicas];

    double* Temperature = new double[num_replicas];
    //if((program_config& TEMPERATURE_CIRCULAR)== TEMPERATURE_CIRCULAR)
    intitTemperature_circular_version(number_of_temp, minTemp, maxTemp, Temperature, program_config);
    //else
    //    intitTemperature(num_replicas, minTemp, maxTemp, Temperature, program_config);
    bestEnergy = initEnergy(num_replicas, W, B, Y, lenY, E, bestSpinModel, bestEnergyArray, debug_mode);
    cout << bestEnergy << endl;

    vector_DelH = VectorizedDelH(DelH, num_replicas, lenY);
    vector_H = convert2Dto1D(H, num_replicas, lenY);
    //printAllH(H, num_replicas, lenY);
    //printH(vector_H, num_replicas * lenY, "vector_H");
    vector_Y = convert_int_2Dto1D(Y, num_replicas, lenY);
    vector_best_all_config = new int[num_replicas* lenY];
    vector_E = convert2Dto1D(E, num_replicas, numberOfIteration);

    
    prepare_full_MetropolisKernel(vector_H, vector_DelH, DelH_sign, vector_W, B, vector_Y, vector_best_all_config, lenY, vector_E, Temperature, number_of_temp, exchange_attempts, extend_exchange_mode, &bestEnergy, bestEnergyArray, bestSpinModel, num_replicas, numberOfIteration, debug_mode, program_config);
    
    unVectorData(vector_Y, Y, vector_best_all_config, best_all_config, vector_E, E, numberOfIteration, num_replicas, lenY );
    
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
void ising(int ExecuteMode, double** W, double* B, int** Y, int** best_all_config, int lenY, double** E, double T, int num_replicas, int numberOfIteration, int exchange_attempts, int extend_exchange_mode, int* bestSpinModel, int number_of_temp, double minTemp, double maxTemp, int debug_mode, int program_config) {

    double bestEnergy;
    double* Temperature = new double[num_replicas];
    double** H = ComputeH_forAllReplica(num_replicas, W, B, Y, lenY);
    double*** DelH = ComputeDelH_forAllReplica(num_replicas, W, Y, lenY);
    int** DelH_sign = Declare2D_ArrayInt(num_replicas, lenY);
    fill2DarrayInt(DelH_sign, 1, num_replicas, lenY);
    double** DelHGpu;
    double* WGpu;
    double* bestEnergyArray = new double[num_replicas];
    
    

    
    //initialize the bestEnergy, Energy (E), Magnet (M), and bestSpinModel
    bestEnergy = initEnergy(num_replicas, W, B, Y, lenY, E, bestSpinModel, bestEnergyArray, debug_mode);

    cout << "The best initial energy: " << bestEnergy << endl;
    if (ExecuteMode == QUBOGPUFULL) {
        //prepare_Full_GPU_Mode();
        bestEnergy = full_GPU_Mode(num_replicas, W, B, Y, best_all_config, lenY, E, bestSpinModel, bestEnergy, numberOfIteration, exchange_attempts, extend_exchange_mode, number_of_temp, minTemp, maxTemp, debug_mode, program_config);
        return;
    }
    else if (ExecuteMode == QUBOGPU) {
        // Vectorization of parammeters for Gpu
        DelHGpu = convertDelHtoGpuDelH(DelH, num_replicas, lenY);
        WGpu = convert2Dto1D(W, lenY, lenY);
    }
    
    //initialize the Temperature array range
    intitTemperature(num_replicas, minTemp, maxTemp, Temperature, program_config);
    
    // Preperation of replica exchange parameters
    int exchangeFlag = 0;   // Flag to enable the exchange between neighbour replicas
    // Perform the Metropolis function numberOfIteration times for each replica 
    for (int step = 1; step < numberOfIteration; step++) {
        //cout << "step: " << step << endl;
        //cout << "******************* before prepareMetropolisKernel calling/ step: " << step << endl;
        for (int r = 0; r < num_replicas; r++) {
            T = Temperature[r];
            double previousE = E[r][step - 1];
                        
            if (ExecuteMode == QUBOGPU) {                
                prepareMetropolisKernel(H[r], DelHGpu[r], DelH_sign[r], WGpu, B, Y[r], lenY, E[r], T, step, exchange_attempts, bestEnergy, bestSpinModel, r, debug_mode);
                    
            }else { // Ising or QUBO
                previousE = metropolis(ExecuteMode, W, B, H[r], DelH[r], Y[r], lenY, previousE, T, step, r, debug_mode);
                E[r][step] = previousE;
                if (bestEnergy > E[r][step]) {
                    memcpy(bestSpinModel, Y[r], sizeof(int) * lenY);
                    bestEnergy = E[r][step];
                }
            }            
        }
        // Replica exchange attempts
        if (ExecuteMode != QUBOGPU && step % exchange_attempts == 0) {
            replicaExchange(Temperature, num_replicas, Y, E, step, H, DelH, exchangeFlag);
        }
        else if (ExecuteMode == QUBOGPU) {
            step = step + exchange_attempts-1;
            replicaExchangeGpu(Temperature, num_replicas, Y, E, step, H, DelHGpu, DelH_sign, exchangeFlag);
            
            cout << "after prepareMetropolisKernel calling/ step: " << step << endl;
        }

    }
    cout << "Best found Energy" << bestEnergy << endl; 
}

/**********************************************************************************************/





int main()
{

    int L = 0;                      // Number of spins for each replica
    int Lsqrt = 0;                  /* This parameter is just used for graphical representation of 2-D Ising model
                                     and it is not important for other cases Lsqrt * Lsqrt = L         */
    double T = 0.0;                 // Temperature ---> when 
    int num_replicas = 1;           // Number of replica in the replica exchange MCMC
    int number_of_temp = 0;
    double minTemp = 0;             // Min temperature of replica exchange MCMC
    double maxTemp = 0;             // Max temperature of replica exchange MCMC
    int numberOfIteration = 1;      // Number of Iteration in for MCMC
    int exchange_attempts = 0;      // After how many iteration, an exchange should be applied
    int extend_exchange_mode = 0;

    /*      E = -\sum_{i,j} A_{i,j} s_i s_j - \sum_i B_i s_i   */
    string Afile = "";              //      file path for A matrix in Ising (QUBO) model 
    string Bfile = "";              //      file path for Bias (B) in Ising (QUBO) model 
    string outputPath = "";          //     The path to the directory to save the output 
    int ExecuteMode = QUBOMode;     //      Execution mode: IsingMode or QUBOMode
    int debug_mode = NO_DEBUG;
    int problem_type = NORMAL;

    /*      Read the setting file and initialize the parameters    */
    int program_config = readSetting(L, Lsqrt, Afile, Bfile, outputPath, ExecuteMode, num_replicas, numberOfIteration, exchange_attempts, extend_exchange_mode, number_of_temp, minTemp, maxTemp, debug_mode, problem_type);
    cout << "program config" << program_config << endl;
    cout << "DEBUG_MODE" << debug_mode<<endl;
    cout << "problem_type: " << problem_type << endl;

    
    if (debug_mode != NO_DEBUG && debug_mode<= ALL_DEBUG_ADMIN) {
        std::cout << "step" << "," << "replica (blockId)" << "," << "Temprature Index" << "," << "bitIndex" << "," << "action" << "," << "previousEnergy" << "," << "deltaE" << "," << "newEnergy" << "," << "H" << "," << "del_H" << "," << "flipped_bit" << "," << "index_del_h" << endl;
    }
    //cout << "L: " << L << " Lsqrt: " << Lsqrt << " Afile: " << Afile << " Bfile: " << Bfile << " ExecuteMode: " << ExecuteMode << " num_replicas: " << num_replicas << " numberOfIteration: " << numberOfIteration << " exchange_attempts: " << exchange_attempts << endl;
    cout << "minTemp: " << minTemp << " maxTemp: " << maxTemp << endl;


    // Create the output folder
    createFolder(outputPath);

    //Initialize the spins for each replica
    int** X = createVector(ExecuteMode, L, num_replicas);
    int** best_all_config = createVector(ExecuteMode, L, num_replicas); // just for loging
    int* bestSpinModel = new int[L];            // Best found solution

    double** E = new double* [num_replicas];     //  Energy for different replica
    for (int r = 0; r < num_replicas; r++) {
        E[r] = new double[numberOfIteration];
    }

    //  Initialize the bias    
    double* B;
    B = initB(L, ReadDataFromFile, Bfile);

    //Initalize the interconnection of spins
    double** A;
    A = initW(L, Lsqrt, ReadDataFromFile, Afile, problem_type);
    //writeMatrixToFile(outputPath + "\\WInit.csv", A, L);

    normalize_optimization_problem(problem_type, A, B, L);
    B = initB(L, FillWithZero, "");
    // Log all the initial spin states
    if ((debug_mode & DEBUG_INIT_CONFIG) == DEBUG_INIT_CONFIG)
        writeSpinsInFile(num_replicas, X, L, Lsqrt, outputPath, "Initlattice");

    // Optimization Function
    ising(ExecuteMode, A, B, X, best_all_config, L, E, T, num_replicas, numberOfIteration, exchange_attempts, extend_exchange_mode, bestSpinModel,number_of_temp, minTemp, maxTemp, debug_mode, program_config);
    printf("Execution is done, waiting for loging\n");
    
    //  Record Logs: Magnet, Energy, final spin states, and best spin model
    recordLogs(outputPath, E, numberOfIteration, num_replicas, L, Lsqrt, X, bestSpinModel, best_all_config, debug_mode);

    return 0;
}