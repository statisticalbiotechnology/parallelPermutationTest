/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


 #include <stdio.h>
 #include <cuda_runtime.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <assert.h>
 #include <iostream>
 #include "cuda_runtime_api.h"
 #include "cuda.h"
 #include <string> 
 
 using namespace std;

unsigned int safeDiv(int a, int b) {
/*
 * Function:  Divsion that ceils the quotienten to get an int
 * --------------------
 *
 *  a: The numerator
 *  b: The denominator
 * 
 *  returns: Ceiled quotienten
 */
    return ceil(a / (b*1.0));
}


bool getThreads(int width, int n_samples, int& threadX, int& threadZ) {
/*
 * Function:  Assign threads and blocks
 * --------------------
 *
 *  width: The width of the array
 *  n_samples: Number of experiments
 *  threadX: Number of threads assigned to x-axis
 *  threadZ: Number of threads assigned to z-axis
 *  returns: True, if there exists an thread assignment
 */

    int maxBlocks = 65535;
    threadX = 512;
    threadZ = 1;

    bool search = true;
    bool X = false;
    bool Z = false;

    while (search) {

        if (safeDiv(width, threadX) < maxBlocks) {
            X = true;
        } else {
            printf ("Couldn't allocate enough threads! Consider decreaseing the number of experiments");
            exit (EXIT_FAILURE);
        }

        if (safeDiv(n_samples, threadZ) < maxBlocks) {
            Z = true;
        } else {
            threadX = threadX / 2;
            threadZ = threadZ * 2;
            X = false;
        }

        if (X && Z) {
            search = false;
        }

        if (threadX ==0) {
            printf ("Couldn't allocate enough threads! Consider decreaseing the number of experiments");
            return false;
        }
    
        if (threadZ ==0) {
            printf ("Couldn't allocate enough threads! Consider decreaseing the number of experiments");
            return false;
        }
    
    }
    return true;
  }


bool check_memory(size_t mem){
/*
 * Function:  Check if there is enough memory on GPU
 * --------------------
 *
 *  mem: Memory required to allocate data
 *
 *  returns: True, if there exists enough memory
 */

    int num_gpus;
    size_t free, total;
    cudaGetDeviceCount( &num_gpus );
    for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
        cudaSetDevice( gpu_id );
        int id;
        cudaGetDevice( &id );
        cudaMemGetInfo( &free, &total );
        if (free <= mem) {
            cout << "Error: Your need " << ceil((mem)/1000000 * 0.953674) << " Mib, but there are only " << ceil(free /1000000*0.953674) << " Mib avaialble. Consider running your data in batches."<< endl;
            return false;
        }
        
    }
    return true;
  }

size_t cuda_mem_avilable(){
/*
 * Function:  Get available memory on GPU
 * --------------------
 *
 *  returns: Available memory on GPU(bits)
 */
    int num_gpus;
    size_t free, total;
    cudaGetDeviceCount( &num_gpus );
    for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
        cudaSetDevice( gpu_id );
        int id;
        cudaGetDevice( &id );
        cudaMemGetInfo( &free, &total );

        return free;
        
        
    }
}

size_t mem_required(int m, int n, int S,int n_samples) {
/*
 * Function: Get required memory for allocation of data
 * --------------------
 *
 *  m: Number of samples in A
 *  n: Number of samples in B
 *  S: Maxsum in experiments
 *  n_samples: Number of experiments
 * 
 *  returns: Memory of data(bits)
 */
    int height = m + 1;
    int width = S + 1;

    int z_height = m+n;
    size_t memory = z_height * n_samples * sizeof(int) + 2 * width * height * n_samples * sizeof(double);
    return memory;

}
  
__global__ void compute_perm(double *d_N,double *d_N_old,int *d_z, int height, int width, int n_samples, int sample_len, int i) {
/*
 * Function: Get required memory for allocation of data
 * --------------------
 *
 *  d_N: Array to add counts of permutations 
 *  d_N_old: Array of old counts of permutations 
 *  d_z: Combination of elements from A and B
 *  height: Height of the arrays d_N and d_N_old
 *  width: Width of the arrays d_N and d_N_old
 *  n_samples: Number of experiments, or depth of d_N and d_N_old
 *  sample_len: Length of d_z
 *  i: Iteration i
 *  returns: Updated counts in d_N for iteration i
 */
    
 
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z; 
      
     
    if(j < height && s < width && d < n_samples)  {
        if (i<j) { 
            d_N[(j + d * height)*width  + s] = 0;
        }
        else if (j == 0 && d_z[sample_len * d + i-1] == s) { 
            d_N[(j + d * height) * width + s] = d_N_old[(j + d * height) * width + s] + 1;
        }
        else if (j > 0 && d_z[sample_len * d + i-1] <= s) {    
            d_N[(j + d * height) * width + s] = d_N_old[((j-1) + d * height) * width  + (s - d_z[sample_len * d + i-1])] + d_N_old[(j + d*height) * width + s];
        } 
        else {
            d_N[(j + d * height)*width + s] = d_N_old[(j + d*height) * width + s];
        }   
    }
}
  
double * greenCuda(int *Z_data_, int * all_S_, int m, int n, int S, int n_samples) {
/*
 * Function: Compute permutation distribution
 * --------------------
 *
 *  Z_data_: Combinvation of A and B for all experiments
 *  all_S_: All sums for each experiment
 *  m: Number of samples in A
 *  n: Number of samples in B
 *  S: Maxsum in experiments
 *  n_samples: Number of experiments
 *
 *  returns: Permutation distribution for all experiments
 */
    cudaError_t err = cudaSuccess;
    int i;
    int s;
      
    int height = m + 1;
    int width = S + 1;
    int z_height = m+n;
 
    int *z;
    double *N, *N_old;

    size_t memory = z_height * n_samples * sizeof(int) + 2 * width * height * n_samples * sizeof(double);
    if (!check_memory(memory)){
        return NULL;
    };
    
    cudaMallocManaged(&z, sizeof(int) * z_height * n_samples);
    cudaMallocManaged(&N, sizeof(double) * width * height * n_samples);
    cudaMallocManaged(&N_old, sizeof(double) * width * height * n_samples);

    double *dx = (double *)malloc(sizeof(double) * width * n_samples);

    for (i = 0; i < z_height*n_samples; ++i) {
        z[i] = Z_data_[i];
    }
   
    int threadX, threadZ;
    if (!getThreads(width, n_samples, threadX, threadZ)){
        return NULL;
    }
     
    dim3 threads(threadX,1,threadZ);
    auto safediv = [](auto a, auto b) {return static_cast<unsigned int>(ceil(a / (b*1.0))); };
    dim3 blocks(safediv(width, threads.x), safediv( height, threads.y),safediv( n_samples, threads.z));
  
    for (i = 1; i < (m + n) + 1; i++) {
        if (i % 2 == 1) {   
            compute_perm<<<blocks,threads>>>(N, N_old, z, height, width, n_samples, z_height, i);    

        } 
        else {
            compute_perm<<<blocks,threads>>>(N_old, N, z, height, width,n_samples, z_height, i);    
        }
    }

    cudaDeviceSynchronize();
         
    double msum;

    if (i % 2 == 1) {
   
        for (i=0; i< n_samples; i++) {
       
            for (s = 0; s < S+1; s++) {
                dx[(S + 1) * i + s] = N_old[((m-1) + i * height)*width  + s];         
            }       
        }
    } else {

        for (i=0; i< n_samples; i++) {
          
           
            for (s = 0; s < S+1; s++) {
                dx[(S + 1) * i + s] = N[((m-1) + i * height)*width  + s];
            } 
        }
    }  

    cudaFree(N);
    cudaFree(N_old);
    cudaFree(z);
 
    return dx; 
}


  