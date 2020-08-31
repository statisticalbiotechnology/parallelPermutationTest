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

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

 #include <stdio.h>

 // For the CUDA runtime routines (prefixed with "cuda_")
 #include <cuda_runtime.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <assert.h>
 #include <iostream>
 
 __global__ void count_permutation(double *d_N,double *d_N_old, int *d_z, int height, int width, int i)
  { 
      
      int s = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y; 
      
      if( j <= height && s < width) 
      {
         if (i<j) {
             d_N[width*j + s] = 0;
         }
         else if (j == 0 && d_z[i - 1] == s)
         {
             d_N[width*j + s] = d_N_old[width*j + s] + 1;
         }
        
         else if (j > 0 && d_z[i - 1] <= s)
         {   
             d_N[width*j + s] = d_N_old[width*(j-1) + (s - d_z[i -1])] + d_N_old[width*j + s];
         }
         
         else {
             d_N[width*j + s] = d_N_old[width*j + s];
         }
          
         
         
      }
  } 
 
 
  
  double * greenCuda(int *Z_data_, int m, int n, int S)
  {

    cudaError_t err = cudaSuccess;
    
     int i;
     int s;
 
     
     int height = m + 1;
     int width = S + 1;
 
     int z_height = m+n;
 
     int *z;


     double *dx = (double *)malloc(sizeof(double)*width);


    if (Z_data_ == NULL || dx==NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

     err = cudaMallocHost((void **) &z, sizeof(int)*z_height);
    


     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector z (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

     double *N;
     
     err = cudaMallocHost((void **) &N, sizeof(double)*width*m);

     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector N (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

     double *N_old;
     err = cudaMallocHost((void **) &N_old, sizeof(double)*width*m);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector N_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
     z = Z_data_; 
     /* for (i = 0; i < z_height; ++i) {
        z[i] = Z_data_[i];
       } */
      
     int *d_z;
     cudaMalloc((void **) &d_z, sizeof(int)*z_height);

     err = cudaMallocHost((void **) &N_old, sizeof(double)*width*m);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_z (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


     double *d_N;
     err = cudaMalloc((void **) &d_N, sizeof(double)*width*m);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_N (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


     double *d_N_old;
     err = cudaMalloc((void **) &d_N_old, sizeof(double)*width*m);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_N_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
     
     err = cudaMemcpy(d_z, z, sizeof(int)*z_height, cudaMemcpyHostToDevice);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector z (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

     err = cudaMemcpy(d_N_old, N_old, sizeof(double)*width*m, cudaMemcpyHostToDevice);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector N_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
     err = cudaMemcpy(d_N, N, sizeof(double)*width*m, cudaMemcpyHostToDevice);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector N (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
     
    dim3 threads(512, 2);
    auto safediv = [](auto a, auto b) {return static_cast<unsigned int>(ceil(a / (b*1.0))); };
    dim3 blocks(safediv(width, threads.x), safediv( m, threads.y));


     for (i = 1; i < (m + n) + 1; i++)
         {
         if (i % 2 == 1)
             {   
                 count_permutation<<<blocks,threads>>>(d_N, d_N_old, d_z, height, width, i);    
     
             } else {
                 count_permutation<<<blocks,threads>>>(d_N_old, d_N, d_z, height, width, i);    
             
             }
         }
 
         
      double msum = 0;
 
      
      
     if (i % 2 == 1) {
             
         cudaMemcpy(N_old, d_N_old, sizeof(double)*width*m, cudaMemcpyDeviceToHost);
             
         for (s = 0; s < S+1; s++) {
             msum += N_old[width*(m-1) + s];        
         }
         
             
         for (s = 0; s < S+1; s++) {
             dx[s] = N_old[width*(m-1) + s] / msum;     
         }
         
         
     } else {
         
         cudaMemcpy(N, d_N, sizeof(double)*width*m, cudaMemcpyDeviceToHost);
 
         for (s = 0; s < S+1; s++) {
             msum += N[width*(m-1) + s];
         }
         
             
         for (s = 0; s < S+1; s++) {
             dx[s] = N[width*(m-1) + s] / msum;    
         }
     } 
        
    err = cudaFree(d_N);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_N (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_N_old);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_N_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_z);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_z (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFreeHost(N);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector N (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFreeHost(N_old);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector N_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* err = cudaFreeHost(z);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector z (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } */
 
     return dx;
     
  }


  

  __global__ void count_permutation_multiple_samples(double *d_N,double *d_N_old, int *d_z, int *d_allS, int height, int width, int n_samples, int sample_len, int i)
  { 
    unsigned s = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned d = blockIdx.z * blockDim.z + threadIdx.z; 
    

    if( j <= height && s < width && d < n_samples && s <= d_allS[d]) 
    {
       if (i<j) {
           d_N[(width * j + s) * n_samples + d] = 0;
       }
       else if (j == 0 && d_z[sample_len * d + i-1] == s)
       {
           d_N[(width * j + s) * n_samples + d] = d_N_old[(width * j + s) * n_samples + d] + 1;
       }
      
       else if (j > 0 && d_z[sample_len * d + i-1] <= s)
       {   
           d_N[(width * j + s) * n_samples + d] = d_N_old[(width * (j-1) + (s - d_z[sample_len * d + i-1])) * n_samples + d] + d_N_old[(width * j + s) * n_samples + d];
       }
       
       else {
           d_N[(width * j + s) * n_samples + d] = d_N_old[(width * j + s) * n_samples + d];
       }
         
      }
  } 
 
 
 
 
  double * greenCudaMultipleSamples(int *Z_data_, int * all_S_, int m, int n, int S, int n_samples)
  {

    cudaError_t err = cudaSuccess;

     int i;
     int s;
     
     int height = m + 1;
     int width = S + 1;
 
     int z_height = m+n;
 
     int *z;
     int *allS;


     double *dx = (double *)malloc(sizeof(double) * width * n_samples);


    if (Z_data_ == NULL || dx==NULL || all_S_ == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

     err = cudaMallocHost((void **) &z, sizeof(int) * z_height * n_samples);
    
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector z (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMallocHost((void **) &allS, sizeof(int) * n_samples);
    
    if (err != cudaSuccess)
   {
       fprintf(stderr, "Failed to allocate device vector allS (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }
    
     double *N;
     
     err = cudaMallocHost((void **) &N, sizeof(double) * width * m * n_samples);

     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector N (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

     double *N_old;
     err = cudaMallocHost((void **) &N_old, sizeof(double) * width * m * n_samples);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector N_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < z_height*n_samples; ++i) {
        z[i] = Z_data_[i];
       }

       for (i = 0; i < n_samples; ++i) {
        allS[i] = all_S_[i];
       }
 
     /* z = Z_data_;
     allS = all_S_; */
      
     int *d_z;
     cudaMalloc((void **) &d_z, sizeof(int) * z_height * n_samples);

     int *d_allS;
     cudaMalloc((void **) &d_allS, sizeof(int) * n_samples);
     
     err = cudaMallocHost((void **) &N_old, sizeof(double) * width * m * n_samples);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector N_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


     double *d_N;
     err = cudaMalloc((void **) &d_N, sizeof(double) * width * m * n_samples);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_N (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


     double *d_N_old;
     err = cudaMalloc((void **) &d_N_old, sizeof(double) * width * m * n_samples);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_N_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
     
     err = cudaMemcpy(d_allS, allS, sizeof(int) * n_samples, cudaMemcpyHostToDevice);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_allS (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_z, z, sizeof(int) * z_height * n_samples, cudaMemcpyHostToDevice);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector z (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

     err = cudaMemcpy(d_N_old, N_old, sizeof(double) * width * m * n_samples, cudaMemcpyHostToDevice);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector N_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
     err = cudaMemcpy(d_N, N, sizeof(double) * width * m * n_samples, cudaMemcpyHostToDevice);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector N (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
     
    /* dim3 threads(512, 2, 1); */
    dim3 threads(4, 4, 4);
    auto safediv = [](auto a, auto b) {return static_cast<unsigned int>(ceil(a / (b*1.0))); };
    dim3 blocks(safediv(width, threads.x), safediv( m, threads.y), safediv( n_samples, threads.z));


    for (i = 1; i < (m + n) + 1; i++)
    {
        if (i % 2 == 1)
        {   
            count_permutation_multiple_samples<<<blocks,threads>>>(d_N, d_N_old, d_z, d_allS, height, width, n_samples, z_height, i);    

        } else {
            count_permutation_multiple_samples<<<blocks,threads>>>(d_N_old, d_N, d_z, d_allS, height, width,n_samples, z_height, i);    
        
        }
    }
 
         
    double msum;
 
      
      
    if (i % 2 == 1) {
        cudaMemcpy(N_old, d_N_old, sizeof(double) * width * m * n_samples, cudaMemcpyDeviceToHost);
        
        for (i=0; i< n_samples; i++) {
            msum = 0;
            for (s = 0; s < S+1; s++) {
                msum += N_old[(width * (m-1) + s) * n_samples + i];
            }
    
        
            for (s = 0; s < S+1; s++) {
                dx[(S + 1) * i + s] = N_old[(width * (m-1) + s) * n_samples + i] / msum;
                
            } 
        }
            
    } else {
    
        cudaMemcpy(N, d_N, sizeof(double) * width * m * n_samples, cudaMemcpyDeviceToHost);
    
        for (i=0; i< n_samples; i++) {
            msum = 0;
           for (s = 0; s < S+1; s++) {
            
                msum += N[(width * (m-1) + s) * n_samples + i];
            }
    
        
            for (s = 0; s < S+1; s++) {
                dx[(S + 1) * i + s] = N[(width * (m-1) + s) * n_samples + i] / msum;
            } 
        }
    } 
    
        
    err = cudaFree(d_N);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_N (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_N_old);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_N_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_z);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_z (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_allS);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_allS (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFreeHost(z);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector z (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFreeHost(allS);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector allS (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFreeHost(N);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector N (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFreeHost(N_old);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector N_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* err = cudaFreeHost(z);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector z (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
  */
     return dx;
     
  }



  /* __global__ void count_permutation(double *d_N,double *d_N_old, unsigned int *d_z, int height, int width, int i)
  { 
      
    unsigned int s = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; 
      
      if( j <= height && s < width) 
      {
         if (i<j) {
             d_N[width*j + s] = 0;
         }
         else if (j == 0 && d_z[i - 1] == s)
         {
             d_N[width*j + s] = d_N_old[width*j + s] + 1;
         }
        
         else if (j > 0 && d_z[i - 1] <= s)
         {   
             d_N[width*j + s] = d_N_old[width*(j-1) + (s - d_z[i -1])] + d_N_old[width*j + s];
         }
         
         else {
             d_N[width*j + s] = d_N_old[width*j + s];
         }
          
         
         
      }
  } 
 
 
  
  double * greenCuda(unsigned int *Z_data_, int m, int n, int S)
  {

    cudaError_t err = cudaSuccess;
    
     int i;
     int s;
 
     
     int height = m + 1;
     int width = S + 1;
 
     int z_height = m+n;
 
     unsigned int *z;


     double *dx = (double *)malloc(sizeof(double)*width);


    if (Z_data_ == NULL || dx==NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

     err = cudaMallocHost((void **) &z, sizeof(unsigned int)*z_height);
    


     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector z (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

     double *N;
     
     err = cudaMallocHost((void **) &N, sizeof(double)*width*m);

     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector N (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

     double *N_old;
     err = cudaMallocHost((void **) &N_old, sizeof(double)*width*m);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector N_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
     for (i = 0; i < z_height; ++i) {
        z[i] = Z_data_[i];
       }
      
     unsigned int *d_z;
     cudaMalloc((void **) &d_z, sizeof(unsigned int)*z_height);

     
     double *d_N;
     err = cudaMalloc((void **) &d_N, sizeof(double)*width*m);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_N (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


     double *d_N_old;
     err = cudaMalloc((void **) &d_N_old, sizeof(double)*width*m);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_N_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
     
     err = cudaMemcpy(d_z, z, sizeof(unsigned int)*z_height, cudaMemcpyHostToDevice);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector z (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

     err = cudaMemcpy(d_N_old, N_old, sizeof(double)*width*m, cudaMemcpyHostToDevice);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector N_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
     err = cudaMemcpy(d_N, N, sizeof(double)*width*m, cudaMemcpyHostToDevice);
     if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector N (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
     
    dim3 threads(64, 8);
    auto safediv = [](auto a, auto b) {return static_cast<unsigned int>(ceil(a / (b*1.0))); };
    dim3 blocks(safediv(width, threads.x), safediv( m, threads.y));


     for (i = 1; i < (m + n) + 1; i++)
         {
         if (i % 2 == 1)
             {   
                 count_permutation<<<blocks,threads>>>(d_N, d_N_old, d_z, height, width, i);    
     
             } else {
                 count_permutation<<<blocks,threads>>>(d_N_old, d_N, d_z, height, width, i);    
             
             }
         }
 
         
      double msum = 0;
 
      
      
     if (i % 2 == 1) {
             
         cudaMemcpy(N_old, d_N_old, sizeof(double)*width*m, cudaMemcpyDeviceToHost);
             
         for (s = 0; s < S+1; s++) {
             msum += N_old[width*(m-1) + s];        
         }
         
             
         for (s = 0; s < S+1; s++) {
             dx[s] = N_old[width*(m-1) + s] / msum;     
         }
         
         
     } else {
         
         cudaMemcpy(N, d_N, sizeof(double)*width*m, cudaMemcpyDeviceToHost);
 
         for (s = 0; s < S+1; s++) {
             msum += N[width*(m-1) + s];
         }
         
             
         for (s = 0; s < S+1; s++) {
             dx[s] = N[width*(m-1) + s] / msum;    
         }
     } 
        
    err = cudaFree(d_N);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_N (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_N_old);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_N_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_z);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_z (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFreeHost(N);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector N (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFreeHost(N_old);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector N_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFreeHost(z);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector z (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
     return dx;
     
  } */