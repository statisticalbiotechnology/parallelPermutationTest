import numpy as np
from numba import cuda
from cuda_fill_array import fill_array_u4_v_u2, fill_array_f8_v_u2, fill_array_f8_v_u4, fill_array_u8_v_u2, fill_array_f4_v_u2, fill_array_f8_v_u8
import math
#from utils2 import getNumeratorCPU

import numpy as np
from multiprocessing.pool import ThreadPool
import itertools
from multiprocessing import sharedctypes
import multiprocessing 

def getNumeratorCPU(m, n, S, z, dtype, cores=-1):
    if cores == -1:
        cores = multiprocessing.cpu_count()
        
    
    N = np.zeros([S + 1, m], dtype)
    N_old = N.copy()
    
    
    N = np.ctypeslib.as_ctypes(np.zeros((S + 1, m)))
    N_shared_array = sharedctypes.RawArray(N._type_, N)
    
    def multiprocessing_func(xLim, yLim, N_shared_array):
        tmp = np.ctypeslib.as_array(N_shared_array)
    
        underLimX, upperLimX = xLim
        underLimY, upperLimY = yLim

    
        for j in range(underLimY, upperLimY):
            for s in range(underLimX, upperLimX):
                if j >= m + 1 or s > S or j < 1:
                    pass
                elif s == 0 and int(j-1)==0:
                    tmp[s,j-1] =  1
                elif i < j:
                    tmp[s, j - 1] = 0
                elif j > 1 and z[i-1] <= s:
                    tmp[s,j-1] = N_old[s - z[i -1], j-2] + N_old[s,j-1]
                elif j > 1 and z[i-1] > s:
                    tmp[s,j-1] = N_old[s,j-1]
    
    starttime = time.time()
        
    x_len, y_len = S + 1, m + 1

    batchsize =  int(x_len / cores)
        
    for i in range(1,(m+n)+1): 
        processes = []
        for underLim in range(0, x_len, batchsize):
            if underLim + batchsize > x_len:
                upperLim = x_len
            else:
                upperLim = underLim + batchsize
                
            p = multiprocessing.Process(target=multiprocessing_func, args=((underLim, upperLim),(0,y_len), N_shared_array))
            processes.append(p)
            p.start()
        
        for process in processes:
            process.join()
        
        N_old = np.ctypeslib.as_array(N_shared_array).copy()
        

    return N_old[:,-1]

class significance_of_mean_cuda(object):
    """Fast p-value calculation.
    Credit:
        Relevant githubs:
            Micheal H: https://github.com/hoehleatsu/permtest
            Lukas Käll: https://github.com/statisticalbiotechnology/exactpermutation
        Relevant articles:
            Bert Green: A Practical Interactive Program for Randomization Tests of Location
            Marcello Pagano & David Tritchler: On Obtaining Permutation Distributions in Polynomial Time
            Jens Gebhard and Norbert Schmitz: Permutation tests- a revival?! II. An efficient algorithm for computing the critical region 
    """
    def __init__(self,num_bin = None, dtype_v=np.uint64, dtype_A=np.float64, new_version=False, verbose=True, gpu=True, n_cores=-1):
        """
        Args:
            num_bin (int): NThe number of bins to divide each sample-set.
            dtype_v (type): The datatype of small arrays and values.
            dtype_A (type): The datatype type of large arrays.
        """
        self.gpu = gpu
        self.n_cores = n_cores
        self.num_bin = num_bin
        self.dtype_v = dtype_v
        self.dtype_A = dtype_A
        self.verbose = verbose
        if self.dtype_v == np.uint16 and self.dtype_A == np.uint32:
            self._get_perm = fill_array_u4_v_u2

        elif self.dtype_v == np.uint16 and self.dtype_A == np.uint64:
            self._get_perm = fill_array_u8_v_u2

        elif self.dtype_v == np.uint16 and self.dtype_A == np.float32:
            self._get_perm = fill_array_f4_v_u2

        elif self.dtype_v == np.uint16 and self.dtype_A == np.float64:
            self._get_perm = fill_array_f8_v_u2
            
        elif self.dtype_v == np.uint32 and self.dtype_A == np.float64:
            self._get_perm = fill_array_f8_v_u4
        elif self.dtype_v == np.uint64 and self.dtype_A == np.float64:
            self._get_perm = fill_array_f8_v_u8
        elif self.dtype_v == np.float32 and self.dtype_A == np.float64:
            self._get_perm = fill_array_f8_v_f4
        elif self.dtype_v == np.float64 and self.dtype_A == np.float64:
            self._get_perm = fill_array_f8_v_f8
        else:
            raise ValueError("The selected value tkype combination is currently not available!")
   
    def _get_digitized_score(self, X, bins):
        """Digitize the values for each sample.

        Args:
            X (array): Concatenated sample from original samples A and B.
            bins(int): The number of bins to divide the sample values.

        Returns:
            digitized array
        """
        digitized = np.zeros(X.shape,dtype=self.dtype_v)
        for i, (x,b) in enumerate(zip(X,bins)):
            digitized[i,:] = np.digitize(x, b).astype(self.dtype_v) - 1
        return digitized

    def _ensure_contiguous(self, z, S, A0, A1, init=None):
        """Assert all arrays are contiguous.
        Args:
            A0 (array): Initialize A0 array.
            A1 (array): Second array to start fill.
            S(int): Sum up to :m.
            z(array): Digitized array.
        Returns:
            Contiguous arrays.
        """
        return (np.ascontiguousarray(z, self.dtype_v), np.ascontiguousarray(S, self.dtype_v),
                np.ascontiguousarray(A0, self.dtype_A), np.ascontiguousarray(A1, self.dtype_A))

    def _load_gpu(self, z, S, A0, A1):
        """Load arrays onto the GPU's.
        Args:
            z (array): Digitized array.
            A0 (array): Initialized A0 array.
            A1 (array): Second array to fill.
            S(int): Sum up to :m.
        Returns:
            GPU arrays.
        """
        stream = cuda.stream()
        return (stream, cuda.to_device(z, stream), cuda.to_device(S, stream),
                cuda.to_device(A0, stream), cuda.to_device(A1, stream))
            
            

    def _run_calculations(self, dA0, dA1, dz, dS, length, threadsperblock, blockspergrid, stream, A0, A1):
        """Start to fill the rest of working array.
        Args:
            dz (array): Digitized GPU-array.
            dA0 (array): Initialized A0 GPU-array.
            dA1 (array): A1 GPU-array.
            dS(int): Sum up to :m. GPU-array
            blockdim(tripple): Dimension of GPU-block
            griddim(tripple): Dimension of GPU-grid
            stream: GPU-stream
        Returns:
            The two last calculated sub-arrays (onto the GPU), dA0 and dA1.
        """
        for i in range(1, length + 1):
                self._get_perm[blockspergrid, threadsperblock, stream](dA0, dA1, self.dtype_v(i), dz, dS)
                tmp = dA0
                dA0 = dA1
                dA1 = tmp  
        return dA0, dA1
    
    def _get_calculated_array(self, dA0,dA1, A1,A0, stream, m):
        """Retrieve the final subarray to host.
        Args:
            dA0 (array): Initialized A0 GPU-array.
            dA1 (array): A1 GPU-array.
            m (int): Length of sample A.
            stream: GPU-stream.
        Returns:
            Returns the necessary part for the p-value calculation from the final sub-array.
        """
        dA0.to_host(stream)
        #dA1.to_host(stream)
        stream.synchronize()
        return A1[:, -1, :]

    def _calculate_p_values(self, Z, n_samples, S, A ,bins, midP=False):
        """Calculate p-value for each sub-array
        Args:
            Z (array): The necessary part of the array to calculate sample p-values.
            n_samples (int): The total number of samples.
            S(int): Sum up to :m.
            A(array): The Values from sample A.
            bins(array): Bins for digitization.
        Returns:
            p-values
        """
        P = np.zeros(n_samples)
        for i, (a,b) in enumerate(zip(A,bins)):
            pmf = Z[:,i] / np.sum(Z[:,i])
            a_ = np.digitize(a, b).astype(self.dtype_v) - 1
            if midP:
                P[i] = pmf[int(sum(a_))] / 2 + np.sum(pmf[int(sum(a_))+1:(int(S[i])+1)])
            else:
                P[i] = np.sum(pmf[int(sum(a_)):(int(S[i])+1)])
        return P

    def _exact_perm_gpu_shift(self, m, n, S, z):
        """Run the shift-method on the GPU.
        Args:
            m (int): Sample size of sample A
            n (int): Sample size of sample B
            S(int): Sum up to :m.
            z (array): Digitized array.
        Returns:
             A necessary part of the calculated array to retrieve p-values.
        """
        n_samples = z.shape[0]

        A0 = np.zeros([int(np.max(S)) + 1, m, n_samples], self.dtype_A)

        NN, NM, _ = A0[:, :, :].shape
        
        threadsperblock = (64, 3, 2)
        blockspergrid = (int(np.ceil((NN)/ threadsperblock[0])),
                         int(np.ceil(NM/threadsperblock[1] + 1)),
                         int(np.ceil(n_samples / threadsperblock[2] + 1)))
            
        A1 = np.zeros([int(np.max(S)) + 1, m, n_samples], self.dtype_A)

        AMem = A1.nbytes / 1048576
        zMem = z.nbytes / 1048576
        SMem = S.nbytes / 1048576

        AMem = A1.nbytes / 1000000
        zMem = z.nbytes / 1000000
        SMem = S.nbytes / 1000000


        memoryAllocation = 2*AMem + zMem + SMem
        if self.verbose:
            print("This data requires {} MiB on the GPU.".format(memoryAllocation))

        z, S, A0, A1 = self._ensure_contiguous(z, S, A0, A1)
        
        stream, dz, dS, dA0, dA1 = self._load_gpu(z,S,A0,A1)
        
        dA0, dA1 = self._run_calculations(dA0, dA1, dz, dS, m + n, threadsperblock, blockspergrid, stream, A0, A1)
        return self._get_calculated_array(dA0,dA1, A1,A0, stream, m)
        

    def run(self, A, B, midP =False):
        """Run method on the GPU.
        Args:
            A (array): Samples A.
            B (array): Samples B.
        Returns:
             p-values
        """
        
        self.m = A.shape[1]
        self.n = B.shape[1]

        self.n_samples = A.shape[0]

        X = np.concatenate([A,B],axis=1)
        X.sort()

        if not self.num_bin:
            self.num_bin = np.ceil(np.max(X)) - np.floor(np.min(X)) + 1

        bins = np.asarray([np.linspace(np.min(x), np.max(x), self.num_bin) for x in X])



        self.digitized = self._get_digitized_score(X, bins)

        #Add the empty set.
        self.digitized = np.pad(self.digitized, ((0,0),(1,0)),'constant', constant_values=(0, 0))
        self.m = self.m + 1

        self.S = np.sum(self.digitized[:, self.m:], axis=1).astype(self.dtype_v)
        if self.gpu:
            self.numerator = self._exact_perm_gpu_shift(int(self.m), int(self.n), self.S, self.digitized)
        else:
            #self._exact_perm_cpu_shift(int(self.m), int(self.n), self.S[0], list(self.digitized[0]))
            self.numerator = getNumeratorCPU(int(self.m), int(self.n), self.S[0], list(self.digitized[0]), self.n_cores) 
            self.numerator = self.numerator.reshape(-1,1)
        self.p_values = self._calculate_p_values(self.numerator, self.n_samples, self.S, A, bins, midP)

    def _exact_perm_cpu_shift(self, m, n, S, z, dtype=np.float64):
        N = np.zeros([S + 1, m], dtype)
        N_old = N.copy()
    
        for i in range(1,(m+n)+1):
            for j in range(0, m +1):
                for s in range(0,S+1):
                
                    if j >= m + 1 or s > S or j < 1:
                        pass
                    elif s == 0 and int(j-1)==0:
                        N[s,j-1] =  1
                    elif i < j:
                        N[s, j - 1] = 0
                    elif j > 1 and z[i-1] <= s:
                        N[s,j-1] = N_old[s - z[i -1], j-2] + N_old[s,j-1]
                    elif j > 1 and z[i-1] > s:
                        N[s,j-1] = N_old[s,j-1]
    
            N_old = N.copy()

        return N_old[:,-1]

        
    def get_numerator(self):
        """Get numerator.
        Returns:
             numerator
        """
        return self.numerator
    
    def get_p_values(self):
        """Get p-values.
        Returns:
             Get p-values
        """
        return self.p_values

    
