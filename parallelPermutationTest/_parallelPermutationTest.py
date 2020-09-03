from permutationTest import greenCUDA, GreenOpenMP, Green, coinShift, gpu_available_mem

import numpy as np
import time

import sys

np.set_printoptions(threshold=sys.maxsize)


def batch(Arr, n=1):
    """Divide data into batches
        Args:
            Arr (array): Data to divide into batch
            n(int): batch-size
        Returns:
            Yields batches of data
    """
    l = Arr.shape[0]
    for ndx in range(0, l, n):
        yield Arr[ndx:min(ndx + n, l),:]


def GreenFloatCuda_memcheck(A,B, num_bin):
    def digitized_score(X, bins):
        """Digitize the values for each sample.
        Args:
            X (array): Concatenated sample from original samples A and B.
            bins(int): The number of bins to divide the sample values.
        Returns:
            digitized array
        """
        digitized = np.zeros(X.shape,dtype=np.int32)
        for i, (x,b) in enumerate(zip(X,bins)):
            digitized[i,:] = np.digitize(x, b).astype(np.int32) - 1
        return digitized
        
    def GreenFloatDataPreProcess(A, B, num_bin):
        """Preprocess data for Green's algorithm.
        Args:
            A (array): Samples A.
            B (array): Samples B.
            num_bin (int): Number of bins for digitization.
        Returns:
             digitized data, size of A, size of B, maximum sum, number of samples,
             bins for data, and array of Sums.  
        """
        
        m = A.shape[1]
        n = B.shape[1]
        n_samples = A.shape[0]

        X = np.concatenate([A,B],axis=1)
        X.sort()

        bins = np.asarray([np.linspace(np.min(x), np.max(x), int(num_bin)) for x in X])

        digitized = digitized_score(X, bins)

        S = np.sum(digitized[:, m:], axis=1).astype(np.int32)
        return digitized.ravel(), m, n, S.max(), n_samples, bins,S

    aDim, bDim = A.ndim,B.ndim
    assert aDim == bDim, "A and B does not have same dimensions!"
    
    if aDim == 1:
        A = A.reshape(1,-1)
        B = B.reshape(1,-1)
    a_n, b_n = A.shape[0], B.shape[0]
    assert a_n == b_n, "A and B does not have the same amount of experiments!"

    z, m, n, Smax, n_samples, bins, S =  GreenFloatDataPreProcess(A,B, num_bin)

    height = m + 1
    width = Smax + 1
    z_height = m+n

    int_bits = 4
    double_bits = 8

    memory_bits = z_height * n_samples * int_bits + 2 * width * height * n_samples * double_bits
    memory_MIB = round(memory_bits / 1000000 * 0.953674,2)

    gpu_mem = gpu_available_mem()
    
    diff = gpu_mem - memory_MIB
    diff = round(diff,2)

    if diff >0:
        print("The data requires {}Mib, and the GPU has {}Mib available, so there are {}Mib left after data allocation.".format(memory_MIB, gpu_mem, diff))
    else:
        print("Warning: The data requires {}Mib, and the GPU has {}Mib available, so there is {}Mib too little memory. Consider dividing the data into batches.".format(memory_MIB, gpu_mem, abs(diff)))


def GreenFloatCuda(A,B, num_bin, return_dperm=False, batch_size=None):
    """Calculate permutation distribution and p-values for float samples.
        Args:
            A (array): Contining experiments(row) with samples(column).
            B(array): Contining experiments(row) with samples(column).
            num_bin(int): Number of bins to divide samples in.
            return_dperm(boolean): Return permutation distribution.
            batch_size(int): Divide experiments in batches.
        Returns:
            Array with p-values and permutation distribution(optional).
    """
    def pValFloat(NNN, n_samples, S, A , B, bins, midP=False):
        """Calculate p-value for each sub-array
        Args:
            pdist (array): Permutation distribution
            n_samples (int): The total number of samples.
            S(int): Sum up to :m.
            A(array): The Values from sample A.
            bins(array): Bins for digitization.
            midP(boolean): Calculate mid p-value.
        Returns:
            p-values
        """
        P = np.zeros(n_samples)
        for i, (a, b, _bin, NN) in enumerate(zip(A, B, bins, NNN)):

            if len(a)>len(b):
                score = sum(np.digitize(b, _bin).astype(np.int32) - 1)
            else:
                score = sum(np.digitize(a, _bin).astype(np.int32) - 1)

            
            if midP:
                one_side = NN[score] / 2 
            else:
                one_side = NN[score]


            if score < S[i]:
                one_side += min(np.sum(NN[score + 1 :]), np.sum(NN[:score]))    
                p = one_side / float(np.sum(NN))
            
            P[i] = 2 * min( p, (1-p))

        return P

    def digitized_score(X, bins):
        """Digitize the values for each sample.
        Args:
            X (array): Concatenated sample from original samples A and B.
            bins(int): The number of bins to divide the sample values.
        Returns:
            digitized array
        """
        digitized = np.zeros(X.shape,dtype=np.int32)
        for i, (x,b) in enumerate(zip(X,bins)):
            digitized[i,:] = np.digitize(x, b).astype(np.int32) - 1
        return digitized
        
    def GreenFloatDataPreProcess(A, B, num_bin):
        """Preprocess data for Green's algorithm.
        Args:
            A (array): Samples A.
            B (array): Samples B.
            num_bin (int): Number of bins for digitization.
        Returns:
             digitized data, size of A, size of B, maximum sum, number of samples,
             bins for data, and array of Sums.  
        """
        
        m = A.shape[1]
        n = B.shape[1]
        n_samples = A.shape[0]

        X = np.concatenate([A,B],axis=1)
        X.sort()

        bins = np.asarray([np.linspace(np.min(x), np.max(x), int(num_bin)) for x in X])

        digitized = digitized_score(X, bins)

        K = min(m,n)

        S = np.sum(digitized[:, -K:], axis=1).astype(np.int32)
        return digitized.ravel(), min(m,n), max(m,n), S.max(), n_samples, bins, S

    aDim, bDim = A.ndim,B.ndim
    assert aDim == bDim, "A and B does not have same dimensions!"
    
    if aDim == 1:
        A = A.reshape(1,-1)
        B = B.reshape(1,-1)
    a_n, b_n = A.shape[0], B.shape[0]
    assert a_n == b_n, "A and B does not have the same amount of experiments!"

    p_val_list = list()
    pdist_list = list()

    _, _, _, Smax, _, _, _ =  GreenFloatDataPreProcess(A,B, num_bin)

    if not batch_size:
        a_batch_size, b_batch_size = A.shape[0], B.shape[0]
    else:
        assert batch_size <= A.shape[0], "Batch-size larger than number of examples."
        a_batch_size, b_batch_size = batch_size, batch_size
     
    for i, (a,b) in enumerate(zip(batch(A,a_batch_size), batch(B,b_batch_size))):
        z, m, n, _, n_samples, bins, S =  GreenFloatDataPreProcess(a,b, num_bin)
    
        z = z.astype(np.uint32)
        S = S.astype(np.int32)
    
        NN = np.array(greenCUDA(z, S, int(m), int(n), int(Smax), int(n_samples))).reshape(n_samples, Smax + 1)
        
        p_values = pValFloat(NN.reshape(n_samples, Smax+1), n_samples, S, a ,b,bins)

        pdist_list.append(NN)
        p_val_list.append(p_values)

    pdist_arr = np.vstack(pdist_list)
    p_values_arr = np.hstack(p_val_list)
    if return_dperm:
        return p_values_arr, pdist_arr
    else:
        return p_values_arr
       
def GreenIntCuda_memcheck(A,B):
    def GreenIntDataPreProcess(A, B):
        """Preprocess data for Green's algorithm.
        Args:
            A (array): Samples A.
            B (array): Samples B.
        Returns:
             combined data, size of A, size of B, maximum sum, number of samples,
             and array of Sums.  
        """
        m = A.shape[1]
        n = B.shape[1]
        A_n_samples = A.shape[0]
        B_n_samples = B.shape[0]
    
        assert A_n_samples == B_n_samples, "A and B does not have same experiments!"
        n_samples = A_n_samples

        z = np.concatenate((A,B),axis=1)
        z.sort(1)
        z -= z.min(1, keepdims=True)

        S = z[:, m:].sum(1, keepdims=True)
    
        return z.ravel(), m, n, S.max(), n_samples, S.ravel()

    aDim, bDim = A.ndim,B.ndim
    assert aDim == bDim, "A and B does not have same dimensions!"
    
    if aDim == 1:
        A = A.reshape(1,-1)
        B = B.reshape(1,-1)
    a_n, b_n = A.shape[0], B.shape[0]
    assert a_n == b_n, "A and B does not have the same amount of experiments!"

    z, m, n, Smax, n_samples, S =  GreenIntDataPreProcess(A,B)

    height = m + 1
    width = Smax + 1
    z_height = m+n

    int_bits = 4
    double_bits = 8

    memory_bits = z_height * n_samples * int_bits + 2 * width * height * n_samples * double_bits
    memory_MIB = round(memory_bits / 1000000 * 0.953674,2)

    gpu_mem = gpu_available_mem()

    diff = gpu_mem - memory_MIB
    diff = round(diff,2)

    if diff >0:
        print("The data requires {}Mib, and the GPU has {}Mib available, so there are {}Mib left after data allocation.".format(memory_MIB, gpu_mem, diff))
    else:
        print("Warning: The data requires {}Mib, and the GPU has {}Mib available, so there is {}Mib too little memory. Consider dividing the data into batches.".format(memory_MIB, gpu_mem, abs(diff)))


def GreenIntCuda(A,B, return_dperm=False, batch_size=None):
    """Calculate permutation distribution and p-values for integer samples.
        Args:
            A (array): Contining experiments(row) with samples(column).
            B(array): Contining experiments(row) with samples(column).
            num_bin(int): Number of bins to divide samples in.
            return_dperm(boolean): Return permutation distribution.
            batch_size(int): Divide experiments in batches.
        Returns:
            Array with p-values and permutation distribution(optional).
    """
    def GreenPvalInt(NNN, n_samples, S, A,B, z, midP=False):
        """Calculate p-value for each sub-array
        Args:
            pdist (array): Permutation distribution
            n_samples (int): The total number of samples.
            S(int): Sum up to :m.
            A(array): The Values from sample A.
        Returns:
            p-values
        """
        P = np.zeros(n_samples)
        dperm = list()
        for i, (a, b, NN) in enumerate(zip(A, B, NNN)):

            if len(a)>len(b):
                score = (b - np.min(z[i])).sum()
            else:
                score = (a - np.min(z|i)).sum()
            
            if midP:
                one_side = NN[score] / 2 
            else:
                one_side = NN[score]

            if score < S[i]:
                one_side += min(np.sum(NN[score + 1 :]), np.sum(NN[:score]))    
            p = one_side / float(np.sum(NN))
            
            P[i] = 2 * min( p, (1-p))
            
            ''' a = a - a.min()
            p = np.sum(pmf[int(sum(a)) : (int(S[i]) + 1)])
            P[i] = 2 * min( p, (1-p))    '''     
        return P

    def GreenIntDataPreProcess(A, B):
        """Preprocess data for Green's algorithm.
        Args:
            A (array): Samples A.
            B (array): Samples B.
        Returns:
             combined data, size of A, size of B, maximum sum, number of samples,
             and array of Sums.  
        """
        m = A.shape[1]
        n = B.shape[1]
        A_n_samples = A.shape[0]
        B_n_samples = B.shape[0]
    
        assert A_n_samples == B_n_samples, "A and B does not have same experiments!"
        n_samples = A_n_samples

        z = np.concatenate((A,B),axis=1)
        z.sort(1)
        z -= z.min(1, keepdims=True)

        K = min(m,n)

        S = np.sum(z[:, -K:], axis=1).astype(np.int32)

        return z.ravel(), min(m,n), max(m,n), S.max(), n_samples, S.ravel()
     
    aDim, bDim = A.ndim,B.ndim
    assert aDim == bDim, "A and B does not have same dimensions!"
    
    if aDim == 1:
        A = A.reshape(1,-1)
        B = B.reshape(1,-1)
    a_n, b_n = A.shape[0], B.shape[0]
    assert a_n == b_n, "A and B does not have the same amount of experiments!"


    _, _, _, Smax, _, _ =  GreenIntDataPreProcess(A,B)

    if not batch_size:
        a_batch_size, b_batch_size = A.shape[0], B.shape[0]
    else:
        assert batch_size <= A.shape[0], "Batch-size larger than number of examples."
        a_batch_size, b_batch_size = batch_size, batch_size

    pdist_list = list()
    p_val_list = list()
    for a,b in zip(batch(A,a_batch_size), batch(B,b_batch_size)):
        z, m, n, _, n_samples, S =  GreenIntDataPreProcess(a,b)
    
        z = z.astype(np.uint32)
        S = S.astype(np.int32)
    
        NN = np.array(greenCUDA(z, S, int(m), int(n), int(Smax), int(n_samples))).reshape(n_samples, Smax+1)
        p_values = GreenPvalInt(NN, n_samples, S, a, b, z)

        pdist_list.append(NN)
        p_val_list.append(p_values)

    pdist = np.vstack(pdist_list).ravel()
    p_values = np.hstack(p_val_list).ravel()

    if return_dperm:
        return p_values, pdist
    else:
        return p_values

def CoinShiftInt(A,B, return_dperm=False):
    """Calculate permutation distribution and p-values for integer samples.
        Args:
            A (array): Contining experiments(row) with samples(column).
            B(array): Contining experiments(row) with samples(column).
            num_bin(int): Number of bins to divide samples in.
            return_dperm(boolean): Return permutation distribution.
            batch_size(int): Divide experiments in batches.
        Returns:
            Array with p-values and permutation distribution(optional).
    """
    def getDataCoinShift(A, B):
        """Preprocess data for Green's algorithm.
        Args:
            A (array): Samples A.
            B (array): Samples B.
        Returns:
             combined data, size of A, size of B, maximum sum, number of samples,
             and array of Sums.  
        """
        scores = np.concatenate((A,B)).astype(np.int32)
        n = scores.shape[0]
        m = min(B.shape[0], A.shape[0])

        add = np.min(scores)
        scores = scores - add
        scores.sort()

        K = m
        m_b = sum(scores[-K:])

        score_a = np.ones(n, dtype=np.int32)
        
        im_a = m
        im_b = m_b
        return score_a, scores, im_a, im_b, n, add

    def get_p_coin(NN, A, B,S,add, midP=False):
        """Calculate p-value for each sub-array
        Args:
            pdist (array): Permutation distribution
            s(array): The Values from sample A.
        Returns:
            p-values
        """

        if len(a)>len(b):
            score = (b - add).sum()
        else:
            score = (a - add).sum()
        
        score = score - 1
        
        if midP:
            one_side = NN[score] / 2 
        else:
            one_side = NN[score]

        if score < S:
            one_side += min(np.sum(NN[score + 1 :]), np.sum(NN[:score]))    
        p = one_side / float(np.sum(NN))
            
        p = 2 * min(p, (1 - p))
        
        return p

    aDim, bDim = A.ndim,B.ndim
    assert aDim == bDim, "A and B does not have same dimensions!"
    
    if aDim == 1:
        A = A.reshape(1,-1)
        B = B.reshape(1,-1)
    a_n, b_n = A.shape[0], B.shape[0]
    assert a_n == b_n, "A and B does not have the same amount of experiments!"

    pdist_list = list()
    p_val_list = list()
    for a, b in zip(A, B):

        score_a, scores, im_a, im_b, n, add = getDataCoinShift(a, b)

        NN = np.array(coinShift(score_a, scores, im_a, im_b, n))
        p_val = get_p_coin(NN, a, b, im_b, add)
        
        pdist_list.append(NN)
        p_val_list.append(p_val)

    pdist = np.vstack(pdist_list)
    p_values = np.hstack(p_val_list)
    if return_dperm:
        return p_values, pdist
    else:
        return p_values

def GreenInt(A,B, return_dperm=False):
    """Calculate permutation distribution and p-values for integer samples.
        Args:
            A (array): Contining experiments(row) with samples(column).
            B(array): Contining experiments(row) with samples(column).
            num_bin(int): Number of bins to divide samples in.
            return_dperm(boolean): Return permutation distribution.
            batch_size(int): Divide experiments in batches.
        Returns:
            Array with p-values and permutation distribution(optional).
    """
    def getDataGreen(A,B):
        """Preprocess data for Green's algorithm.
        Args:
            A (array): Samples A.
            B (array): Samples B.
        Returns:
            combined data, size of A, size of B, maximum sum, number of samples,
            and array of Sums.
        """
        x = A
        y = B
        m = x.shape[0]
        n = y.shape[0]
    
        z = np.concatenate((A,B))
        z.sort()
        z = z - min(z)
        K = min(m,n)

        S = np.sum(z[-K:]).astype(np.int32)
        return z, int(min(m, n)), int(max(m, n)), int(S)


    def get_p(NN, a, b, S, z, midP=False):
        """Calculate p-value for each sub-array
        Args:
            pdist (array): Permutation distribution
            a(array): The Values from sample A.
        Returns:
            p-values
        """

        if len(a)>len(b):
            score = (b - np.min(z)).sum()
        else:
            score = (a - np.min(z)).sum()
            
        if midP:
            one_side = NN[score] / 2 
        else:
            one_side = NN[score]
        if score < S:
            one_side += min(np.sum(NN[score + 1 :]), np.sum(NN[:score]))    
        p = one_side / float(np.sum(NN))
            
        p = 2 * min( p, (1-p))
        return p

    aDim, bDim = A.ndim,B.ndim
    assert aDim == bDim, "A and B does not have same dimensions!"
    
    if aDim == 1:
        A = A.reshape(1,-1)
        B = B.reshape(1,-1)
    a_n, b_n = A.shape[0], B.shape[0]
    assert a_n == b_n, "A and B does not have the same amount of experiments!"

    pdist_list = list()
    p_val_list = list()
    for a, b in zip(A, B):

        z, m, n, S = getDataGreen(a, b)

        NN = np.array(Green(z, m, n, S))
        p_val = get_p(NN, a, b, S, z)
        
        pdist_list.append(NN)
        p_val_list.append(p_val)

    pdist = np.vstack(pdist_list).ravel()
    p_values = np.hstack(p_val_list).ravel()
    if return_dperm:
        return p_values, pdist
    else:
        return p_values

def GreenIntMultiThread(A,B, return_dperm=False):
    """Calculate permutation distribution and p-values for integer samples.
        Args:
            A (array): Contining experiments(row) with samples(column).
            B(array): Contining experiments(row) with samples(column).
            num_bin(int): Number of bins to divide samples in.
            return_dperm(boolean): Return permutation distribution.
            batch_size(int): Divide experiments in batches.
        Returns:
            Array with p-values and permutation distribution(optional).
    """
    def getDataGreen(A,B):
        """Preprocess data for Green's algorithm.
        Args:
            A (array): Samples A.
            B (array): Samples B.
        Returns:
            combined data, size of A, size of B, maximum sum, number of samples,
            and array of Sums.
        """
        x = A
        y = B
        m = x.shape[0]
        n = y.shape[0]
    
        z = np.concatenate((A,B))
        z.sort()
        z = z - min(z)
        S = z[m:].sum()

        K = min(m,n)

        S = np.sum(z[-K:]).astype(np.int32)
        return z, int(min(m, n)), int(max(m, n)), int(S)

    def get_p(NN, a, b, S, z, midP=False):
        """Calculate p-value for each sub-array
        Args:
            pdist (array): Permutation distribution
            a(array): The Values from sample A.
        Returns:
            p-values
        """
        if len(a)>len(b):
            score = (b- np.min(z)).sum()
        else:
            score = (a - np.min(z)).sum()
            
        if midP:
            one_side = NN[score] / 2 
        else:
            one_side = NN[score]
        if score < S:
            one_side += min(np.sum(NN[score + 1 :]), np.sum(NN[:score]))    
        p = one_side / float(np.sum(NN))
            
        p = 2 * min( p, (1-p))
        return p

    aDim, bDim = A.ndim,B.ndim
    assert aDim == bDim, "A and B does not have same dimensions!"
    
    if aDim == 1:
        A = A.reshape(1,-1)
        B = B.reshape(1,-1)
    a_n, b_n = A.shape[0], B.shape[0]
    assert a_n == b_n, "A and B does not have the same amount of experiments!"

    pdist_list = list()
    p_val_list = list()
    for a, b in zip(A, B):

        z, m, n, S = getDataGreen(a, b)

        NN = np.array(GreenOpenMP(z, m, n, S))
        p_val = get_p(NN, a, b, S, z)
        
        pdist_list.append(NN)
        p_val_list.append(p_val)

    pdist = np.vstack(pdist_list).ravel()
    p_values = np.hstack(p_val_list).ravel()
    if return_dperm:
        return p_values, pdist
    else:
        return p_values



def GreenFloat(A,B, num_bin, return_dperm=False):
    """Calculate permutation distribution and p-values for integer samples.
        Args:
            A (array): Contining experiments(row) with samples(column).
            B(array): Contining experiments(row) with samples(column).
            num_bin(int): Number of bins to divide samples in.
            return_dperm(boolean): Return permutation distribution.
            batch_size(int): Divide experiments in batches.
        Returns:
            Array with p-values and permutation distribution(optional).
    """

    def digitized_score(X, bins):
        """Digitize the values for each sample.
        Args:
            X (array): Concatenated sample from original samples A and B.
            bins(int): The number of bins to divide the sample values.
        Returns:
            digitized array
        """
        digitized = np.zeros(X.shape,dtype=np.int32)
        for i, (x,b) in enumerate(zip(X,bins)):
            digitized[i,:] = np.digitize(x, b).astype(np.int32) - 1
        return digitized
    
    def GreenFloatDataPreProcess(A, B, num_bin):
        """Preprocess data for Green's algorithm.
        Args:
            A (array): Samples A.
            B (array): Samples B.
            num_bin (int): Number of bins for digitization.
        Returns:
             digitized data, size of A, size of B, maximum sum, number of samples,
             bins for data, and array of Sums.  
        """
        
        m = A.shape[1]
        n = B.shape[1]
        n_samples = A.shape[0]

        X = np.concatenate([A,B],axis=1)
        X.sort()

        bins = np.asarray([np.linspace(np.min(x), np.max(x), int(num_bin)) for x in X])

        digitized = digitized_score(X, bins)

        K = min(m,n)

        S = np.sum(digitized[:, -K:], axis=1).astype(np.int32)
        return digitized.ravel(), min(m,n), max(m,n), S.max(), n_samples, bins, S

    def pValFloat(NN, n_samples, S, A , B, bins, midP=False):
        """Calculate p-value for each sub-array
        Args:
            pdist (array): Permutation distribution
            n_samples (int): The total number of samples.
            S(int): Sum up to :m.
            A(array): The Values from sample A.
            bins(array): Bins for digitization.
            midP(boolean): Calculate mid p-value.
        Returns:
            p-values
        """

        P = np.zeros(n_samples)
        dperm = list()
        a,b,_bin,s = A[0],B[0],bins[0], S[0]

        if len(a)>len(b):
            score = sum(np.digitize(b, _bin).astype(np.int32) - 1)
        else:
            score = sum(np.digitize(a, _bin).astype(np.int32) - 1)

        if midP:
            one_side = NN[score] / 2 
        else:
            one_side = NN[score]

        if score<S:    
            one_side += min(np.sum(NN[score + 1 :]), np.sum(NN[:score]))    
            p = one_side / float(np.sum(NN))
            
        P[0] = 2 * min( p, (1-p))
                
        return P


    aDim, bDim = A.ndim,B.ndim
    assert aDim == bDim, "A and B does not have same dimensions!"
    
    if aDim == 1:
        A = A.reshape(1,-1)
        B = B.reshape(1,-1)
    a_n, b_n = A.shape[0], B.shape[0]
    assert a_n == b_n, "A and B does not have the same amount of experiments!"

    pdist_list = list()
    p_val_list = list()
    for a, b in zip(A, B):

        z, m, n, _, n_samples, bins, S =  GreenFloatDataPreProcess(a[np.newaxis,:],b[np.newaxis,:], num_bin)
    
        z = z.astype(np.uint32)
        S = S.astype(np.int32)
    
        NN = np.array(Green(z, m, n, S))

        p_val = pValFloat(NN, n_samples, S, a[np.newaxis,:] ,b[np.newaxis,:], bins)
        
        pdist_list.append(NN)
        p_val_list.append(p_val)

    pdist = np.vstack(pdist_list).ravel()
    p_values = np.hstack(p_val_list).ravel()
    if return_dperm:
        return p_values, pdist
    else:
        return p_values

def GreenFloatMultiThread(A,B, num_bin, return_dperm=False):
    """Calculate permutation distribution and p-values for integer samples.
        Args:
            A (array): Contining experiments(row) with samples(column).
            B(array): Contining experiments(row) with samples(column).
            num_bin(int): Number of bins to divide samples in.
            return_dperm(boolean): Return permutation distribution.
            batch_size(int): Divide experiments in batches.
        Returns:
            Array with p-values and permutation distribution(optional).
    """

    def digitized_score(X, bins):
        """Digitize the values for each sample.
        Args:
            X (array): Concatenated sample from original samples A and B.
            bins(int): The number of bins to divide the sample values.
        Returns:
            digitized array
        """
        digitized = np.zeros(X.shape,dtype=np.int32)
        for i, (x,b) in enumerate(zip(X,bins)):
            digitized[i,:] = np.digitize(x, b).astype(np.int32) - 1
        return digitized
    
    def GreenFloatDataPreProcess(A, B, num_bin):
        """Preprocess data for Green's algorithm.
        Args:
            A (array): Samples A.
            B (array): Samples B.
            num_bin (int): Number of bins for digitization.
        Returns:
             digitized data, size of A, size of B, maximum sum, number of samples,
             bins for data, and array of Sums.  
        """
        
        m = A.shape[1]
        n = B.shape[1]
        n_samples = A.shape[0]

        X = np.concatenate([A,B],axis=1)
        X.sort()

        bins = np.asarray([np.linspace(np.min(x), np.max(x), int(num_bin)) for x in X])

        digitized = digitized_score(X, bins)

        K = min(m,n)

        S = np.sum(digitized[:, -K:], axis=1).astype(np.int32)
        return digitized.ravel(), min(m,n), max(m,n), S.max(), n_samples, bins, S

    def pValFloat(NN, n_samples, S, A , B, bins, midP=False):
        """Calculate p-value for each sub-array
        Args:
            pdist (array): Permutation distribution
            n_samples (int): The total number of samples.
            S(int): Sum up to :m.
            A(array): The Values from sample A.
            bins(array): Bins for digitization.
            midP(boolean): Calculate mid p-value.
        Returns:
            p-values
        """

        P = np.zeros(n_samples)
        dperm = list()
        a,b,_bin,s = A[0],B[0],bins[0], S[0]

        if len(a)>len(b):
            score = sum(np.digitize(b, _bin).astype(np.int32) - 1)
        else:
            score = sum(np.digitize(a, _bin).astype(np.int32) - 1)

        if midP:
            one_side = NN[score] / 2 
        else:
            one_side = NN[score]

        if score<S:    
            one_side += min(np.sum(NN[score + 1 :]), np.sum(NN[:score]))    
            p = one_side / float(np.sum(NN))
            
        P[0] = 2 * min( p, (1-p))
                
        return P


    aDim, bDim = A.ndim,B.ndim
    assert aDim == bDim, "A and B does not have same dimensions!"
    
    if aDim == 1:
        A = A.reshape(1,-1)
        B = B.reshape(1,-1)
    a_n, b_n = A.shape[0], B.shape[0]
    assert a_n == b_n, "A and B does not have the same amount of experiments!"

    pdist_list = list()
    p_val_list = list()
    for a, b in zip(A, B):

        z, m, n, _, n_samples, bins, S =  GreenFloatDataPreProcess(a[np.newaxis,:],b[np.newaxis,:], num_bin)
    
        z = z.astype(np.uint32)
        S = S.astype(np.int32)
    
        NN = np.array(GreenOpenMP(z, m, n, S))

        p_val = pValFloat(NN, n_samples, S, a[np.newaxis,:] ,b[np.newaxis,:], bins)
        
        pdist_list.append(NN)
        p_val_list.append(p_val)

    pdist = np.vstack(pdist_list)
    p_values = np.hstack(p_val_list)
    if return_dperm:
        return p_values, pdist
    else:
        return p_values

def CoinShiftFloat(A,B, num_bin, return_dperm=False):
    """Calculate permutation distribution and p-values for integer samples.
        Args:
            A (array): Contining experiments(row) with samples(column).
            B(array): Contining experiments(row) with samples(column).
            num_bin(int): Number of bins to divide samples in.
            return_dperm(boolean): Return permutation distribution.
        Returns:
            Array with p-values and permutation distribution(optional).
    """

    def digitized_score(X, bins):
        """Digitize the values for each sample.
        Args:
            X (array): Concatenated sample from original samples A and B.
            bins(int): The number of bins to divide the sample values.
        Returns:
            digitized array
        """
        digitized = np.zeros(X.shape,dtype=np.int32)
        for i, (x,b) in enumerate(zip(X,bins)):
            digitized[i,:] = np.digitize(x, b).astype(np.int32) - 1
        return digitized


    def getDataCoinShift(A, B, num_bin):
      
        scores = np.concatenate([A,B],axis=1)
        

        bins = np.asarray([np.linspace(np.min(x), np.max(x), int(num_bin)) for x in scores])
        scores = digitized_score(scores, bins).ravel()


        n = scores.shape[0]
        m = min(B.shape[1], A.shape[1])

        add = np.min(scores)
        scores = scores - add
        scores.sort()

        K = m
        
        m_b = sum(scores[-K:])

        score_a = np.ones(n, dtype=np.int32)
        
        im_a = m
        im_b = m_b
        return score_a.ravel(), scores.ravel(), im_a, im_b, n, bins, add

    def get_p_coin(NN, A, B, bins,S, midP=False):
        """Calculate p-value for each sub-array
        Args:
            pdist (array): Permutation distribution
            A(array): The Values from sample A.
            bins(array): bins in which A is divided.
        Returns:
            p-values
        """

        
        a,b,_bin= A[0],B[0],bins[0]

        if len(a)>len(b):
            score = sum(np.digitize(b, _bin).astype(np.int32) - 1)
        else:
            score = sum(np.digitize(a, _bin).astype(np.int32) - 1)
        score = score - 1
        if midP:
            one_side = NN[score] / 2 
        else:
            one_side = NN[score]

        if score<S:    
            one_side += min(np.sum(NN[score + 1 :]), np.sum(NN[:score]))    
            p = one_side / float(np.sum(NN))
            
        p = 2 * min( p, (1-p))
                
        return p

    aDim, bDim = A.ndim,B.ndim
    assert aDim == bDim, "A and B does not have same dimensions!"
    
    if aDim == 1:
        A = A.reshape(1,-1)
        B = B.reshape(1,-1)
    a_n, b_n = A.shape[0], B.shape[0]
    assert a_n == b_n, "A and B does not have the same amount of experiments!"

    pdist_list = list()
    p_val_list = list()
    for a, b in zip(A, B):
        score_a, scores, im_a, im_b, n, bins, add = getDataCoinShift(a[np.newaxis, :], b[np.newaxis, :], num_bin)
        
        NN = np.array(coinShift(score_a, scores, im_a, im_b, n))

        p_val = get_p_coin(NN, a[np.newaxis,:], b[np.newaxis,:], bins, im_b)
        
        pdist_list.append(NN)
        p_val_list.append(p_val)

    pdist = np.vstack(pdist_list)
    p_values = np.hstack(p_val_list)
    if return_dperm:
        return p_values, pdist
    else:
        return p_values
