import numpy as np
from numba import cuda
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind
import numpy.random as npr


def bootstrap(invec):
    """ Function for generating bootstrap sampled versions of a vector """
    idx = npr.randint(0, len(invec), len(invec))
    return [invec[i] for i in idx]

def estimatePi0(p, numBoot=100, numLambda=100, maxLambda=0.95):
    """
    Function for estimaring pi_0, i.e. the prior null probability
    for p values.
    Args:
        p (list(float)): The list of p values for which pi_0 should be estimated
        numBoot (int): The number of bootstrap rounds that should be made.
        numLambda (int): The number of lambda tresholds that should be evaluated.
        maxLambda (float): The upper bond of the range of lambda treshold.
    Returns:
        pi_0, a float containing the pi_0 estimate.
    """
    p.sort()
    n=len(p)
    lambdas=np.linspace(maxLambda/numLambda,maxLambda,numLambda)
    Wls=np.array([n-np.argmax(p>=l) for l in lambdas])
    pi0s=np.array([Wls[i] / (n * (1 - lambdas[i])) for i in range(numLambda)])
    minPi0=np.min(pi0s)
    mse = np.zeros(numLambda)
    for boot in range(numBoot):
        pBoot = bootstrap(p)
        pBoot.sort()
        WlsBoot =np.array([n-np.argmax(pBoot>=l) for l in lambdas])
        pi0sBoot =np.array([WlsBoot[i] / (n *(1 - lambdas[i])) for i in range(numLambda)])
        mse = mse + np.square(pi0sBoot-minPi0)
    minIx = np.argmin(mse)
    return pi0s[minIx]

def qvalues(pvalues,p_col = "p", q_col ="q", pi0 = 1.0):
    """
    Function for estimaring q values.
    Args:
        pvalues (DataFrame): A DataFrame that contain at least one column with pvalues.
        p_col (str): The name of the column that contain pvalues.
        q_col (str): The name of the column that that shall contain the estimated
                    q-values. The column will be created if not already existing.
        pi0 (float): The prior probability of the null hypothesis. If set to None, this is estimated from data.
                    Defaults to 1.0
    Returns:
        The modified DataFrame.
    """
    m = pvalues.shape[0] # The number of p-values
    pvalues.sort_values(p_col,inplace=True) # sort the pvalues in acending order
    if pi0 is None:
        pi0 = estimatePi0(list(pvalues[p_col].values))

    # calculate a FDR(t) as in Storey & Tibshirani
    num_p = 0.0
    for ix in pvalues.index:
        num_p += 1.0
        fdr = pi0*pvalues.loc[ix,p_col]*m/num_p
        pvalues.loc[ix,q_col] = fdr

    # calculate a q(p) as the minimal FDR(t)
    old_q=1.0
    for ix in reversed(list(pvalues.index)):
        q = min(old_q,pvalues.loc[ix,q_col])
        old_q = q
        pvalues.loc[ix,q_col] = q
    return pvalues

def MW(A, B):
    p_mw = list()
    for a,b in zip(A, B):
        p_mw.append(mannwhitneyu(a,b, alternative="less")[1])
    return p_mw

def ttests(A,B):
    p_t = list()
    for x, y in zip(A, B):
        t, p = ttest_ind(y, x)
        p = p/2
        if t<0:
            p = 1-p
        p_t.append(p)
    return p_t

def table(val, S):
    table = np.zeros(S + 1)
    for e in val:
        table[e] +=1
    return table

def significance_of_mean(a,b,num_bin = None, data_type=np.float64): #

    # discretize
    ab = np.sort(np.concatenate((a, b), axis=None))[::-1] #sort descending (biggest first)
    
    if not num_bin:
        num_bin = np.ceil(max(ab)) - np.floor(min(ab)) + 1
    
    

    bins = np.linspace(min(ab), max(ab), num_bin)    
    #bins = np.linspace(np.floor(min(ab)), np.ceil(max(ab)), num_bin)
    digitized = np.digitize(ab, bins)
    
    if len(a)>len(b):
        score = sum(np.digitize(b,bins))
    else:
        score = sum(np.digitize(a,bins))
    K = min(len(a),len(b))
    S = np.sum(digitized[0:K])+1 #hoehle: This is the maximum sum we can get! why +1? A: S is the number of bins, lowest score=0, highest score=sum(digitiyzed) make a total of sum(digitiyzed)+1 bins
    L = len(ab) #length of m+n
    NN = score_distribution_numpy_full(digitized, K, S, L, data_type)
    # Total number of configurations
    total = np.sum(NN)
    # Just the one sided version of the test, i.e. alternative="greater"
    if len(a)>len(b):
        more_extreme = np.sum(NN[:score])
    else:
        more_extreme = np.sum(NN[score:])
    p = more_extreme/total
    return p, total, more_extreme

def score_distribution_numpy(digitized,K,S,L, data_type=np.float64):
    # N(s,l) number of ways to reach a sum of s using k of the l first readouts
    # Calculated by iterating over the
    Nold = np.zeros((S,L), dtype=data_type)
    # Initiating (i.e. k=0 case)
    for l in range(L):
        d=digitized[l]
        Nold[d,l] = data_type(1)
    # Do each of the other picks
    for k in range(1,K):
        Nnew = np.zeros((S,L), dtype=data_type)
        C = np.zeros((S,1), dtype=data_type)
        for l in range(1,L): #hoehle: why 1:L and not 0:L?
            d = digitized[l]
            C +=  Nold[:,l-1:l]
            Nnew[d:S,l] = C[0:S-d,0]
        Nold = Nnew
    NN = np.sum(Nold,axis=1)
    return NN

def score_distribution_numpy_full(digitized,K,S,L, data_type=np.float64):
    # N(s,l) number of ways to reach a sum of s using k of the l first readouts
    # Calculated by iterating over the
    N = np.zeros((S,L,K), dtype=data_type)
    # Initiating (i.e. k=0 case)
    for l in range(L):
        d=digitized[l]
        N[d,l,0] = data_type(1)
    # Do each of the other picks
    for k in range(1,K):
        C = np.zeros((S,1), dtype=data_type)
        for l in range(1,L): #hoehle: why 1:L and not 0:L?
            d = digitized[l]
            C +=  N[:,l-1:l,k-1]
            N[d:S,l,k] = C[0:S-d,0]
    NN = np.sum(N[:,:,-1],axis=1)
    return NN

def significance_of_mean_cuda(a,b,num_bin = None, dtype_v=np.uint64, dtype_A=np.float64):
    x = a
    y = b
    
    m = len(x)
    n = len(y)
    
    z = x + y
    z.sort()
    
    if not num_bin:
        num_bin = np.ceil(max(z)) - np.floor(min(z)) + 1
    bins = np.linspace(min(z), max(z), num_bin)    
    #bins = np.linspace(np.floor(min(z)), np.ceil(max(z)), num_bin)
    

    digitized = np.digitize(z, bins).astype(dtype_v) - 1   
    S = sum(digitized[m:])
    

    N = exact_perm_numba_shift(int(m), int(n), int(S), digitized, dtype_v, dtype_A)
    
    NN = N[:,:,-1]    
    pmf = NN[:,m+n-1] / np.sum(NN[:,m+n-1])
    x_ = np.digitize(x, bins).astype(dtype_v) - 1
    
    p_value = np.sum(pmf[int(sum(x_)):(int(S)+1)])
    
    

    return p_value

@cuda.jit("(f8[:,:,:], u8, u8)")
def get_perumations(X, k_, z_):
    n = X.shape[0]
    m = X.shape[2]
    
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if j >= 2 and j < m + 1 and  i < n :
        if i >= z_:
            X[i, k_- 1, j-1] = X[i - int(z_), k_-2 ,j-2] +  X[i,k_-2,j-1]
        else:
            X[i, k_-1, j-1] = X[i,k_-2, j-1]

def exact_perm_numba_shift(m, n, S, z, dtype_v, dtype_A):
    N_cuda = np.zeros([S + 1, m + n, m], dtype_A)
    NN, NM = N_cuda[:,0,:].shape

    for k in range(1,(m+n)+1):
        N_cuda[:,k -1,0] = table(z[0:k],S )
    
    #A = N_cuda.copy()
    A = N_cuda
    
    blockdim = (256, 3)
    griddim = (int(np.ceil((NN )/ blockdim[0])), int(np.ceil(NM/blockdim[1] + 1)))
    A = np.ascontiguousarray(A) 
    stream = cuda.stream()
    dA = cuda.to_device(A, stream)
    
    for k in range(2, (m+n)+1):
        dk = dtype_v(k) 
        dz = dtype_v(z[k - 1])
        get_perumations[griddim,blockdim, stream](dA, dk, dz)
    dA.to_host(stream)
    stream.synchronize()
    return A

def getNumerator(m, n, S, z, dtype):
    N = np.zeros([S + 1, m], dtype)
    N_old = N.copy()
    
    for i in range(1,(m+n)+1):
        for j in range(1, m +1):
            for s in range(S+1):
                if i < j:
                    N[s,j-1] = 0
                elif j == 1 and z[i-1] == s:
                    N[s,j-1] = N_old[s,j-1] + 1
                elif j == 1 and z[i-1] != s:
                    N[s,j-1] = N_old[s,j-1]
                elif j > 1 and z[i-1] <= s:
                    N[s,j-1] = N_old[s - z[i -1], j-2] + N_old[s,j-1]
                elif j > 1 and z[i-1] > s:
                    N[s,j-1] = N_old[s,j-1]
    
        N_old = N.copy()
        
    return N_old[:, -1]

def pValue(Numerator, sample):
    return np.round((Numerator / np.sum(Numerator))[sum(sample):].sum(), 3)

def getdf(P, num_examples, test=None):
    P.sort()
    p_arr = np.array(P)
    offset = 1.0/float(num_examples)
    ideal_arr = np.linspace(offset,1.0-offset,num_examples)
    if test:
        Pdf = pd.DataFrame({'Observed p-value':p_arr,'Theoretical p-value':ideal_arr, "Test":[test]*ideal_arr.shape[0]})
    return Pdf

def my_scatter_plot(df,save_name):
    sns.set(style="white")
    sns.set_context("talk")
    low = min(df["Theoretical p-value"])
    hi = max(df["Theoretical p-value"])
    f, ax = plt.subplots(figsize=(7, 7))
    ax.set(xscale="log", yscale="log")
    g=sns.regplot(x='Theoretical p-value', y ='Observed p-value', data=df,  ax=ax, fit_reg=False, scatter_kws={"s": 5})
    g.plot([low,hi], [low,hi], 'k-', linewidth=.5)
    sns.despine()
    f.tight_layout()
    f.savefig(save_name)
