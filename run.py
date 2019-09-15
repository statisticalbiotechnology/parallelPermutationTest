from significance_of_mean_cuda import significance_of_mean_cuda
from utils import significance_of_mean
import numpy as np
import time
import multiprocessing
import concurrent.futures as cf
import matplotlib.pyplot as plt

np.random.seed(1)
A = np.asarray([np.random.beta(2.0,5.0,3) for _ in range(2)])
B = np.asarray([np.random.beta(2.0, 5.0, 3) for _ in range(2)])

#A = np.asarray([np.random.randint(0,2000,200) for _ in range(1)])
#B = np.asarray([np.random.randint(0, 2000, 200) for _ in range(1)])

#A = np.asarray([[0,3,0]])
#B = np.asarray([[1, 2, 5]])


args_list = list()
x = [0,3,0]
y = [1,2,5]
m = len(x)
n = len(y)
z = x + y;z.sort()
S = sum(z[m:])
dtype = np.uint16
args_list.append([m, n, S, z, dtype])

print(z)


#num_bin = np.ceil(max(z)) - np.floor(min(z)) + 1
num_bin = 200


#bins = 200

start = time.time()
SGM = significance_of_mean_cuda(num_bin, dtype_v=np.uint16,dtype_A=np.uint32)
PC1 = SGM.run(A,B)
end = time.time()
print(end - start)

print(PC1)

def p_value_calc(args):
    a,b = args
    p=significance_of_mean(a,b, num_bin)[0]
    return p

def calibration_series_generator(A,B):
    num_tests = A.shape[0]
    for i in range(num_tests):
        a_sample = A[i].tolist()
        b_sample = B[i].tolist()
        yield ([a_sample,b_sample])

def calibration_test(A,B):
    with cf.ProcessPoolExecutor(max_workers=1) as pool:
        p_list = list(pool.map(p_value_calc, calibration_series_generator(A,B)))
    return p_list

#start = time.time()
#P = list()
#P = calibration_test(A,B)
#end = time.time()
#print(end - start)

#print(np.allclose(PC1, P))






start = time.time()
SGM = significance_of_mean_cuda(num_bin,dtype_v=np.uint16,dtype_A=np.uint32, new_version=True)
PC2 = SGM.run(A,B)
end = time.time()
print(end - start)

print(np.allclose(PC2, PC1))


def exact_perm_shift2(m, n, S, z, dtype):
    init = np.zeros([S + 1, m + n], dtype)
    for k in range(1,(m+n)+1):
        init[:, k - 1] = table(z[0:k], S)
    
    
    N = np.zeros([S + 1, m], dtype)
    N_old = N.copy()
    for k in range(1,(m+n)+1):
        if k==1:
            N[:,0] = init[:,0]
        else:
            for j in range(1, m +1):
                if j == 1:
                    N[:,j-1] = init[:,k-1]
                else:
                    #N[:,k-1] = shift(N_old[:,k-2], z[k -1]) + N[:,k-2]
                    N[:,j-1] = shift(N_old[:,j-2], z[k -1]) + N_old[:,j-1]
        #print(N)
        N_old = N
        N = np.zeros([S + 1, m], dtype)
    return N_old[:,-1]