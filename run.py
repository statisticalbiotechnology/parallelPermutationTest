from significance_of_mean_cuda import significance_of_mean_cuda
from utils import significance_of_mean
import numpy as np
import time
import multiprocessing
import concurrent.futures as cf
import matplotlib.pyplot as plt

num_examples = 1

N = [50,100, 150, 200, 250]
s = [10, 50, 100, 200]

np.random.seed(42)
A = np.asarray([np.random.normal(0, 1, N[3]) for _ in range(num_examples)])
B = np.asarray([np.random.normal(0, 1, N[3]) for _ in range(num_examples)])
bins = s[0]

def p_value_calc(args):
    a,b, bins = args
    p=significance_of_mean(a,b, bins)[0]
    return p

def calibration_series_generator(A,B, S):
    num_tests = A.shape[0]
    for i in range(num_tests):
        a_sample = A[i].tolist()
        b_sample = B[i].tolist()
        yield ([a_sample,b_sample, S])

def calibration_test(A,B,bins):
    with cf.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()-3) as pool:
        p_list = list(pool.map(p_value_calc, calibration_series_generator(A,B, bins)))
    return p_list
    
start = time.time()
P = calibration_test(A,B,bins)
end = time.time()
print(end - start)



SGM = significance_of_mean_cuda(bins,dtype_v=np.uint16,dtype_A=np.float64)
SGM.run(A, B)
print(P)
print(SGM.get_p_values())
print(np.allclose(SGM.get_p_values(),P))
