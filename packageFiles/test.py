import numpy as np
import time
import pandas as pd


from permutationTest import GreenIntCuda, GreenFloatCuda, getDataGreen, PreProcessoinShift, get_p, get_p_coin, GreenOpenMP, Green, coinShift




np.random.seed(3)
n = 10
m = n
n_samples = 500
A = np.asarray([np.random.randint(0,n,m,dtype=np.int32) for _ in range(n_samples)])
B = np.asarray([np.random.randint(0,n,n,dtype=np.int32) for _ in range(n_samples)])

print(Green(*getDataGreen(A[0], B[0])))

""" 
 
a,b = A, B


x = GreenIntCuda(a,b) """


''' print(checkGPUMemory(int(200), int(7107), int(100 + 1), int(100))) '''


""" SGM = significance_of_mean_cuda(100, dtype_v=np.uint32,dtype_A=np.float64)
SGM.run(a,b)

print(SGM.p_values) """

""" 

NotTNP_Arr = np.load("data/notTNP.npy")
TNPdf = np.load("data/TNPdf.npy")



A,B = TNPdf[0], NotTNP_Arr[0]
A,B = A[np.newaxis,:], B[np.newaxis,:]

start = time.time()
pval, pdist = GreenFloatCuda(A,B,6,return_dperm=True)
end = time.time()
print(end - start)
print("----------------------")

start = time.time()
pval, pdist = GreenFloatCuda(A,B,6,return_dperm=True)
end = time.time()
print(end - start)
print("----------------------")

print(pval)
 """