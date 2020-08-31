# ParallelizedShiftExactPermTest
Parallelized version of the Shift-method for exact permutations

## Getting Started

### Prerequisites

python3.6 has been used to test out the repository. The installation uses anaconda, but it is not necessary.

A GPU that uses CUDA is necessary.

Only Python-specific requirements are Numba, Numpy, and Matplotlib.

## Run time performance increase.

The GPU allows for substantial speed up for larger matrices.

```
listsizes = [20,60,120,160]
plain_shift = list()
gpu_shift = list()
bins = 200
for size in listsizes:
    np.random.seed(1)
    A = np.asarray([np.random.beta(2.0,5.0,size) for _ in range(5)])
    B = np.asarray([np.random.beta(2.0,5.0,size) for _ in range(5)])
    start = time.time()
    P = calibration_test(A,B)
    end = time.time()
    plain_shift.append(round(end - start,3))
    print("Plain")
    print(round(end - start,3))
    
    start = time.time()
    SGM = significance_of_mean_cuda(bins,dtype_v=np.uint16,dtype_A=np.float64)
    SGM.run(A,B)
    P = SGM.get_p_values()
    end = time.time()
    t_gpu = end - start
    gpu_shift.append(t_gpu)
    print("GPU: ", t_gpu)
    
    print(np.allclose(PC,P))
```


![alt text](/figures/normal_N.png)

## Authors

* **Markus Ekvall, Lukas Käll and Micheal Höhle** 

## Acknowledgments

* Pagano and Tritchler(1983), and Zimmerman (1985) for unparalleled version of the shift-method.

