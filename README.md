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
    SGM = significance_of_mean_cuda(bins,dtype_v=np.uint32,dtype_A=np.float64)
    PC = SGM.run(A,B)
    end = time.time()
    end = time.time()
    gpu_shift.append(round(end - start,3))
    print("GPU")
    print(round(end - start,3))
    
    print(np.allclose(PC,P))
```


![alt text](/figures/comparison.png)

Please check the notebook for the code of the algorithm's and to see how the experiment was set-up.
## How the parallization works.

It's highly recommended to read https://github.com/hoehleatsu/permtest/blob/master/computation.pdf to be able to follow the explanation of the parallelization implementation.

To obtain the sought of permutations N(m,m+n), a matrix can be created and then recursevely fill the entries for each N(j,k) using the relation(order of don't matter):

<a href="https://www.codecogs.com/eqnedit.php?latex=N(j,k)=&space;N(j-1,k-1)$\bigoplus$&space;z_{k}&space;&plus;&space;N(j,k-1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N(j,k)=&space;N(j-1,k-1)$\bigoplus$&space;z_{k}&space;&plus;&space;N(j,k-1)" title="N(j,k)= N(j-1,k-1)$\bigoplus$ z_{k} + N(j,k-1)" /></a>

The matrix looks as follows

![alt text](/figures/whole_array1.png)

Only the 2D array of j and k is sufficient to see the parallelization-pattern, see figure below.

![alt text](/figures/vector_relatiness.png)

From the figure above, one realizes that for a fixed k, each j-element in that kth row is independent of each other. Hence, it's possible to calculate each j element in that row parallelly. Furthermore, each k row is consecutively calculated in this manner.


![alt text](/figures/how_they_are_parallized.png)

The matrix is calculated in a loop over k to finally obtain the sought of N(m,m+n).

![alt text](/figures/extraxt_the_wanted_array.png)

The algorithm becomes much more memory efficient then a regular one since the whole array do not have to be loaded into working memory directly. It is sufficient to have two sub-array that alternate between A0 and A1 between each loop over k. This memory efficiency makes to calculate a vast amount of samples at once onto the GPU. It works as follows:
A0 is initialized.
1. Calculate A1 from A0.
2. Let A1 be A0, and A0 be A1.
Repeat K times.

![alt text](/figures/A0_A1.png)

The necessary part of the final A1 for p-values calculations will not be affected by this routine, and it should decrease memory from:

<a href="https://www.codecogs.com/eqnedit.php?latex=O(SM(M&plus;N))=O(SM^{2}&plus;SMN)\&space;to&space;\&space;O(2SM)=O(SM)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?O(SM(M&plus;N))=O(SM^{2}&plus;SMN)\&space;to&space;\&space;O(2SM)=O(SM)" title="O(SM(M+N))=O(SM^{2}+SMN)\ to \ O(2SM)=O(SM)" /></a>

## Authors

* **Markus Ekvall** 

## Acknowledgments

* Lukas Käll and Micheal Höhle
* Pagano and Tritchler(1983), and Zimmerman (1985) for unparalleled version of the shift-method.

