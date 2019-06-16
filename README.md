# ParallelizedShiftExactPermTest
Parallelized version of the Shift-method for exact permutations

## Getting Started

### Prerequisites

python3.6 has been used to test out the repository. The installation uses anaconda, but it is not necessary.

A GPU that uses CUDA is necessary.


### Installing

Use this following command to install all required packages.

```
bash install.sh
```

You need to set correct path to cuda

```
export PATH=/usr/local/cuda-X.X/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-X.X/lib64\{LD_LIBRARY_PATH:+:${$LIBRARY_PATH}}
```

Here X.X is you cuda version e.g., 9.2, 10.0 etc. The code have only been tested for cuda9.2 and cuda10.0. 

## Run time performance increase.

The GPU allows for substantial speed up for larger matrices. It speeds up both thicker and smaller matrices (see figure one below).


![alt text](/figures/comparison.png)

## How the parallization works.

To obtain the sought permutations N(m,m+n) an matrix can be build up where each N(j,k) is dependent on the previous cacluated permuations N(j-1,k-1) and N(j,k-1). The matrix looks as follows

![alt text](/figures/whole_array.png)

To see the parallization pattern only the 2D array of j and k has to be considered, see figure below.

![alt text](/figures/vector_relatiness.png)

From the figure one relized for a fixed k, each j element is independent of eachother. This makes it possible to paralelly calucate each j element. Futhermore, this can is done for each consecuative k in the matrix.

![alt text](/figures/how_they_are parallized.png)

This make possible to just loop over k to to fnally obtain the sought N(m,m+n).

![alt text](/figures/extraxt_the_wanted_array.png)