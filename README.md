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

## Authors

* **Markus Ekvall** 

## Acknowledgments

* Lukas Käll and Micheal Höhle
* Pagano and Tritchler(1983), and Zimmerman (1985) for unparalleled version of the shift-method.

