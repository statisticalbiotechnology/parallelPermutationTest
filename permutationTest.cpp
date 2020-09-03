#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <omp.h>
#include <chrono>
#include <stdlib.h> 
#include <iostream>

#include <string.h>


extern double * greenCuda(int * Z_data_, int * all_S_, int m_, int n_, int S_, int n_samples_);

size_t cuda_mem_avilable();
size_t mem_required(int m,int n,int S,int n_samples);


static PyObject* gpu_available_mem(PyObject* self, PyObject* args) {
/*
 * Function:  Calculate permutation distribution 
 * --------------------
 *
 *  m: Number of samples in A
 *  n: Number of samples in B
 *  S: Maxsum in experiments
 *  n_samples: Number of experiments
 *  z: Combination of samples of A and B
 *  all_S: Sums from each experiments
 * 
 *  returns: Permutation distribution for each experiment
 */

    int mem = (int)ceil((cuda_mem_avilable())/1000000 * 0.953674);
   
    return Py_BuildValue("i", mem);
}

static PyObject* greenCUDA(PyObject* self, PyObject* args) {
/*
 * Function:  Calculate permutation distribution 
 * --------------------
 *
 *  m: Number of samples in A
 *  n: Number of samples in B
 *  S: Maxsum in experiments
 *  n_samples: Number of experiments
 *  z: Combination of samples of A and B
 *  all_S: Sums from each experiments
 * 
 *  returns: Permutation distribution for each experiment
 */


    double * outMul;

    int m;
    int n;
    int S;
    int n_samples;

    PyArrayObject* z;
    PyArrayObject* all_S;
  
    PyArg_ParseTuple(args, "O!O!iiii", &PyArray_Type, &z, &PyArray_Type, &all_S, &m, &n, &S, &n_samples);
  
    outMul = greenCuda((int *) z -> data,(int *) all_S -> data, m,n,S, n_samples);

    if (outMul == NULL) {
        int cuda_mem = (int)ceil((cuda_mem_avilable())/1000000 * 0.953674);
        int mem = (int)ceil((mem_required(m, n, S, n_samples))/1000000 * 0.953674);

        char cuda_mem_str[20]; 
        char mem_str[20]; 

        sprintf(cuda_mem_str, "%d", cuda_mem);
        sprintf(mem_str, "%d", mem);

        if (cuda_mem <=mem) {        
            char str[1000];
            strcpy(str, "MemoryError: Your data requires ");
            strcat(str, mem_str);
            strcat(str, "Mib, but your GPU does only have ");
            strcat(str, cuda_mem_str);
            strcat(str, "Mib available. Consider dividing your data into batches.");

            PyErr_SetString(PyExc_RuntimeError, str);
            return NULL;
        } else {
            PyErr_SetString(PyExc_RuntimeError, "There are not enough threads avaialble for your data. Consider dividing your data into batches.");
            return NULL;
        }
    }

    PyObject* result = PyList_New(0);
    for (int i = 0; i < (S+1)*n_samples; i++){
        PyList_Append(result, PyFloat_FromDouble(outMul[i]));
        }
   
    return result;
}


int min(int num1, int num2);

int min(int num1, int num2) 
{
    return (num1 > num2 ) ? num2 : num1;
}

static PyObject* coinShift(PyObject* self, PyObject* args){
    /* 
    Credit: hnilsson (https://github.com/cran/coin)
     */
  
  

  int s_a = 0;
  int s_b = 0;
  int i;
  int k;
  int j;
  int isb;

  int im_a;
  int im_b;
  int n;

  int sum_a = 0;
  int sum_b = 0;

  double msum = 0.0;


PyArrayObject* score_a;
PyArrayObject* score_b;

PyArg_ParseTuple(args, "O!O!iii", &PyArray_Type, &score_a, &PyArray_Type, &score_b, &im_a, &im_b,&n);


int * iscore_a = (int *)score_a->data;
int * iscore_b = (int *)score_b->data;

  for (i = 0; i < n; i++) {
        sum_a += iscore_a[i];
        sum_b += iscore_b[i];
    }

    

    sum_a = min(sum_a, im_a);
    /* sum_b = min(sum_b, im_b); */

    
    std::vector<double> dH((sum_a + 1) * (sum_b + 1),0.0);
   

    dH[0] = 1.0;

    
    for (k = 0; k < n; k++)
    {
        s_a += iscore_a[k];
        s_b += iscore_b[k];

        for (i = min(im_a, s_a); i >= iscore_a[k]; i--) {
    
            isb = i * (sum_b + 1);
             
            for (j = min(im_b,s_b); j >= iscore_b[k]; j--){
                dH[isb + j] +=
                    dH[(i - iscore_a[k]) * (sum_b + 1) + (j - iscore_b[k])];
            }
        }
    }


    isb = im_a * (sum_b + 1);

    std::vector<double> dx(sum_b,0.0);
    
    isb = im_a * (sum_b + 1);
    
    PyObject* result = PyList_New(0);
    for (j = 0; j < sum_b; j++){
      PyList_Append(result, PyFloat_FromDouble(dH[isb + j + 1]));
        }

    


    return result;
    
}


static PyObject* GreenOpenMP(PyObject* self, PyObject* args){
  
  int i;
  int j;
  int s;

  int m;
  int n;
  int S;
  


PyArrayObject* py_z;

PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &py_z, &m, &n, &S);


int * z = (int *)py_z->data;



int width = S + 1;
std::vector<double> N(width*m,0.0);
std::vector<double> N_old(width*m,0.0);




double sum = 0;

for (i = 1; i < (m + n) + 1; i++)
{
    sum += z[i - 1];
    if (i % 2 == 1)
    {   
        for (j = 1; j < min(i+1,m+1); j++) {
            #pragma omp parallel for
            for (s = 0; s < min(sum+1,S+1); s++) {
                if (i<j) {
                    N[width*(j-1) + s] = 0;
                }
                else if (j == 1 && z[i - 1] == s)
                {
                    N[width*(j-1) + s] = N_old[width*(j-1) + s] + 1;
                }
               
                else if (j > 1 && z[i - 1] <= s)
                {   
                    N[width*(j-1) + s] = N_old[width*(j-2) + (s - z[i -1])] + N_old[width*(j-1) + s];
                }
                
                else {
                    N[width*(j-1) + s] = N_old[width*(j-1) + s];
                }
            }
        }

       
    } else {
        for (j = 1; j < min(i+1,m+1); j++) {
            #pragma omp parallel for
            for (s = 0; s < min(sum+1,S+1); s++) {
                if (i<j) {
                    N_old[width*(j-1) + s] = 0;
                }
                else if (j == 1 && z[i - 1] == s)
                {
                    N_old[width*(j-1) + s] = N[width*(j-1) + s] + 1;
                }
                
                else if (j > 1 && z[i - 1] <= s)
                {
                    N_old[width*(j-1) + s] = N[width*(j-2) + (s - z[i -1])] + N[width*(j-1) + s];                    
                    
                }
                
                else {
                    N_old[width*(j-1) + s] = N[width*(j-1) + s];
                }
            }
        }
    } 
}

PyObject *result = PyList_New(0);
double msum = 0;
if (i % 2 == 1) {
    
    for (s = 0; s < S+1; s++) {
        PyList_Append(result, PyFloat_FromDouble(N_old[width*(m-1) + s]));
    }


} else {
    
    for (s = 0; s < S+1; s++) {
        PyList_Append(result, PyFloat_FromDouble(N[width*(m-1) + s]));
        
    }
} 


return result;

}

static PyObject* Green(PyObject* self, PyObject* args){
  
  int i;
  int j;
  int s;

  int m;
  int n;
  int S;
  


PyArrayObject* py_z;

PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &py_z, &m, &n, &S);


int * z = (int *)py_z->data;

int width = S + 1;
std::vector<double> N(width*m,0.0);
std::vector<double> N_old(width*m,0.0);


double sum = 0;

for (i = 1; i < (m + n) + 1; i++)
{
    sum += z[i - 1];
    if (i % 2 == 1)
    {   
        for (j = 1; j < min(i+1,m+1); j++) {
            for (s = 0; s < min(sum+1,S+1); s++) {
                if (i<j) {
                    N[width*(j-1) + s] = 0;
                }
                else if (j == 1 && z[i - 1] == s)
                {
                    N[width*(j-1) + s] = N_old[width*(j-1) + s] + 1;
                }
               
                else if (j > 1 && z[i - 1] <= s)
                {   
                    N[width*(j-1) + s] = N_old[width*(j-2) + (s - z[i -1])] + N_old[width*(j-1) + s];
                }
                
                else {
                    N[width*(j-1) + s] = N_old[width*(j-1) + s];
                }
            }
        }

       
    } else {
        for (j = 1; j < min(i+1,m+1); j++) {
            for (s = 0; s < min(sum+1,S+1); s++) {
                if (i<j) {
                    N_old[width*(j-1) + s] = 0;
                }
                else if (j == 1 && z[i - 1] == s)
                {
                    N_old[width*(j-1) + s] = N[width*(j-1) + s] + 1;
                }
                
                else if (j > 1 && z[i - 1] <= s)
                {
                    N_old[width*(j-1) + s] = N[width*(j-2) + (s - z[i -1])] + N[width*(j-1) + s];                    
                    
                }
                
                else {
                    N_old[width*(j-1) + s] = N[width*(j-1) + s];
                }
            }
        }
    } 
}

PyObject *result = PyList_New(0);
double msum = 0;

if (i % 2 == 1) {
    for (s = 0; s < S+1; s++) {
        msum += N_old[width*(m-1) + s];
    }

    
    for (s = 0; s < S+1; s++) {
        PyList_Append(result, PyFloat_FromDouble(N_old[width*(m-1) + s] ));
    }


} else {

    for (s = 0; s < S+1; s++) {
        PyList_Append(result, PyFloat_FromDouble(N[width*(m-1) + s]));
        
    }
} 


return result;

}


static PyMethodDef myMethods[] = {
    {"greenCUDA", greenCUDA, METH_VARARGS, "Calculte permutation distribution with CUDA."},
    {"GreenOpenMP",GreenOpenMP,METH_VARARGS,"Calculate the permuation distribution."},
    {"Green",Green,METH_VARARGS,"Calculate the permuation distribution."},
    {"coinShift",coinShift,METH_VARARGS,"Calculate the permuation distribution with coins implementation of the Shift Method."},
    {"gpu_available_mem",gpu_available_mem,METH_NOARGS,"Get available on GPU."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef permutationTest = {
    PyModuleDef_HEAD_INIT, "permutationTest",
    "permutationTest", -1, myMethods
};

PyMODINIT_FUNC PyInit_permutationTest(void) {
    import_array(); 
    return PyModule_Create(&permutationTest);
}
