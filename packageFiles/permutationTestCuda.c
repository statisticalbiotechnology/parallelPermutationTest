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

/* static PyObject* greenCUDA(PyObject* self, PyObject* args) {
    double * out;

    int m;
    int n;
    int S;

    PyArrayObject* z;
      
    PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &z, &m, &n, &S);

    out = greenCuda((unsigned int *) z -> data, m,n,S);
    
    PyObject* result = PyList_New(0);
    for (int i = 0; i < S+1; i++){
      PyList_Append(result, PyFloat_FromDouble(out[i]));
        }

    return result;
}
 */


static PyMethodDef myMethods[] = {
    {"greenCUDA", greenCUDA, METH_VARARGS, "Calculte permutation distribution with CUDA."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef permutationTestCuda = {
    PyModuleDef_HEAD_INIT, "permutationTestCuda",
    "permutationTestCuda", -1, myMethods
};

PyMODINIT_FUNC PyInit_permutationTestCuda(void) {
    import_array(); 
    return PyModule_Create(&permutationTestCuda);
}
