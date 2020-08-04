#include "Python.h"
#include <stdio.h>
#include<vector>
#include <omp.h>
#include <iostream>
#include <chrono>



int min(int num1, int num2);

int min(int num1, int num2) 
{
    return (num1 > num2 ) ? num2 : num1;
}

static PyObject* coinShiftPdist(PyObject* self, PyObject* args){
  
  

  int s_a = 0;
  int s_b = 0;
  int i;
  int k;
  int j;
  int isb;

  int im_a;
  int im_b;

  int sum_a = 0;
  int sum_b = 0;

  double msum = 0.0;

  const char * score_a;
  int size_a;

  const char * score_b;
  int size_b;



PyArg_ParseTuple(args, "y#y#ii", &score_a, &size_a, &score_b, &size_b, &im_a, &im_b);



int * iscore_a = (int *)score_a;
int n = size_a / sizeof(int);

int * iscore_b = (int *)score_b;




  for (i = 0; i < n; i++) {
        sum_a += iscore_a[i];
        sum_b += iscore_b[i];
    }

    

    sum_a = min(sum_a, im_a);
    sum_b = min(sum_b, im_b);

    
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
    for (j = 0; j < sum_b; j++) {
        dx[j] = dH[isb + j + 1];
        msum += dx[j];
        }

    PyObject* result = PyList_New(0);
    for (j = 0; j < sum_b; j++){
      PyList_Append(result, PyFloat_FromDouble(dx[j]/msum));
        }

    


    return result;
    
}


static PyObject* GrennPdistOpenMP(PyObject* self, PyObject* args){
  
  int i;
  int j;
  int s;

  int m;
  int n;
  int S;
  



const char * i_z;
int size_z;
  


PyArg_ParseTuple(args, "y#iii", &i_z,&size_z, &m, &n, &S);


int * z = (int *)i_z;



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
        msum += N_old[width*(m-1) + s];
    }

    
    for (s = 0; s < S+1; s++) {
        PyList_Append(result, PyFloat_FromDouble(N_old[width*(m-1) + s] / msum));
    }


} else {
    for (s = 0; s < S+1; s++) {
        msum += N[width*(m-1) + s];
    }

    
    for (s = 0; s < S+1; s++) {
        PyList_Append(result, PyFloat_FromDouble(N[width*(m-1) + s] / msum));
        
    }
} 


return result;

}

static PyObject* GrennPdist(PyObject* self, PyObject* args){
  
  int i;
  int j;
  int s;

  int m;
  int n;
  int S;
  



const char * i_z;
int size_z;
  


PyArg_ParseTuple(args, "y#iii", &i_z,&size_z, &m, &n, &S);


int * z = (int *)i_z;



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
        PyList_Append(result, PyFloat_FromDouble(N_old[width*(m-1) + s] / msum));
    }


} else {
    for (s = 0; s < S+1; s++) {
        msum += N[width*(m-1) + s];
    }

    
    for (s = 0; s < S+1; s++) {
        PyList_Append(result, PyFloat_FromDouble(N[width*(m-1) + s] / msum));
        
    }
} 


return result;

}



static PyMethodDef mainMethods[] = {
 {"GrennPdistOpenMP",GrennPdistOpenMP,METH_VARARGS,"Calculate the permuation distribution."},
 {"GrennPdist",GrennPdist,METH_VARARGS,"Calculate the permuation distribution."},
 {"coinShiftPdist",coinShiftPdist,METH_VARARGS,"Calculate the permuation distribution with coins implementation of the Shift Method."},
 {NULL,NULL,0,NULL}
};


static PyModuleDef permutationTest = {
 PyModuleDef_HEAD_INIT,
 "permutationTest","Permutation Distribution",
 -1,
 mainMethods
};

PyMODINIT_FUNC PyInit_permutationTest(void){
 return PyModule_Create(&permutationTest);
}


