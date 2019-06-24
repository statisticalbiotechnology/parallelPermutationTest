from numba import cuda
@cuda.jit("(u4[:,:,:],u4[:,:,:], u2, u2[:,:,:],u2[:])")
def get_permutationsA_u4_v_u2(X1,X2, k_, z_,S):
    n = X1.shape[0]
    m = X1.shape[1]
    L = X1.shape[2]

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z

    if j >= 2 and j < m + 1 and i <= S[z] and z < L:
        if i >= z_[z, 0,k_- 1]:
            X2[i, j-1,z] = X1[i - int(z_[z, 0,k_- 1]) ,j-2,z] +  X1[i,j-1,z]
        else:
            X2[i, j-1,z] = X1[i,j-1,z]
        if i < k_ and (j-2)==0 and k_ <= int(z_[z, 2, 0:k_][i]):
            X2[int(z_[z, 0, 0:k_][i]) , 0,z] = int(z_[z, 1, 0:k_][i])

@cuda.jit("(f8[:,:,:],f8[:,:,:], u2, u2[:,:,:],u2[:])")
def get_permutationsA_f8_v_u2(X1,X2, k_, z_,S):
    n = X1.shape[0]
    m = X1.shape[1]
    L = X1.shape[2]

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z

    if j >= 2 and j < m + 1 and i <= S[z] and z < L:

        if i >= z_[z, 0,k_- 1]:
            E = X1[i - int(z_[z, 0,k_- 1]) ,j-2,z] +  X1[i,j-1,z]
        else:
            E = X1[i,j-1,z]

        X2[i, j-1,z] = E

        if i < k_ and (j-2)==0 and k_ <= int(z_[z, 2, 0:k_][i]):
            X2[int(z_[z, 0, 0:k_][i]) , 0,z] = int(z_[z, 1, 0:k_][i])

@cuda.jit("(f8[:,:,:],f8[:,:,:], u4, u4[:,:,:],u4[:])")
def get_permutationsA_f8_v_u4(X1,X2, k_, z_,S):
    n = X1.shape[0]
    m = X1.shape[1]
    L = X1.shape[2]

    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z

    if j >= 2 and j < m + 1 and i <= S[z] and z < L:
       
        if i >= z_[z, 0,k_- 1]:
            E = X1[i - int(z_[z, 0,k_- 1]) ,j-2,z] +  X1[i,j-1,z]
        else:
            E = X1[i,j-1,z]

        X2[i, j-1,z] = E
        
        if i < k_ and (j-2)==0 and k_ <= int(z_[z, 2, 0:k_][i]):
            X2[int(z_[z, 0, 0:k_][i]) , 0,z] = int(z_[z, 1, 0:k_][i])
