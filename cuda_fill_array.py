from numba import cuda
@cuda.jit("(u4[:,:,:],u4[:,:,:], u2, u2[:,:],u2[:])")
def fill_array_u4_v_u2(X1,X2, k_, z_,S):
    n = X1.shape[0]
    m = X1.shape[1]
    L = X1.shape[2]

    i, j, z = cuda.grid(3)
    
    if j < 1 or j >= m + 1 or i > S[z] or z >= L:
        return

    if k_ == 1:
        if j == 1 and i == z_[z, k_ - 1]:
            X2[i, j - 1, z] += 1
        else:
            return
    else:
        if j == 1:
            if i == z_[z, k_ - 1]:
                X2[i, j - 1, z]= X1[i, j - 1, z] + 1
            else:
                X2[i, j - 1, z]= X1[i, j - 1, z]
        else:
            if i >= z_[z, k_ - 1]:
                X2[i, j - 1, z] = X1[i - int(z_[z, k_ - 1]), j - 2, z] + X1[i, j - 1, z]
            else:
                X2[i, j-1,z] = X1[i,j-1,z]

@cuda.jit("(f8[:,:,:],f8[:,:,:], u2, u2[:,:],u2[:])")
def fill_array_f8_v_u2(X1,X2, k_, z_,S):
    n = X1.shape[0]
    m = X1.shape[1]
    L = X1.shape[2]

    i, j, z = cuda.grid(3)
    
    if j < 1 or j >= m + 1 or i > S[z] or z >= L:
        return

    if k_ == 1:
        if j == 1 and i == z_[z, k_ - 1]:
            X2[i, j - 1, z] += 1
        else:
            return
    else:
        if j == 1:
            if i == z_[z, k_ - 1]:
                X2[i, j - 1, z]= X1[i, j - 1, z] + 1
            else:
                X2[i, j - 1, z]= X1[i, j - 1, z]
        else:
            if i >= z_[z, k_ - 1]:
                X2[i, j - 1, z] = X1[i - int(z_[z, k_ - 1]), j - 2, z] + X1[i, j - 1, z]
            else:
                X2[i, j-1,z] = X1[i,j-1,z]

@cuda.jit("(f8[:,:,:],f8[:,:,:], u4, u4[:,:],u4[:])")
def fill_array_f8_v_u4(X1,X2, k_, z_,S):
    n = X1.shape[0]
    m = X1.shape[1]
    L = X1.shape[2]

    i, j, z = cuda.grid(3)
    
    if j < 1 or j >= m + 1 or i > S[z] or z >= L:
        return

    if k_ == 1:
        if j == 1 and i == z_[z, k_ - 1]:
            X2[i, j - 1, z] += 1
        else:
            return
    else:
        if j == 1:
            if i == z_[z, k_ - 1]:
                X2[i, j - 1, z]= X1[i, j - 1, z] + 1
            else:
                X2[i, j - 1, z]= X1[i, j - 1, z]
        else:
            if i >= z_[z, k_ - 1]:
                X2[i, j - 1, z] = X1[i - int(z_[z, k_ - 1]), j - 2, z] + X1[i, j - 1, z]
            else:
                X2[i, j-1,z] = X1[i,j-1,z]
