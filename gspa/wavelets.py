import numpy as np
from scipy.linalg import qr
                      
def normalize(A):
    """
    Input: A, an n x m matrix 
    Output: A with each column divided by its L2 norm
    """
    
    for i in range(A.shape[1]):
        A[:,i]=A[:,i]/np.linalg.norm(A[:,i])
    return A

def column_subset(A,epsilon):
    """
    Input: an m x n matrix A, tolerance epsilon
    Output: Subset of A's columns s.t. the projection of A into these columns; 
            can approximate A with error < epsilon |A|_2
    """
    
    R,P = qr(A,pivoting=True,mode='r')
    A_P = A[:,P]
    
    A_nrm = np.sum(A*A)
    tol = epsilon*A_nrm
    R_nrm = 0
    
    for i in range(0,R.shape[0]):
        R_nrm += np.sum(R[i]*R[i])
        err = A_nrm-R_nrm
        if err < tol:
            return A_P[:,:i]
        
    return A_P

def flatten(wavelet_list, size_of_wavelets_per_scale):
    N = wavelet_list[0].shape[0]
    flat_waves = np.zeros((N,np.sum(size_of_wavelets_per_scale)))
    curr = 0
    for i,wavelet in enumerate(wavelet_list):
        last = curr + size_of_wavelets_per_scale[i]
        flat_waves[:,curr:last] = wavelet
        curr = last
        
    return (np.array(flat_waves), np.array(size_of_wavelets_per_scale))
