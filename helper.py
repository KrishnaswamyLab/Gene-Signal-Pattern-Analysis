import numpy as np
from scipy.linalg import qr
from tqdm import tqdm
from sklearn import decomposition

# calculate lazy random walk matrix
def calculate_diffusion_operator(G):
    N = G.N
    A = G.A
    Dmin1 = np.diag([1/np.sum(row) for row in A])
    P = 1/2 * (np.eye(N)+A@Dmin1)
    return P
                      
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
    
    for i in tqdm(range(0,R.shape[0])):
        R_nrm += np.sum(R[i]*R[i])
        err = A_nrm-R_nrm
        if err < tol:
            return A_P[:,:i]
        
    return A_P

def svd(signals, random_state=42):
    n_components = min(2048, signals.shape[1]) 
    pc_op = decomposition.TruncatedSVD(n_components=n_components, random_state=42)
    data_pc = pc_op.fit_transform(signals)

    data_pc_std = data_pc / np.std(data_pc[:, 0])
    return (data_pc_std)
