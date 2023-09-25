import numpy as np
import scipy

def run_eigenscore(G, signals, args):
    signals = scipy.sparse.csr_matrix(signals)
    
    degree_mat = np.zeros((G.N, G.N))
    for i in range(G.N):
        degree_mat[i][i] = G.d[i]

    degree_mat = degree_mat**(1/2)

    evs = G.U[:, 1:17]
    eigenscores = np.zeros((signals.shape[0], 16))
    for i in range(signals.shape[0]):
        for j in range(16):
            eigenscores[i][j] = (np.inner((degree_mat * signals[i].todense().T).T, evs[:, j].reshape(1, -1)) / np.linalg.norm((degree_mat * signals[i].todense().T))).item()
            
    return eigenscores
