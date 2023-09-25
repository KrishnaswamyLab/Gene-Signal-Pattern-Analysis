import numpy as np
from sklearn import decomposition
from scipy.spatial.distance import cdist

def svd(signals):
    n_components = min(2048, signals.shape[0], signals.shape[1]) 
    pc_op = decomposition.PCA(n_components=n_components)
    data_pc = pc_op.fit_transform(signals)
    
    # normalize before autoencoder
    data_pc_std = data_pc / np.std(data_pc[:, 0])
    
    return (data_pc_std)

def project(signals, cell_dictionary):
    return(np.dot(signals, cell_dictionary))

def calculate_localization(x, signals, metric='euclidean'):
    x = x.reshape(1,-1)
    return(cdist(x, signals, metric=metric).reshape(-1,))
