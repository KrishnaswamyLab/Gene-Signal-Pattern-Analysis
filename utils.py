import numpy as np
from sklearn import decomposition
from scipy.spatial.distance import cdist

def svd(signals, random_state=42):
    n_components = min(2048, signals.shape[0], signals.shape[1]) 
    pc_op = decomposition.TruncatedSVD(n_components=n_components, random_state=42)
    data_pc = pc_op.fit_transform(signals)

    data_pc_std = data_pc / np.std(data_pc[:, 0])
    
    return (data_pc_std)

def learn_representation(signals, dictionary):
    projection = np.dot(signals, dictionary)
    projection_svd = svd(projection)

    return (projection_svd)

def calculate_localization(signals, dictionary):
    signals_projection = np.dot(signals, dictionary)
    uniform_signal = np.ones((1,signals.shape[1]))
    uniform_signal = uniform_signal / np.linalg.norm(uniform_signal, axis=1).reshape(-1,1)

    uniform_projection = np.dot(uniform_signal, dictionary)
    distance_to_uniform = cdist(uniform_projection, signals_projection)

    return (distance_to_uniform)