import scipy
import pygsp
import numpy as np

class Graph_Fourier_MMD:
    def __init__(self, Graph = None):
        if Graph != None:
            self.G = Graph
            self.T = self.G.L.trace()
        else:
            raise ValueError("Graph Required")
            
    def feature_map(self, signals, method = 'chebyshev', filter_name='default'):
        
        n = signals.shape[1]

        if n == 1:
            raise ValueError("Need more than two signals to compare")
            return False 
        
        signals = np.hstack([signals[:,i].reshape(-1,1)/np.sum(signals[:,i]) for i in range(signals.shape[1])])
        
        
        if method != "chebyshev":
            self.G.compute_fourier_basis()
        

        if filter_name == 'heat':
            heat_filter = pygsp.filters.Heat(self.G)
            use_filter = heat_filter
        elif filter_name == "default":
            recip_filter = pygsp.filters.Filter(self.G, kernels = [lambda x : [i**(-1/2) if i > 0 else 0 for i in x]])
            use_filter = recip_filter
        else:
            raise NameError("Filter name must either be 'heat' or 'default'")
        
        if method == "chebyshev":
            ret = use_filter.filter(signals, method = 'chebyshev', order = 128) * (self.T)**(1/2)
        else:
            ret = use_filter.filter(signals, method = 'exact') * (self.T)**(1/2)
            
        return ret.reshape(-1, signals.shape[1])

    def locality(self, transformed):
        return np.array([np.linalg.norm(t) for t in transformed.T])

    def distance(self, transformed):
        distance_array = scipy.spatial.distance.pdist(transformed.T)
        distances = scipy.spatial.distance.squareform(distance_array)

        if n == 2:
            return distances[0,1]
        else:
            return distances
