import numpy as np
import multiscale_phate as mp

def graph_condensation(data, random_state=42, n_jobs=-1, condensation_threshold=10000, n_pca=100):
    mp_op = mp.Multiscale_PHATE(random_state=random_state, n_jobs=n_jobs, n_pca=n_pca)
    levels = mp_op.fit(data)
    number_of_condensed_points = np.array([np.unique(x).shape[0] for x in mp_op.NxTs])
    condensed_level = np.argwhere(number_of_condensed_points <= condensation_threshold)[0][0]
    return (np.array(mp_op.NxTs[condensed_level]))

def aggregate_signals_over_condensed_nodes(data, condensation_groupings):
    clust_unique, clust_unique_ids = np.unique(condensation_groupings, return_index=True)
    loc = []
    for c in clust_unique:
        loc.append(np.where(condensation_groupings == c)[0])

    counts_condensed = []
    for l in loc:
        counts_condensed.append(data[l].mean(axis=0))
        
    return (np.array(counts_condensed))
