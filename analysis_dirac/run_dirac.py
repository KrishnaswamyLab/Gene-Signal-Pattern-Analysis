import torch, phate, scipy, scprep, keras, magic, graphtools, sys
sys.path.append('..')
import torch_geometric.transforms as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data
from torch_geometric.nn import GAE
from collections import defaultdict
from keras import layers
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn import linear_model, decomposition
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr,  pearsonr
from localization import *
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from DiffusionEMD import DiffusionCheb

run = sys.argv[1]

def svd(signals):
    pc_op = decomposition.TruncatedSVD(n_components=2048)
    data_pc = pc_op.fit_transform(signals)
    
    data_pc_std = data_pc / np.std(data_pc[:, 0])
    
    return (data_pc_std)

def autoencoder(data_pc_std):
    input = keras.Input(shape=(data_pc_std.shape[1],))
    encoded = layers.Dense(512, activation='relu')(input)
    encoded = layers.Dense(128, activation='linear')(encoded)
    decoded = layers.Dense(512, activation='relu')(encoded)
    decoded = layers.Dense(data_pc_std.shape[1], activation='linear')(decoded)

    autoencoder = keras.Model(input, decoded)
    encoder = keras.Model(input, encoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)

    history = autoencoder.fit(data_pc_std, data_pc_std,
                    epochs=100,
                    batch_size=256,
                    shuffle=True,
                    callbacks=[callback],
                   )
    raw_n_epochs = len(history.history['loss'])
    data_ae = encoder(data_pc_std).numpy()
    
    return (data_ae)

datasets = {}

print ('Load data...')
trajectory_data = np.load('../data/splatter_simulated_data.npz')
data_libnorm_sqrt = trajectory_data['data']
pseudotime = trajectory_data['pseudotime']
data_libnorm_sqrt= data_libnorm_sqrt[np.argsort(pseudotime)]
pseudotime = np.array(list(range(10000))) / 10000

print ('Compute PHATE...')
phate_op = phate.PHATE(random_state=0, use_pygsp=True, verbose=0, n_jobs=1)
data_phate = phate_op.fit_transform(data_libnorm_sqrt)

print ('Compute Wavelets...')
loc = Localizer(phate_op.graph)
loc.CalculateWavelets(use_reduced=False)
loc.FlattenAndNormalize() ## to get normalized and flattened wavelets
datasets['GSPA'] = loc.FlatWaves

loc = Localizer(phate_op.graph)
loc.CalculateWavelets(use_reduced=True)
loc.FlattenAndNormalize() ## to get normalized and flattened wavelets
datasets['GSPA_QR'] = loc.FlatWaves

del(loc)

all_signals = np.eye(10000)

print ('Compute MAGIC...')
magic_op = magic.MAGIC(verbose=False, n_jobs=8)
magic_op.graph = phate_op.graph
datasets['MAGIC'] = magic_op.transform(all_signals.T).T

dirac_embeddings = defaultdict(dict)
f = open(f"results/spearmanr_{run}.txt", "a")

for spacing in [int(2**i) for i in range(1,11)][::-1]:
    datasets_curr_run = {**datasets}
    
    for (name, signals) in datasets_curr_run.items():
        print (f'{name} Spacing {spacing}')
    
        index = np.array(list(range(run, 10000-run, spacing)))
        labels_y = pseudotime[index]
    
        all_signals_subsampled = signals[index]
    
        data_pc_std = svd(all_signals_subsampled)
        data_ae = autoencoder(data_pc_std)

        if spacing == 64:
            np.save(f'results/{name}_{spacing}.npy')
    
        kf = RepeatedKFold(n_splits=2, n_repeats=20)
        splits = kf.split(data_ae)
    
        for (train_index, test_index) in splits:
            X_train = data_ae[train_index]
            X_test = data_ae[test_index]
            y_train = labels_y[train_index]
            y_test = labels_y[test_index]
    
            regr = linear_model.Ridge()
            regr.fit(X_train, y_train)
            y_pred = regr.predict(X_test)
    
            spearmans[name].append(spearmanr(y_test, y_pred).correlation)
    
        f.write(f'{spacing} {name} Spearman {np.median(spearmans[name])}\n')