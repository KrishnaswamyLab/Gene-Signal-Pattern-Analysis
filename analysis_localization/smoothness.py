import numpy as np
import phate, scipy
from scipy.stats import spearmanr
import scprep
import os

model = 'Smoothness'

print ('Load data...')
trajectory_data = np.load('../data/splatter_simulated_data.npz')
data = trajectory_data['data']
NCELLS = data.shape[0]

print ('Build cellular graph...')
phate_op = phate.PHATE(random_state=0, use_pygsp=True, n_jobs=-1)
phate_op.fit(data)
G = phate_op.graph

signals = np.load('../data/localization_signals.npz')['signals']
signals = signals / np.linalg.norm(signals, axis=1).reshape(-1,1)
signals = scipy.sparse.csr_matrix(signals)

labels_y = np.load('../data/localization_signals.npz')['spread']

score = signals * (G.L) * signals.T
score = np.diag(score.todense())

# set up results output
if not os.path.exists(f'./uniform_results/{model}/'):
    os.makedirs(f'./uniform_results/{model}')
f = open(f'./uniform_results/{model}/spearmanr.txt', 'a')
f.write(f'0 Spearman {spearmanr(score, labels_y).correlation}\n')
f.close()
