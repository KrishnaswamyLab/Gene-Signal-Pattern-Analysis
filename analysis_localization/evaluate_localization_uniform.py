import numpy as np
from collections import defaultdict
import os, sys
from scipy.stats import spearmanr
from sklearn import linear_model
from sklearn.model_selection import RepeatedKFold

model = sys.argv[1]

# confirm model choice
if model not in ['Signals', 'DiffusionEMD', 'GSPA', 'GSPA_QR', 'MAGIC', 'Node2Vec_Gcell', 'GAE_noatt_Gcell', 'GAE_att_Gcell', 'Node2Vec_Ggene', 'GAE_noatt_Ggene', 'GAE_att_Ggene']:
    sys.exit('Model choice not in [Signals DiffusionEMD GSPA GSPA_QR MAGIC Node2Vec_Gcell GAE_noatt_Gcell GAE_att_Gcell Node2Vec_Ggene GAE_noatt_Ggene GAE_att_Ggene]')

spearmans = defaultdict(list)
labels_y = np.load('../data/localization_signals.npz')['spread']

# get embeddings
localization_results = {}
localization_scores = {}
for run in os.listdir(f'../results/localization/{model}'):
    name = run.split('_results.npz')[0]
    localization_results[name] = np.load(f'../results/localization/{model}/{run}')['signal_embedding']
    localization_scores[name] = np.load(f'../results/localization/{model}/{run}')['localization_score']
    
# set up results output
if not os.path.exists(f'./uniform_results/{model}/'):
    os.makedirs(f'./uniform_results/{model}')
f = open(f'./uniform_results/{model}/spearmanr.txt', 'a')

for (name, score) in localization_scores.items():
    spearmans[name] = spearmanr(score, labels_y).correlation
    f.write(f'{name} Spearman {spearmans[name]}\n')
    
f.close()
