import numpy as np
from collections import defaultdict
import os, sys, glob
from scipy.stats import spearmanr
from sklearn import linear_model
from sklearn.model_selection import RepeatedKFold

model = sys.argv[1]
dataset = sys.argv[2]

if dataset == '2_branches':
    datafile = 'splatter_simulated_data_2_branches.npz'
    extension = '_2_branches'
if dataset == 'sparse_branches':
    datafile = 'splatter_simulated_data_sparse_branches.npz'
    extension = '_sparse_branches'
if dataset == '3_branches':
    datafile = 'splatter_simulated_data_3_branches.npz'
    extension = '_3_branches'
elif dataset == 'linear':
    datafile = 'splatter_simulated_data.npz'
    extension = ''

# confirm model choice
if model not in ['Eigenscore','GFMMD', 'Signals', 'DiffusionEMD', 'GSPA', 'GSPA_QR', 'MAGIC', 'Node2Vec_Gcell', 'GAE_noatt_Gcell', 'GAE_att_Gcell', 'Node2Vec_Ggene', 'GAE_noatt_Ggene', 'GAE_att_Ggene']:
    sys.exit('Model choice not in [Eigenscore GFMMD Signals DiffusionEMD GSPA GSPA_QR MAGIC Node2Vec_Gcell GAE_noatt_Gcell GAE_att_Gcell Node2Vec_Ggene GAE_noatt_Ggene GAE_att_Ggene]')

spearmans = defaultdict(list)
labels_y = np.load(f'../data/localization_signals{extension}.npz')['spread']

# get embeddings
localization_results = {}
localization_scores = {}

for id in [7, 8, 9]:
    run = f'../results/localization/{model}/{id}_results{extension}.npz'
    res = np.load(run, allow_pickle=True)
    name = res['config'][()]['save_as']
    localization_results[name] = res['signal_embedding']
    localization_scores[name] = res['localization_score']
    
# set up results output
if not os.path.exists(f'./uniform_results/{model}/'):
    os.makedirs(f'./uniform_results/{model}')
f = open(f'./uniform_results/{model}/spearmanr{extension}_789.txt', 'a')

for (name, score) in localization_scores.items():
    spearmans[name] = spearmanr(score, labels_y).correlation
    f.write(f'{name} Spearman {spearmans[name]}\n')
    
f.close()
