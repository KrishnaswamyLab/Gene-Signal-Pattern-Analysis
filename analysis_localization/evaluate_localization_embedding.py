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
for run in os.listdir(f'../results/localization/{model}'):
    name = run.split('_results.npz')[0]
    localization_results[name] = np.load(f'../results/localization/{model}/{run}')['signal_embedding']
    
# set up results output
if not os.path.exists(f'./embedding_results/{model}/'):
    os.makedirs(f'./embedding_results/{model}')
f = open(f'./embedding_results/{model}/spearmanr.txt', 'a')

for (name, embedding) in localization_results.items():
    kf = RepeatedKFold(n_splits=2, n_repeats=10)
    splits = kf.split(embedding)

    for (train_index, test_index) in splits:
        X_train = embedding[train_index]
        X_test = embedding[test_index]
        y_train = labels_y[train_index]
        y_test = labels_y[test_index]

        regr = linear_model.Ridge()
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        
        spearmans[name].append(spearmanr(y_test, y_pred).correlation)
        
    f.write(f'{name} Spearman {np.median(spearmans[name])}\n')
    
f.close()
