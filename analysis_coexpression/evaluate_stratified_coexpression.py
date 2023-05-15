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

trajectory_data = np.load('../data/splatter_simulated_data.npz')
data = trajectory_data['data']
true_counts = trajectory_data['true_counts']
true_lib_size = true_counts.T.sum(axis=1)

# get coexpression embeddings
coexpression_results = {}
for run in os.listdir(f'../results/coexpression/{model}'):
    name = run.split('_results.npz')[0]
    coexpression_results[name] = np.load(f'../results/coexpression/{model}/{run}')['signal_embedding']
    
# set up results output
if not os.path.exists(f'./results/{model}/'):
    os.makedirs(f'./results/{model}')
f = open(f"results/{model}/spearmanr.txt", "a")

print ('Stratify Spearman correlation...')
spearman_res = spearmanr(true_counts)
np.fill_diagonal(spearman_res.correlation, 0)
corr_bins = np.linspace(spearman_res.correlation.min(), spearman_res.correlation.max(), 4)

min_bin_size = float('inf')
for i,corr in enumerate(corr_bins):
    if i == 0: continue
    choices = np.array(np.where((spearman_res.correlation > corr_bins[i-1]) & (spearman_res.correlation < corr) & (spearman_res.correlation != 0))).T
    if choices.shape[0] < min_bin_size:
        min_bin_size = choices.shape[0]

spearmans = defaultdict(list)

print ('Stratify library size...')
min_lib_size= float('inf')
for i,corr in enumerate(corr_bins):

    choices_bin = []
    if i == 0: continue

    ## res.correlation does not equal zero, excluding self edges
    choices = np.array(np.where((spearman_res.correlation > corr_bins[i-1]) & (spearman_res.correlation < corr) & (spearman_res.correlation != 0))).T
    
    lib_size_mean_per_pair = np.vstack((true_lib_size[choices[:, 0]], true_lib_size[choices[:, 1]])).mean(axis=0)
    lib_size = np.linspace(lib_size_mean_per_pair.min(), lib_size_mean_per_pair.max(), 3)

    for j,bin in enumerate(lib_size):
        if j == 0: continue
        choices_ = np.array(np.where((lib_size_mean_per_pair > lib_size[j-1]) & (lib_size_mean_per_pair < bin))).T
        if choices_.shape[0] < min_lib_size:
            min_lib_size = choices_.shape[0]

print ('Sample by stratification...')
samples = []
for i,corr in enumerate(corr_bins):

    choices_bin = []
    if i == 0: continue

    ## res.correlation does not equal zero, excluding self edges
    choices = np.array(np.where((spearman_res.correlation > corr_bins[i-1]) & (spearman_res.correlation < corr) & (spearman_res.correlation != 0))).T
    
    lib_size_mean_per_pair = np.vstack((true_lib_size[choices[:, 0]], true_lib_size[choices[:, 1]])).mean(axis=0)
    lib_size = np.linspace(lib_size_mean_per_pair.min(), lib_size_mean_per_pair.max(), 3)

    for j,bin in enumerate(lib_size):
        if j == 0: continue
        choices_ = np.array(np.where((lib_size_mean_per_pair > lib_size[j-1]) & (lib_size_mean_per_pair < bin))).T
        choices_bin.append(choices_[np.random.choice(choices_.shape[0], size=min_lib_size, replace=False, )])

    samples.append(choices[np.vstack(choices_bin).flatten()])

samples = np.vstack(samples)

for (name, embedding) in coexpression_results.items():
    kf = RepeatedKFold(n_splits=2, n_repeats=10)
    splits = kf.split(samples)

    for (train_index, test_index) in splits:
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        train_index = samples[train_index]
        test_index = samples[test_index]

        for (a,b) in train_index:
            X_train.append(np.hstack((embedding[a], embedding[b])))
            y_train.append(spearman_res.correlation[a][b])
        for (a,b) in test_index:
            X_test.append(np.hstack((embedding[a], embedding[b])))
            y_test.append(spearman_res.correlation[a][b])

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        regr = linear_model.Ridge()
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)

        spearmans[name].append(spearmanr(y_test, y_pred).correlation)
        
    f.write(f'{name} Spearman {np.median(spearmans[name])}\n')
    
f.close()
