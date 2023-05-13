from scipy.stats import spearmanr
import numpy as np
import defaultdict

trajectory_data = np.load('../data/trajectory_data_10000_dropout_0.95.npz')
data = trajectory_data['data']
true_counts = trajectory_data['true_counts']

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

if not os.path.exists(f'./results/{args.model}/'):
    os.makedirs(f'./results/{args.model}')
f = open(f"results/{args.model}/spearmanr.txt", "a")

datasets_curr_run = {**datasets, **stochastic_datasets, **gene_stochastic_datasets}

mses = defaultdict(list)
maes = defaultdict(list)
spearmans = defaultdict(list)
pearsons = defaultdict(list)
r2_scores = defaultdict(list)

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

for (name, signals) in datasets_curr_run.items():
    print (f'{name}...')
    index = np.array(list(range(run, 10000-run, spacing)))
    if name in ['Raw', 'MAGIC']:
        signals = signals.T[index].T #subsample cells

    elif name == 'DiffusionEMD':
        signals = np.array(np.hsplit(signals, 6))
        signals = signals[:, :, index]
        signals = np.concatenate(signals, axis=1)

    elif name in gene_stochastic_datasets.keys():
        signals = signals

    else:
        signals = np.dot(data.T[:, index], signals[index]) #subsample cells, then project

    if name in gene_stochastic_datasets.keys():
        data_ae = signals

    else:
        # stochastic results produce cell embeddings, not gene embeddings, so all go through AE
        print ('SVD...')
        signals = svd(signals)    
        data_ae = autoencoder(signals)

    if run == 0:
        np.savez_compressed(f'{name}_embedding_spacing_{spacing}_run_{run}.npz', data_ae)

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
            X_train.append(np.hstack((data_ae[a], data_ae[b])))
            y_train.append(spearman_res.correlation[a][b])
        for (a,b) in test_index:
            X_test.append(np.hstack((data_ae[a], data_ae[b])))
            y_test.append(spearman_res.correlation[a][b])

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        regr = linear_model.Ridge()
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)

        mses[name].append(mean_squared_error(y_test, y_pred))
        maes[name].append(mean_absolute_error(y_test, y_pred))
        spearmans[name].append(spearmanr(y_test, y_pred).correlation)
        pearsons[name].append(pearsonr(y_test, y_pred)[0])
        r2_scores[name].append(r2_score(y_test, y_pred))