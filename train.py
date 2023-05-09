import os
import json
from config import parser
import numpy as np
import torch, phate
import graphtools

from gspa import calculate_wavelet_dictionary
from utils import learn_representation, calculate_localization
from run.run_node2vec import run_node2vec
from run.run_gae import run_gae
from run.run_ae import run_ae

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print ('Load data...')
    trajectory_data = np.load('/home/aarthivenkat/Gene-Signal-Pattern-Analysis/data/splatter_simulated_data.npz')
    data_libnorm_sqrt = trajectory_data['data']
    pseudotime = trajectory_data['pseudotime']
    true_data_libnorm_sqrt = trajectory_data['true_counts']
    data_libnorm_sqrt= data_libnorm_sqrt[np.argsort(pseudotime)]
    true_data_libnorm_sqrt = true_data_libnorm_sqrt[np.argsort(pseudotime)]
    pseudotime = pseudotime[np.argsort(pseudotime)]
    true_lib_size = true_data_libnorm_sqrt.T.sum(axis=1)

    print ('Build cellular graph...')
    phate_op = phate.PHATE(random_state=args.seed, use_pygsp=True, n_jobs=-1)
    phate_op.fit(data_libnorm_sqrt)
    G = phate_op.graph
    del(phate_op)

    if args.task == 'coexpression':
        signals = data_libnorm_sqrt.T
        signals = signals / np.linalg.norm(signals, axis=1).reshape(-1,1)

    elif args.task == 'localization':
        signals = np.load('/home/aarthivenkat/Gene-Signal-Pattern-Analysis/data/localization_signals.npz')['signals']
        signals = signals / np.linalg.norm(signals, axis=1).reshape(-1,1)

    if args.model == 'GSPA':
        dictionary, wavelet_sizes = calculate_wavelet_dictionary(G)
        signals_reduced = learn_representation(signals, dictionary)
        localization_score = calculate_localization(signals, dictionary)
        run_ae(signals_reduced, args)
    
    elif args.model == 'Node2Vec':
        dictionary = run_node2vec(G, args)
        signals_reduced = learn_representation(signals, dictionary)
        localization_score = calculate_localization(signals, dictionary)
        run_ae(signals_reduced, args)

    elif args.model == 'GAE':
        dictionary = run_gae(G, args)
        signals_reduced = learn_representation(signals, dictionary)
        localization_score = calculate_localization(signals, dictionary)
        run_ae(signals_reduced, args)
        
    if not os.path.exists(f'results/{args.model}/'):
        os.makedirs(f'results/{args.model}')

    np.save(f'results/{args.model}/{args.save_as}_localization.npy', localization_score)
    with open(f'results/{args.model}/{args.save_as}_config.json', 'w') as f:
        json.dump(vars(args), f)
     
if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
