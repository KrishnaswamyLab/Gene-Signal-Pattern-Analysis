from config import parser
import numpy as np
import torch
import graphtools

from gspa import calculate_wavelet_dictionary
from project import learn_representation
from run.run_node2vec import run_node2vec
from run.run_gae import run_gae
from run.run_ae import run_ae

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print ('Load data...')
    trajectory_data = np.load('trajectory_data_10000_dropout_0.95.npz')
    data_libnorm_sqrt = trajectory_data['data']
    pseudotime = trajectory_data['pseudotime']
    true_data_libnorm_sqrt = trajectory_data['true_counts']
    data_libnorm_sqrt= data_libnorm_sqrt[np.argsort(pseudotime)]
    true_data_libnorm_sqrt = true_data_libnorm_sqrt[np.argsort(pseudotime)]
    pseudotime = pseudotime[np.argsort(pseudotime)]
    true_lib_size = true_data_libnorm_sqrt.T.sum(axis=1)

    print ('Build cellular graph...')
    G = graphtools.Graph(data_libnorm_sqrt, random_state=args.seed, n_jobs=-1)

    if args.model == 'GSPA':
        dictionary, wavelet_sizes = calculate_wavelet_dictionary(G)
        signals_reduced = learn_representation(signals, dictionary)
        run_ae(signals_reduced, args)
    
    if args.model == 'Node2Vec':
        dictionary = run_node2vec(G, args)
        signals_reduced = learn_representation(signals, dictionary)
        run_ae(signals_reduced, args)

    elif args.model == 'GAE':
        dictionary = run_gae(G, args)
        signals_reduced = learn_representation(signals, dictionary)
        run_ae(signals_reduced, args)
     
if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
