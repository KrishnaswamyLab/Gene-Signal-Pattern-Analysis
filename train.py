import os
import magic, phate
from config import parser
import numpy as np
import sklearn
import torch
import graphtools
import pygsp
from scipy.spatial.distance import cdist 
from DiffusionEMD import DiffusionCheb

from run.run_gspa import calculate_wavelet_dictionary
from utils import svd, project, calculate_localization
from run.run_node2vec import run_node2vec
from run.run_gae import run_gae
from run.run_ae import run_ae
from run import run_gfmmd 

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print ('Load data...')
    trajectory_data = np.load('./data/splatter_simulated_data.npz')
    data = trajectory_data['data']
    NCELLS = data.shape[0]
    
    print ('Uniform signal...')
    uniform_signal = np.ones((1, NCELLS))
    uniform_signal = uniform_signal / np.linalg.norm(uniform_signal, axis=1).reshape(-1,1)

    print ('Build cellular graph...')
    phate_op = phate.PHATE(random_state=args.seed, use_pygsp=True, n_jobs=-1, verbose=args.verbose)
    phate_op.fit(data)
    G = phate_op.graph
    
    results = {}
    
    ## TASK SETUP
    if args.task == 'coexpression':
        signals = data.T
        signals = signals / np.linalg.norm(signals, axis=1).reshape(-1,1)

    elif args.task == 'localization':
        signals = np.load('./data/localization_signals.npz')['signals']
        signals = signals / np.linalg.norm(signals, axis=1).reshape(-1,1)
        
    ## MODEL SETUP        
    if args.model == 'Signals':
        args.comparison = 'Signals'
        signal_reduced = svd(signals)
        results['signal_embedding'] = run_ae(signal_reduced, args)
        results['localization_score'] = calculate_localization(uniform_signal, signals)
        
    if args.model in ['DiffusionEMD', 'GFMMD']:
        args.comparison = 'OT'
        signals_t = signals.T
        signal_prob = signals_t
        signal_prob = signals_t / signals_t.sum(axis=0)
        uniform_signal_t = uniform_signal.reshape(-1, 1)
        uniform_prob = uniform_signal_t / uniform_signal_t.sum(axis=0)

        if args.model == 'DiffusionEMD':
            dc_op = DiffusionCheb()
            signal_representation = dc_op.fit_transform(G.A, np.hstack((signal_prob, uniform_prob)))
            uniform_representation = signal_representation[-1]
            signal_representation = signal_representation[:-1]

            results['localization_score'] = calculate_localization(uniform_representation, signal_representation, metric='cityblock')
        elif args.model == 'GFMMD':
            gfmmd_op = run_gfmmd.Graph_Fourier_MMD(G)
            signal_representation = gfmmd_op.distance(np.hstack((signal_prob, uniform_prob)))
            uniform_representation = signal_representation[-1]
            signal_representation = signal_representation[:-1]
            results['localization_score'] = gfmmd_op.locality(signal_prob)

        signal_reduced = svd(signal_representation)
        results['signal_embedding'] = run_ae(signal_reduced, args)
        
    elif args.model in ['GSPA', 'GSPA_QR', 'MAGIC', 'Node2Vec_Gcell', 'GAE_noatt_Gcell', 'GAE_att_Gcell']:
        args.comparison = 'Projection'
        if args.model == 'GSPA':
            use_reduced = False
            cell_dictionary, wavelet_sizes = calculate_wavelet_dictionary(G, use_reduced=use_reduced)
        elif args.model == 'GSPA_QR':
            use_reduced = True
            cell_dictionary, wavelet_sizes = calculate_wavelet_dictionary(G, use_reduced=use_reduced)        
        elif args.model == 'Node2Vec_Gcell':
            cell_dictionary = run_node2vec(G, args)
        elif args.model == 'GAE_noatt_Gcell':
            args.attention = False
            cell_dictionary = run_gae(G, args)
        elif args.model == 'GAE_att_Gcell':
            args.attention = True
            cell_dictionary = run_gae(G, args)
        elif args.model == 'MAGIC':
            # related to MAGIC; project gene signals onto diffusion operator
            cell_dictionary = phate_op.diff_op
            
        signals_projected = project(signals, cell_dictionary)
        signals_reduced = svd(signals_projected)
        results['signal_embedding'] = run_ae(signals_reduced, args)
        
        uniform_projected = project(uniform_signal, cell_dictionary)
        results['localization_score'] = calculate_localization(uniform_projected, signals_projected)
        
    elif args.model in ['Node2Vec_Ggene', 'GAE_noatt_Ggene', 'GAE_att_Ggene']:
        args.comparison = 'Ggene'
        signal_graph = sklearn.neighbors.kneighbors_graph(signals, n_neighbors=args.k_neighbors)
        signal_graph = pygsp.graphs.Graph(signal_graph)
        signal_graph.W = signal_graph.A
        
        # localization calculation requires separate graph with uniform signal
        signals_with_uniform = np.vstack((signals, uniform_signal))
        signals_with_uniform_graph = sklearn.neighbors.kneighbors_graph(signals_with_uniform, n_neighbors=args.k_neighbors)
        signals_with_uniform_graph = pygsp.graphs.Graph(signals_with_uniform_graph)
        signals_with_uniform_graph.W = signals_with_uniform_graph.A

        if args.model == 'Node2Vec_Ggene':
            results['signal_embedding'] = run_node2vec(signal_graph, args)
            signals_with_uniform_embedding = run_node2vec(signals_with_uniform_graph, args)
        elif args.model == 'GAE_noatt_Ggene':
            args.attention = False
            results['signal_embedding'] = run_gae(signal_graph, args)
            signals_with_uniform_embedding = run_gae(signals_with_uniform_graph, args)
        elif args.model == 'GAE_att_Ggene':
            args.attention = True
            results['signal_embedding'] = run_gae(signal_graph, args)
            signals_with_uniform_embedding = run_gae(signals_with_uniform_graph, args)

        results['localization_score'] = calculate_localization(signals_with_uniform_embedding[-1], signals_with_uniform_embedding[:-1])
        
    ## small test
    assert(results['signal_embedding'].shape == (signals.shape[0], args.dim))
    assert(results['localization_score'].shape == (signals.shape[0],))

    if not os.path.exists(f'./results/{args.task}/{args.model}/'):
        os.makedirs(f'./results/{args.task}/{args.model}')
        
    np.savez_compressed(f'./results/{args.task}/{args.model}/{args.save_as}_results.npz', signal_embedding=results['signal_embedding'], localization_score=results['localization_score'], config=vars(args))

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
