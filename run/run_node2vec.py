from node2vec import Node2Vec
import numpy as np
import networkx as nx
import pandas as pd
import json
import os

def run_node2vec(G, args):
    
    node2vec = Node2Vec(G, dimensions=args.dim, walk_length=args.walk_length, num_walks=args.num_walks)
    model = node2vec.fit()
    node_ids = model.wv.index_to_key  # list of node IDs
    embedding = model.wv.vectors

    return (embedding)