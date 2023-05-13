from node2vec import Node2Vec
import networkx as nx

def run_node2vec(G, args):
    G = nx.from_numpy_matrix(G.W)
    node2vec = Node2Vec(G, dimensions=args.dim, walk_length=args.walk_length, num_walks=args.num_walks)
    model = node2vec.fit()
    node_ids = model.wv.index_to_key  # list of node IDs
    embedding = model.wv.vectors

    return (embedding)
