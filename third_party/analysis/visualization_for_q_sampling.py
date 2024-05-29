import os
import dgl
import torch
import pickle
import random
from collections import Counter
from networkx.algorithms import isomorphism
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
import scipy
import itertools
import networkx as nx
import numpy as np
from utils import get_model_attribute, load_model
from model import create_models
from models.graph_rnn.data import Graph_to_Adj_Matrix
from models.graph_rnn.train import evaluate_loss as eval_loss_graph_rnn
from datasets.process_dataset import default_label_graph
from torch.utils.data._utils.collate import default_collate as collate

def get_model(fname):
    args = get_model_attribute('saved_args', fname, "cuda:0")
    with open(os.path.join("../..", args.current_dataset_path + 'map.dict'), 'rb') as f:
        feature_map = pickle.load(f)

    grnn, gcn = create_models(args, feature_map)
    load_model(fname, "cuda", grnn, gcn=gcn, optimizer=None, scheduler=None)
    gcn.eval()
    for _, net in grnn.items():
        net.eval()

    processor = Graph_to_Adj_Matrix(args, feature_map, random_bfs=True)
    return grnn, gcn, processor, args, feature_map

def build_graph(graph):
    # path = nx.cycle_graph(n_node)
    default_label_graph(graph)
    dgl_graph = dgl.from_networkx(graph)

    node_mat = torch.zeros((len(graph), 1), requires_grad=False)
    for v, data in graph.nodes.data():
        node_mat[v, 0] = 0
    dgl_graph.ndata['feat'] = node_mat
    graphs = [{'dG': dgl_graph, 'G': graph}]
    return graphs


def sample_pattern(gcn, graphs):
    perms, _, _ = gcn(graphs, m=720)
    perms_str = [''.join(map(str, p[0])) for p in perms]
    perm_freq = Counter(perms_str)
    return perm_freq, perms

def isomorphic_dict(perms, graph):
    len_node = len(graph)
    subgraphs = {i: {} for i in range(2,len_node+1)}
    occurences = []
    for perm in perms:
        occurences.append([])
        perm = perm[0]
        for i in range(2, len_node+1):
            subgraph = graph.subgraph(perm[:i])
            identifier = weisfeiler_lehman_graph_hash(subgraph, iterations=5)
            occurences[-1].append(identifier)
            if identifier not in subgraphs[i]:
                subgraphs[i][identifier]=subgraph
    path_uniques = ['-'.join(occurence) for occurence in occurences]
    path_frequency = Counter(path_uniques)
    subgraph_frequency = Counter(sum(occurences, []))
    return path_frequency, subgraphs, subgraph_frequency
if __name__ == '__main__':
    # gcn

    fname = os.path.join("/mnt/output/GraphRNN_caveman_small_gat_nobfs_2020_12_21_18_00_40", "model_save", "epoch_241.dat") # good
    grnn, gcn, processor, args, feature_map = get_model(fname)
    with torch.no_grad():

        with open('/home/golf/code/graph_generation/datasets/caveman_small/graphs/graph400.dat', 'rb') as f:
            graph = pickle.load(f)
        # graph = nx.grid_2d_graph(3,3)
        # graph = nx.cycle_graph(10)
        # graph = nx.path_graph(10)
        # nodes = list(graph.nodes())
        # node_mapping = {nodes[i]: i for i in range(len(nodes))}
        # graph = nx.relabel_nodes(graph, node_mapping)
        data = build_graph(graph)
        perm_counter, perms = sample_pattern(gcn, data)
        path_frequency, subgraphs, subgraph_frequency = isomorphic_dict(perms, graph)
        snapshot = {
            'path_frequency':path_frequency,
            'subgraphs':subgraphs,
            'subgraph_frequency':subgraph_frequency
        }
        import pickle
        pickle.dump(snapshot, open("q_samples_caveman.pkl",'wb'))

