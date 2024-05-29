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

from likelihood import _get_log_likelihood, _statistic

# class iso_graph(nx.Graph):
#     def __init__(self):
#         super().__init__()
#     def __eq__(self, other):
#         return isomorphism.GraphMatcher(other, self).is_isomorphic()
# load model
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

# construct the graph you want to intepret here!
def build_graph(n_node):
    path = nx.cycle_graph(n_node)
    default_label_graph(path)
    dgl_path = dgl.from_networkx(path)

    node_mat = torch.zeros((n_node, 1), requires_grad=False)
    for v, data in path.nodes.data():
        node_mat[v, 0] = 0
    dgl_path.ndata['feat'] = node_mat
    graphs = [{'dG': dgl_path, 'G': path}]
    return graphs



# interpretation
def sample_pattern(gcn, graphs):
    perms, _, _ = gcn(graphs, m=720)
    perms_str = [''.join(map(str, p[0])) for p in perms]
    perm_freq = Counter(perms_str)
    return perm_freq, perms


# compute elbo
def compute_elbo_gcn(graphs, grnn, q, m, args, feature_map, processor):
    perms, ll_q, log_rep = q(graphs, m=m)
    data = [processor(graphs[0]['G'], perm[0]) for perm in perms]
    data = collate(data)

    nll_p = eval_loss_graph_rnn(args, grnn, data, feature_map)
    ll_p_hat = -nll_p - log_rep
    elbo = torch.mean(ll_p_hat.detach()-ll_q.detach())
    return elbo.item()

def compute_elbo_unif(graphs, grnn, m, n_node, args, feature_map, processor):
    all_perms = list(map(list, itertools.permutations(list(range(n_node)))))
    perms = random.sample(all_perms, m)
    # def exact_rep(perm):
    log_rep = torch.stack([gcn.rep_computer.compute_repetition(graphs[0]['G'], perm) for perm in perms]).cuda()
    data = [processor(graphs[0]['G'], perm) for perm in perms]
    data = collate(data)
    nll_p = eval_loss_graph_rnn(args, grnn, data, feature_map)
    ll_p_hat = -nll_p - log_rep
    ll_q = torch.log(torch.ones(m)/len(all_perms)).cuda()
    elbo = torch.mean(ll_p_hat.detach()-ll_q.detach())
    return elbo.item()


def compute_exact_like(graphs, grnn, n_node, args, feature_map, processor):
    perms = list(map(list, itertools.permutations(list(range(n_node)))))
    log_rep = torch.stack([gcn.rep_computer.compute_repetition(graphs[0]['G'], perm) for perm in perms]).cuda()
    data = [processor(graphs[0]['G'], perm) for perm in perms]
    data = collate(data)
    nll_p = eval_loss_graph_rnn(args, grnn, data, feature_map)
    log_like = torch.logsumexp(-nll_p - log_rep, dim=0)
    return log_like.item()


def compute_approx_like(graphs, grnn, q, m, n_node, args, feature_map, processor):
    # m = 6400
    perms, ll_q, log_rep = q(graphs, m=m)
    data = [processor(graphs[0]['G'], perm[0]) for perm in perms]
    data = collate(data)

    nll_p = eval_loss_graph_rnn(args, grnn, data, feature_map)
    ll_p_hat = -nll_p - ll_q - log_rep

    ll_p_hat = torch.logsumexp(ll_p_hat, dim=1) - torch.log(torch.tensor(m, dtype=torch.float)) #+ torch.tensor(scipy.special.gammaln(n_node+1))
    return ll_p_hat.item()


def isomorphic_dict(perms, graph):
    # generate all possible subgraph
    len_node = len(graph)
    subgraphs = {i: {} for i in range(2,len_node+1)}
    occurences = []
    for perm in perms:
        occurences.append([])
        perm = perm[0]
        for i in range(2, len_node+1):
            subgraph = graph.subgraph(perm[:i])
            identifier = weisfeiler_lehman_graph_hash(subgraph)
            occurences[-1].append(identifier)
            if identifier not in subgraphs[i]:
                subgraphs[i][identifier]=subgraph
    path_uniques = ['-'.join(occurence) for occurence in occurences]
    path_freq = Counter(path_uniques)

    pass
if __name__ == '__main__':
    # gcn
    print("model trained with gcn")

    fname = os.path.join("/mnt/output/GraphRNN_caveman_small_gat_nobfs_2020_12_21_18_00_40", "model_save", "epoch_241.dat") # good
    grnn, gcn, processor, args, feature_map = get_model(fname)
    with torch.no_grad():
        # for n_node in [6, 7, 8, 9, 10]:

        # print("path graph with n_node = {}".format(n_node))
        graphs = build_graph(10)
        # print(g)
        print(graphs[0]['G'])
        perm_counter, perms = sample_pattern(gcn, graphs)
        isomorphic_dict(perms, graphs[0]['G'])
        # print("sampled paths:", sample_pattern(gcn, graphs))
    #         print("approx loglike: ", compute_approx_like(graphs, grnn, gcn, 64, n_node, args, feature_map, processor))
    #         print("approx loglike: ", compute_approx_like(graphs, grnn, gcn, 640, n_node, args, feature_map, processor))
    #         print("approx loglike: ", compute_approx_like(graphs, grnn, gcn, 6400, n_node, args, feature_map, processor))
    #         print("approx loglike: ", compute_approx_like(graphs, grnn, gcn, 12800, n_node, args, feature_map, processor))
    #         if n_node<=8:
    #             print("exact loglike: ", compute_exact_like(graphs, grnn, n_node, args, feature_map, processor))
    #         else:
    #             print("exact loglike: ", None)
    #         print("gcn elbo: ", compute_elbo_gcn(graphs, grnn, gcn, 64, args, feature_map, processor))
    #         print("unif elbo: ", compute_elbo_unif(graphs, grnn, 64, n_node, args, feature_map, processor))
    # print("")
    # print("model trained with uniform")
    # #uniform
    # fname = os.path.join("/home/golf/code/graph_generation/output/GraphRNN_path_unif_nobfs_2020_12_20_18_41_40", "model_save", "epoch_1000.dat") # good
    # grnn, gcn, processor, args, feature_map = get_model(fname)
    # with torch.no_grad():
    #     for n_node in [5, 6, 7, 8, 9, 10]:
    #         print("path graph with n_node = {}".format(n_node))
    #         graphs = build_graph(n_node)
    #         print("approx loglike: ", compute_approx_like(graphs, grnn, n_node, args, feature_map, processor))
    #         if n_node<8:
    #             print("exact loglike: ", compute_exact_like(graphs, grnn, n_node, args, feature_map, processor))
    #         else:
    #             print("exact loglike: ", None)
    #         print("unif elbo: ", compute_elbo_unif(graphs, grnn, 64, n_node, args, feature_map, processor))
