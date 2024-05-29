import os
import dgl
import torch
import pickle
import matplotlib.pyplot as plt
# from networkx import graphviz_layout
import networkx as nx

import glob

methods = ['gt', 'bfs', 'unif', 'gcn']


fig, axes = plt.subplots(4, 8, figsize=(23,12))
#
plt.subplots_adjust(hspace=2, wspace=2)
for i, method in enumerate(methods):
    graphs = glob.glob("/home/golf/code/graph_generation/metrics/analysis/caveman/{}/*.dat".format(method))
    for j, graph_file in enumerate(graphs):
        if j==0:
            ax = axes[j][i]
            ax.set_ylabel(method, fontsize=24)
        with open(graph_file, 'rb') as f:
            G = pickle.load(f)
        pos = nx.spring_layout(G, seed=3068, k=0.5, threshold=1e-5)
        nx.draw(G, pos, ax=axes[i][j], node_size=30, alpha=0.5)
plt.show()

# graphs = glob.glob("/home/golf/code/graph_generation/output/GraphRNN_caveman_small_gat_nobfs_2020_12_21_18_00_40/generated_graphs/*.dat")
#
# for i, graph_file in enumerate(graphs):
#     with open(graph_file, 'rb') as f:
#         graph = pickle.load(f)
#     nx.draw(graph)
#     plt.savefig(graph_file.replace('dat', 'png'))
#     plt.close()