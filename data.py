import time
import random
import pickle
import torch
from torch.utils.data import Dataset
import dgl
import os
import numpy as np
import networkx as nx
import pandas as pd
from torch.utils.data import DataLoader
from datasets.process_dataset import load_graph_dataset
from datasets.preprocess import calc_max_prev_node

class Graph_from_file(Dataset):
    # TODO implement dataset

    def __init__(self, args, graph_list, feature_map):
        print('Reading graphs from fiels...')
        self.dataset_path = args.current_processed_dataset_path
        self.graph_list = graph_list
        self.feature_map = feature_map

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        with open(self.dataset_path + 'graph' + str(self.graph_list[idx]) + '.dat', 'rb') as f:
            G = pickle.load(f)
        f.close()


        dgl_G = dgl.from_networkx(nx_graph=G)


        #set attribute for gcn
        n = len(G.nodes())
        len_node_vec = len(self.feature_map['node_forward'])
        node_mat = torch.zeros((n, 1), requires_grad=False)
        node_map = self.feature_map['node_forward']

        for v, data in G.nodes.data():
            ind = node_map[data['label']]
            node_mat[v, 0] = ind   #1?



        dgl_G.ndata['feat'] = node_mat

        return {"G": G, "dG": dgl_G}

    def collate_batch(self, batch):
        return batch

class NumpyTupleDataset(Dataset):
    """Dataset of a tuple of datasets.

        It combines multiple datasets into one dataset. Each example is represented
        by a tuple whose ``i``-th item corresponds to the i-th dataset.
        And each ``i``-th dataset is expected to be an instance of numpy.ndarray.

        Args:
            datasets: Underlying datasets. The ``i``-th one is used for the
                ``i``-th item of each example. All datasets must have the same
                length.

        """

    def __init__(self, datasets, graph_list, feature_map, property):

        if not datasets:
            raise ValueError('no datasets are given')
        if not graph_list:
            length = len(datasets[0])  # 133885
        else:
            length = len(graph_list)
        for i, dataset in enumerate(datasets):
            if len(dataset) != length:
                raise ValueError(
                    'dataset of the index {} has a wrong length'.format(i))
        # Initialization
        self._datasets = datasets
        self._length = length
        # self._features_indexer = NumpyTupleDatasetFeatureIndexer(self)
        # self.filepath = filepath
        self.transform = self.np_to_nx

        self.graph_list = graph_list
        self.feature_map = feature_map
        self.property = property
        self.cur_cum = 0
        self.batch_size = None
        self.cur_num = 1
        self.gather()

    def gather(self):
        n_node_per_graphs = np.sum(self._datasets[0]!=0, axis=1)
        self.graph_group = np.split(np.arange(self._length), np.unique(n_node_per_graphs, return_index=True)[1][1:])
        self.min_num, self.max_num = np.min(n_node_per_graphs),  np.max(n_node_per_graphs)
        self._datasets[0] = self._datasets[0][self.graph_group[-1]]
        self._datasets[1] = self._datasets[1][self.graph_group[-1]]
        self._datasets[2] = self._datasets[2][self.graph_group[-1]]


    # def set_batch_size(self, batch_size):
    #     self.batch_size = batch_size

    def __len__(self):
        return self._length

    def getitem_v2(self, index):

        node, adj, _ = [dataset[index] for dataset in self._datasets]
        G = self.transform(node, adj)
        dgl_G = dgl.from_networkx(nx_graph=G)

        # set attribute for gcn
        n = len(G.nodes())
        len_node_vec = len(self.feature_map['node_forward'])
        node_mat = torch.zeros((n, 1), requires_grad=False)
        node_map = self.feature_map['node_forward']

        for v, data in G.nodes.data():
            ind = node_map[data['label']]

            node_mat[v, 0] = ind  # 1?

        dgl_G.ndata['feat'] = node_mat
        prop = self.property[index]

        if self.cur_cum + 1 == self.batch_size:
            self.cur_cum = 0
            self.cur_num = np.random.randint(self.min_num, self.max_num)

        return {"G": G, "dG": dgl_G, 'property': prop}

    def __getitem__(self, index):
        if not self.feature_map:
            batches = [dataset[index] for dataset in self._datasets]
            if isinstance(index, (slice, list, np.ndarray)):
                length = len(batches[0])
                batches = [tuple([batch[i] for batch in batches])
                        for i in range(length)]   # six.moves.range(length)]
            else:
                batches = tuple(batches)
        else:
            batches = self.getitem_v2(index)
        return batches


    def get_datasets(self):
        return self._datasets


    @classmethod
    def save(cls, filepath, numpy_tuple_dataset):

        np.savez(filepath, *numpy_tuple_dataset._datasets)
        print('Save {} done.'.format(filepath))

    @classmethod
    def load(cls, filepath, graph_list=None, feature_map=None):
        print('Loading file {}'.format(filepath))
        if not os.path.exists(filepath):
            raise ValueError('Invalid filepath {} for dataset'.format(filepath))
            # return None
        load_data = np.load(filepath)
        if 'qm9' in filepath:
            load_prop = pd.read_csv('./datasets/Qm9/qm9_property.csv')
        elif 'zinc' in filepath:
            load_prop = pd.read_csv('./datasets/Zinc/zinc250k_property.csv')
        else:
            raise FileNotFoundError
        load_prop = load_prop.to_numpy()[:, :2].astype(np.float32)
        result = []
        i = 0
        while True:
            key = 'arr_{}'.format(i)
            if key in load_data.keys():
                result.append(load_data[key])
                i += 1
            else:
                break
        return cls(result, graph_list, feature_map, load_prop)

    @staticmethod
    def np_to_nx(node, adj):
        indices = np.where(node!=0)[0]
        node = node[indices]
        adj = adj[:, indices][:,:, indices].astype(np.uint8)
        adj_no_label = np.sum(adj, axis=0)
        nx_G = nx.from_numpy_matrix(adj_no_label, create_using=nx.Graph)
        for i, label in enumerate(node):
            nx_G.nodes[i]['label'] = label
        edge_types, forward_nodes, backward_nodes = np.where(adj!=0)
        for edge_type in edge_types:
            for forward_node, backward_node in zip(forward_nodes, backward_nodes):
                nx_G[forward_node][backward_node]['label'] = edge_type
        return nx_G
    @staticmethod
    def collate_batch(batch):
        return batch

    @staticmethod
    def nx_to_np(nx_G, n, k):
        ## Qm9: ((9,), (4,9,9), (15,))
        n_node = len(nx_G)
        adj = np.zeros((k, n, n), dtype=np.int32)
        node_array = np.zeros(n)
        for edge in nx.to_edgelist(nx_G):
            i, j, e_type = edge[0], edge[1], edge[2]['label']
            adj[e_type, i, j] = 1
            adj[e_type, j, i] = 1

        for i, node in nx_G.nodes.data():
            node_array[i] = node['label']

        # node_array = np.array([node_data[i]['label'] for i in range(n_node)])

        ## Zinc

        return node_array, adj




def create_dataset(args):

    graph_dataset = load_graph_dataset(args)

    #if args.nobfs:
    #    args.max_prev_node = feature_map['max_nodes'] - 1
    #if args.max_prev_node is None:
    #    args.max_prev_node = calc_max_prev_node(args.current_processed_dataset_path)

    #args.max_head_and_tail = None
    #print('max_prev_node:', args.max_prev_node)

    #random.shuffle(graphs)
    #graphs_train = graphs[: int(0.80 * len(graphs))]
    #graphs_validate = graphs[int(0.80 * len(graphs)): int(0.90 * len(graphs))]
    #graphs_test = graphs[int(0.90 * len(graphs)): ]


    dataset_train, dataset_validate, dataset_test = dgl.data.utils.split_dataset(graph_dataset, frac_list=[0.8, 0.1, 0.1])

    # show graphs statistics


    print('Device:', args.device)
    print('Graph type:', args.dataset)


    if args.task == "train":
        return dataset_train, dataset_validate

    elif args.task == "test":
        return dataset_test
    
    else:
        raise Exception("No such task in args.task: " + str(args.task))
