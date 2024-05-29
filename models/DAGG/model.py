import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate as collate
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from models.DAGG.helper import get_attributes_len_for_graph_rnn
from datasets.preprocess import get_bfs_seq, get_random_bfs_seq
import numpy as np
import torch.nn.init as init
import networkx as nx
from models.DAGG.attention import AttentionDecoder



EPS = 1e-9


class DAGG(nn.Module):
    """
    This class defines the DAGG model, which is an autoregressive graph generative model. It has two main functions: likelihood 
    calculation and sampling. The first function computes the likelihood of a graph with given node orders, and the second 
    function samples graphs by sequentially add nodes and edges to a graph. 
    """
    def __init__(self, args, data_statistics):
        super().__init__()

        self.args = args
        self.data_statistics = data_statistics

        self.processor = Graph_to_Adj_Matrix(args, data_statistics)

        #args.feature_len = self.processor.feature_len
        len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len_for_graph_rnn(
            data_statistics['num_node_labels'], data_statistics['num_edge_labels'],
            args.max_prev_node, args.max_head_and_tail)

        self.len_node_vec = len_node_vec
        self.len_edge_vec = len_edge_vec
        self.num_nodes_to_consider = num_nodes_to_consider


        self.node_level_transformer = AttentionDecoder(args.embedding_size_node_level_transformer, n_head=4)
        self.edge_level_transformer= AttentionDecoder(args.embedding_size_edge_level_transformer, n_head=4)

        self.embedding_node_to_edge = MLP_Plain(
            input_size=args.hidden_size_node_level_transformer, embedding_size=args.embedding_size_node_level_transformer,
            output_size=args.hidden_size_edge_level_transformer).to(device=args.device)
        self.output_node = MLP_Softmax(
            input_size=args.hidden_size_node_level_transformer, embedding_size=args.embedding_size_node_output,
            output_size=len_node_vec).to(device=args.device)
        self.output_edge = MLP_Softmax(
            args.embedding_size_edge_level_transformer, args.embedding_size_edge_level_transformer, len_edge_vec)


        feature_len = len_node_vec + num_nodes_to_consider * len_edge_vec

        self.node_project =  MLP_Plain(feature_len, args.embedding_size_node_level_transformer, args.embedding_size_node_level_transformer)
        self.edge_project =  MLP_Plain(len_edge_vec, args.embedding_size_edge_level_transformer, args.embedding_size_edge_level_transformer)


    # probability calculation
    def forward(self, graphs, pis):
        '''
        The forward function computes the log-likelihood of the given graph using given node orders. 

        Args:
            g: a dgl graph with `n` nodes
            pis: a Tensor of shape `(k, n)`, containing `k` permutations of graph nodes

        Return: negative log-likelihodd of the given graph using `k` node permutations 
        '''

        data = [self.processor(graphs[0], perms) for perms in pis]
        data = collate(data)

        x_unsorted = data['x'].to(self.args.device)

        x_len_unsorted = data['len'].to(self.args.device)
        x_len_max = max(x_len_unsorted)
        x_unsorted = x_unsorted[:, 0:max(x_len_unsorted), :]

        len_node_vec          = self.len_node_vec          
        len_edge_vec          = self.len_edge_vec          
        num_nodes_to_consider = self.num_nodes_to_consider 

        batch_size = x_unsorted.size(0)
        # sort input for packing variable length sequences
        x_len, sort_indices = torch.sort(x_len_unsorted, dim=0, descending=True)
        x = torch.index_select(x_unsorted, 0, sort_indices)


        # Teacher forcing: Feed the target as the next input
        # Start token for graph level Transformer decoder is node feature second last bit is 1
        node_level_input = torch.cat(
            (torch.zeros(batch_size, 1, x.size(2), device=self.args.device), x), dim=1)

        node_level_input[:, 0, len_node_vec - 2] = 1
        steps = x.size(1)

        x_len = x_len.cpu()
        # #mask
        # mask = torch.tril(torch.ones((batch_size, steps+1, steps+1)))

        # Forward propogation
        node_level_input = self.node_project(node_level_input)
        node_level_output,_,_= self.node_level_transformer(node_level_input)

        # Evaluating node predictions
        x_pred_node = self.output_node(node_level_output)


        edge_mat_packed = pack_padded_sequence(
            x[:, :, len_node_vec: min(
                x_len_max - 1, num_nodes_to_consider) * len_edge_vec + len_node_vec],
            x_len, batch_first=True)

        edge_mat, edge_batch_size = edge_mat_packed.data, edge_mat_packed.batch_sizes

        idx = torch.LongTensor(
            [i for i in range(edge_mat.size(0) - 1, -1, -1)]).to(self.args.device)
        edge_mat = edge_mat.index_select(0, idx)

        edge_mat = edge_mat.reshape(edge_mat.size(0), min(
            x_len_max - 1, num_nodes_to_consider), len_edge_vec)
        edge_level_input = torch.cat(
            (torch.zeros(sum(x_len), 1, len_edge_vec, device=self.args.device), edge_mat), dim=1)
        edge_level_input[:, 0, len_edge_vec - 2] = 1

        # Compute descending list of lengths for y_edge
        x_edge_len = []
        # Histogram of y_len
        x_edge_len_bin = torch.bincount(x_len)
        for i in range(len(x_edge_len_bin) - 1, 0, -1):
            # count how many x_len is above and equal to i
            count_temp = torch.sum(x_edge_len_bin[i:]).item()

            # put count_temp of them in x_edge_len each with value min(i, num_nodes_to_consider + 1)
            x_edge_len.extend([min(i, num_nodes_to_consider + 1)] * count_temp)

        x_edge_len = torch.LongTensor(x_edge_len).to(self.args.device)


        hidden_edge = self.embedding_node_to_edge(node_level_output[:, 0:-1, :])


        hidden_edge = pack_padded_sequence(
            hidden_edge, x_len, batch_first=True).data
        idx = torch.LongTensor(
            [i for i in range(hidden_edge.size(0) - 1, -1, -1)]).to(self.args.device)
        hidden_edge = hidden_edge.index_select(0, idx)

        # Set hidden state for edge-level Transformer
        # shape of hidden tensor (num_layers, batch_size, hidden_size)
        x_edge_len = x_edge_len.cpu()
        hidden_edge = hidden_edge.view(hidden_edge.size(0), 1, hidden_edge.size(1))
        edge_level_input = self.edge_project(edge_level_input)
        # edge_level_input = pack_padded_sequence(
        #         edge_level_input, x_edge_len, batch_first=True, enforce_sorted=False)

        edge_level_input = torch.cat([hidden_edge,edge_level_input], dim=1)

        #x_edge_len = x_edge_len.cpu()
        x_pred_edge,_,_ = self.edge_level_transformer(edge_level_input)
        x_pred_edge = x_pred_edge[:,1:]
        # x_pred_edge, _ = pad_packed_sequence(x_pred_edge, batch_first=True)
        # cleaning the padding i.e setting it to zero
        x_pred_edge = self.output_edge(x_pred_edge)
        x_pred_node = pack_padded_sequence(
            x_pred_node, x_len + 1, batch_first=True)
        x_pred_node, lens_pred_node = pad_packed_sequence(x_pred_node, batch_first=True)
        x_pred_edge = pack_padded_sequence(
            x_pred_edge, x_edge_len, batch_first=True)
        x_pred_edge, lens_pred_edge = pad_packed_sequence(x_pred_edge, batch_first=True)

        x_node = torch.cat(
            (x[:, :, :len_node_vec], torch.zeros(batch_size, 1, len_node_vec, device=self.args.device)), dim=1)
        x_node[torch.arange(batch_size), x_len, len_node_vec - 1] = 1

        x_edge = torch.cat((edge_mat, torch.zeros(
            sum(x_len), 1, len_edge_vec, device=self.args.device)), dim=1)
        x_edge[torch.arange(sum(x_len)), x_edge_len - 1, len_edge_vec - 1] = 1

        loss_node = F.binary_cross_entropy(x_pred_node, x_node, reduction='none')
        loss_edge = F.binary_cross_entropy(x_pred_edge, x_edge, reduction='none')



        loss_node = torch.sum(loss_node, dim=[1, 2])
        edge_batch_size_cum = torch.cat(
            [torch.tensor([0]).to(self.args.device), torch.cumsum(edge_batch_size, dim=0).to(self.args.device)])
        edge_indices = []
        for shift, length in enumerate(x_len):
            edge_indices.append((edge_batch_size_cum + shift)[:length])

        loss_edge = torch.cat(
            [torch.sum(loss_edge.index_select(0, indices), dim=[0, 1, 2]).view(1) for indices in edge_indices])

        loss = loss_node + loss_edge
        swapped_loss = torch.empty_like(loss)
        swapped_loss[sort_indices] = loss

        log_like = - swapped_loss 

        return log_like

    def sample(self, size, batch_size=1):
        '''
        Sample graphs from the DAGG model.
        Args: 
            size: int, the number of graph samples to be generated 
            batch_size: int, batch size for the sampling procedure  

        Return: a list [g1,g2,.....] of `size` graph samples
        '''

        max_num_node = self.data_statistics["max_num_nodes"]
        min_num_node = self.data_statistics["min_num_nodes"]

        len_node_vec          = self.len_node_vec          
        len_edge_vec          = self.len_edge_vec          
        num_nodes_to_consider = self.num_nodes_to_consider 


        feature_len = len_node_vec + num_nodes_to_consider * len_edge_vec

        graphs = []

        for _ in range(size // batch_size):

            x_pred_node = np.zeros(
                (batch_size, max_num_node), dtype=np.int32)
            # [batch_size] * [num of nodes] * [num_nodes_to_consider]
            x_pred_edge = np.zeros(
                (batch_size, max_num_node, num_nodes_to_consider), dtype=np.int32)

            node_level_input = torch.zeros(
                batch_size, 1, feature_len, device=self.args.device)
            # Initialize to node level start token
            node_level_input[:, 0, len_node_vec - 2] = 1
            past=None
            
            for i in range(max_num_node):
                # [batch_size] * [1] * [hidden_size_node_level_rnn]
                node_level_input = self.node_project(node_level_input)
                node_level_output,past,_ = self.node_level_transformer(node_level_input, past)
                # [batch_size] * [1] * [node_feature_len]
                node_level_pred = self.output_node(node_level_output)
                # [batch_size] * [node_feature_len] for torch.multinomial
                node_level_pred = node_level_pred.reshape(
                    batch_size, len_node_vec)
                # [batch_size]: Sampling index to set 1 in next node_level_input and x_pred_node
                # Add a small probability for each node label to avoid zeros
                node_level_pred[:, :-2] += EPS
                # Start token should not be sampled. So set it's probability to 0
                node_level_pred[:, -2] = 0
                # End token should not be sampled if i less than min_num_node
                if i < min_num_node:
                    node_level_pred[:, -1] = 0
                sample_node_level_output = torch.multinomial(
                    node_level_pred, 1).reshape(-1)
                node_level_input = torch.zeros(
                    batch_size, 1, feature_len, device=self.args.device)
                node_level_input[torch.arange(
                    batch_size), 0, sample_node_level_output] = 1

                # [batch_size] * [num of nodes]
                x_pred_node[:, i] = sample_node_level_output.cpu().data

                # [batch_size] * [1] * [hidden_size_edge_level_rnn]
                hidden_edge = self.embedding_node_to_edge(node_level_output)
                #hidden_edge = hidden_edge.view(hidden_edge.size(0), 1, hidden_edge.size(1))

                # [batch_size] * [1] * [edge_feature_len]
                edge_level_input = torch.zeros(
                    batch_size, 1, len_edge_vec, device=self.args.device)
                # Initialize to edge level start token
                edge_level_input[:, 0, len_edge_vec - 2] = 1
                edge_level_input = self.edge_project(edge_level_input)
                edge_level_input = torch.cat([hidden_edge, edge_level_input], dim=1)
                past_e = None
                for j in range(min(num_nodes_to_consider, i)):
                    # [batch_size] * [1] * [edge_feature_len]\
                    if j != 0:
                        edge_level_input = self.edge_project(edge_level_input)
                    #edge_level_input = torch.cat([hidden_edge, edge_level_input], dim=1)
                    edge_level_output,past_e,_ = self.edge_level_transformer(edge_level_input,past_e)
                    edge_level_output = self.output_edge(edge_level_output)
                    # [batch_size] * [edge_feature_len] needed for torch.multinomial
                    edge_level_output = edge_level_output[:,-1].reshape(
                        batch_size, len_edge_vec)

                    # [batch_size]: Sampling index to set 1 in next edge_level input and x_pred_edge
                    # Add a small probability for no edge to avoid zeros
                    edge_level_output[:, -3] += EPS
                    # Start token and end should not be sampled. So set it's probability to 0
                    edge_level_output[:, -2:] = 0
                    sample_edge_level_output = torch.multinomial(
                        edge_level_output, 1).reshape(-1)
                    edge_level_input = torch.zeros(
                        batch_size, 1, len_edge_vec, device=self.args.device)
                    edge_level_input[:, 0, sample_edge_level_output] = 1

                    # Setting edge feature for next node_level_input
                    node_level_input[:, 0, len_node_vec + j * len_edge_vec: len_node_vec + (j + 1) * len_edge_vec] = \
                        edge_level_input[:, 0, :]

                    # [batch_size] * [num of nodes] * [num_nodes_to_consider]
                    x_pred_edge[:, i, j] = sample_edge_level_output.cpu().data

            # Save the batch of graphs
            for k in range(batch_size):
                G = nx.Graph()

                for v in range(max_num_node):
                    # End node token
                    if x_pred_node[k, v] == len_node_vec - 1:
                        break
                    elif x_pred_node[k, v] < self.data_statistics['num_node_labels']:
                        G.add_node(
                            v, label=x_pred_node[k, v])
                    else:
                        raise('Error in sampling node features')

                for u in range(len(G.nodes())):
                    for p in range(min(num_nodes_to_consider, u)):
                        if x_pred_edge[k, u, p] < self.data_statistics['num_edge_labels']:
                            if self.args.max_prev_node is not None:
                                v = u - p - 1
                            elif self.args.max_head_and_tail is not None:
                                if p < self.args.max_head_and_tail[1]:
                                    v = u - p - 1
                                else:
                                    v = p - self.args.max_head_and_tail[1]

                            G.add_edge(
                                u, v, label=x_pred_edge[k, u, p])
                        elif x_pred_edge[k, u, p] == self.data_statistics['num_edge_labels']:
                            # No edge
                            pass
                        else:
                            raise('Error in sampling edge features')

                # Take maximum connected component
                if len(G.nodes()):
                    max_comp = max(nx.connected_components(G), key=len)
                    G = nx.Graph(G.subgraph(max_comp))

                graphs.append(G)

        return graphs


class Graph_to_Adj_Matrix:
    def __init__(self, args, data_statistics):

        print('Generating adj matrix...')
        # No. of previous nodes to consider for edge prediction
        self.max_prev_node = args.max_prev_node
        # Head and tail of adjacency vector to consider for edge prediction
        self.max_head_and_tail = args.max_head_and_tail

        if self.max_prev_node is None and self.max_head_and_tail is None:
            raise Exception('Please provide max_prev_node or max_head_and_tail')

        self.data_statistics = data_statistics


        len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len_for_graph_rnn(
            data_statistics['num_node_labels'], data_statistics['num_edge_labels'],
            self.max_prev_node, self.max_head_and_tail)

        self.len_node_vec = len_node_vec
        self.len_edge_vec = len_edge_vec
        self.num_nodes_to_consider = num_nodes_to_consider

        self.feature_len = len_node_vec + num_nodes_to_consider * len_edge_vec

    def __call__(self, graph, perm=None):

        x_item = torch.zeros((self.data_statistics["max_num_nodes"], self.feature_len))

        adj_feature_mat = self.graph_to_matrix(graph, perm)

        x_item[0:adj_feature_mat.shape[0], :adj_feature_mat.shape[1]] = adj_feature_mat

        return {'x': x_item, 'len': len(adj_feature_mat)}

    def graph_to_matrix(self, in_graph, perm):


        n = len(in_graph.nodes())
        
        len_node_vec          = self.len_node_vec           
        num_nodes_to_consider = self.num_nodes_to_consider

        
        n = len(in_graph.nodes())
        

        order_map = {perm[i]: i for i in range(n)}
        graph = nx.relabel_nodes(in_graph, order_map)


        # 3D adjacecny matrix in case of edge_features (each A[i, j] is a len_edge_vec size vector)
        adj_mat_2d = torch.ones((n, num_nodes_to_consider))
        adj_mat_2d.tril_(diagonal=-1)
        adj_mat_3d = torch.zeros((n, num_nodes_to_consider, self.data_statistics["num_edge_labels"]))

        node_mat = torch.zeros((n, len_node_vec))

        for v, data in graph.nodes.data():
            ind = data['label']
            node_mat[v, ind] = 1

        for u, v, data in graph.edges.data():
            if self.max_prev_node is not None:
                if abs(u - v) <= self.max_prev_node:
                    adj_mat_3d[max(u, v), max(u, v) - min(u, v) -
                               1, data['label']] = 1
                    adj_mat_2d[max(u, v), max(u, v) - min(u, v) - 1] = 0

            elif self.max_head_and_tail is not None:
                if abs(u - v) <= self.max_head_and_tail[1]:
                    adj_mat_3d[max(u, v), max(u, v) - min(u, v) -
                               1, data['label']] = 1
                    adj_mat_2d[max(u, v), max(u, v) - min(u, v) - 1] = 0
                elif min(u, v) < self.max_head_and_tail[0]:
                    adj_mat_3d[max(u, v), self.max_head_and_tail[1] +
                               min(u, v), data['label']] = 1
                    adj_mat_2d[max(u, v), self.max_head_and_tail[1] + min(u, v)] = 0

        adj_mat = torch.cat((adj_mat_3d, adj_mat_2d.reshape(adj_mat_2d.size(
            0), adj_mat_2d.size(1), 1), torch.zeros((n, num_nodes_to_consider, 2))), dim=2)
        adj_mat = adj_mat.reshape((adj_mat.size(0), -1))

        return torch.cat((node_mat, adj_mat), dim=1)


class MLP_Softmax(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Softmax, self).__init__()
        self.mlp = nn.Sequential(
            MLP_Plain(input_size, embedding_size, output_size, dropout),
            nn.Softmax(dim=2)
        )

    def forward(self, input):
        return self.mlp(input)


class MLP_Log_Softmax(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Log_Softmax, self).__init__()
        self.mlp = nn.Sequential(
            MLP_Plain(input_size, embedding_size, output_size, dropout),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, input):
        return self.mlp(input)


class MLP_Plain(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Plain, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embedding_size, output_size),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, input):
        return self.mlp(input)












