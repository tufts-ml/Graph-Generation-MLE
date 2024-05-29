import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import numpy as np
import math
from models.Rout.gnn.layer.gcn_layer import GCNLayer
from models.Rout.gnn.layer.mlp_readout_layer import MLPReadout


class GCNNet(nn.Module):

    def __init__(self, args, node_len, out_dim):
        super().__init__()
        in_dim_node = node_len  # node_dim (feat is an integer)
        hidden_dim = args.gnn_hidden_dim
        out_dim = out_dim
        in_feat_dropout = args.gnn_in_feat_dropout
        dropout = args.gnn_dropout
        n_layers = args.gnn_num_layers

        self.batch_norm = args.gnn_batch_norm
        self.residual = args.gnn_residual
        self.n_classes = args.gnn_n_classes
        self.device = args.device
        self.d_model = hidden_dim

        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)  # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.first_gnn = GCNLayer(in_dim_node, hidden_dim, F.relu, dropout,
                                              self.batch_norm, self.residual)
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
                                              self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.layers.append(GCNLayer(hidden_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, out_dim)

    def positionalencoding(self, lengths, permutations):
        # length = sum([len(perm) for perm in permutations])
        l_t = len(permutations[0])
        # pes = [torch.zeros(length, self.d_model) for length in lengths]
        pes = torch.split(torch.zeros((sum(lengths), self.d_model), device=self.device), lengths)
        # print(pes[0].device)
        position = torch.arange(0, l_t, device=self.device).unsqueeze(1) + 1
        div_term = torch.exp((torch.arange(0, self.d_model, 2, dtype=torch.float, device=self.device) *
                              -(math.log(10000.0) / self.d_model)))
        # test = torch.sin(position.float() * div_term)
        for i in range(len(lengths)):
            pes[i][permutations[i], 0::2] = torch.sin(position.float() * div_term)
            pes[i][permutations[i], 1::2] = torch.cos(position.float() * div_term)

        pes = torch.cat(pes)
        return pes

    def forward(self, g, h, p=None):
        # p: positional(order) embedding
        # input embedding

        h = self.first_gnn(g,torch.squeeze(h))


        # GCN
        for conv in self.layers:
            h = conv(g, h)

        # output
        h_out = self.MLP_layer(h)
        # h_out = F.softmax(h_out, dim=0)
        return torch.squeeze(h_out)