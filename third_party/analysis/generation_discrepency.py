
from models.graph_rnn.train import predict_graphs as gen_graphs_graph_rnn
import torch
from utils import load_model, get_model_attribute
import networkx as nx
from datasets.preprocess import get_bfs_seq


class ArgsEvaluate():
    def __init__(self, name, epoch):
        # Can manually select the device too
        '''

        :param name: str, output dir for a certain exp
        :param epoch: int, epoch to evaluate
        '''

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model_path = name + '/model_save' + "/epoch_" +str(epoch) + ".dat"

        self.train_args = get_model_attribute(
            'saved_args', self.model_path, self.device)

        self.num_epochs = get_model_attribute(
            'epoch', self.model_path, self.device)

        # Whether to generate networkx format graphs for real datasets
        self.generate_graphs = True

        # Number of graphs to produce from the model
        self.count = 100
        self.batch_size = 10  # Must be a factor of count

        self.metric_eval_batch_size = 256



        # Specific to GraphRNN
        self.min_num_node = 0
        self.max_num_node = 50

        #Set for Likelihood
        self.need_llh = True
        self.llh_sample = 1000 #number of sample to estimate true likelihood
        self.test_number = 1  #number of test graphs to be estimated for likelihood



        self.graphs_save_path = 'output/' + name + '/generated_graphs/'
        self.current_graphs_save_path = self.graphs_save_path


def is_bfs(G, node_orders):
    tree = {}
    vis = []

    def valid_bfs(v):
        q = []
        q.append(v[0])
        vis.append(v[0])
        while q:
            s = q.pop(0)
            candidates = []
            for neighbour in tree[s]:
                if neighbour not in vis:
                    candidates.append(neighbour)
            if set(vis+candidates) != set(v[:len(vis+candidates)]):
                return False
            candidates = v[len(vis):len(vis+candidates)]
            vis.extend(candidates)
            q.extend(candidates)
        return True


    for edge in G.edges():
        if edge[0] in tree:
            tree[edge[0]].append(edge[1])
        else:
            tree[edge[0]] = [edge[1]]
        if edge[1] in tree:
            tree[edge[1]].append(edge[0])
        else:
            tree[edge[1]] = [edge[0]]

    return valid_bfs(node_orders)


if __name__ == '__main__':

    eval_args = ArgsEvaluate(name='/mnt/output/GraphRNN_caveman_small_gat_nobfs_2020_12_21_18_00_40', epoch=241)
    gen_graphs = gen_graphs_graph_rnn(eval_args)
    # graph = gen_graphs
    count = 0.
    for graph in gen_graphs:
        # bfs_order = get_bfs_seq(graph, 0) # sanity check
        # has_bfs = is_bfs(graph, bfs_order))
        node_orders = list(graph.nodes())
        print(node_orders)
        count += is_bfs(graph, node_orders)
    print(count/len(gen_graphs))
