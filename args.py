from datetime import datetime
import torch
from utils import get_model_attribute
import argparse
import os

class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # Logging & model saving
        self.parser.add_argument('--task', default='train', help='train or evaluate the model')
        self.parser.add_argument('--clean_tensorboard', action='store_true', help='Clean tensorboard')
        self.parser.add_argument('--clean_temp', action='store_true', help='Clean temp folder')
        self.parser.add_argument('--log_tensorboard', action='store_true', help='Whether to use tensorboard for logging')
        self.parser.add_argument('--save_model', default=True, action='store_true', help='Whether to save model')
        self.parser.add_argument('--print_interval', type=int, default=1, help='loss printing batch interval')
        self.parser.add_argument('--epochs_save', type=int, default=1, help='model save epoch interval')
        self.parser.add_argument('--epochs_validate', type=int, default=1, help='model validate epoch interval')

        # Setup
        self.parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='cuda:[d] | cpu')
        self.parser.add_argument('--enable_qdist', default=True, action='store_true', help='Whether to q or uniform distribution for sampling')


        self.parser.add_argument('--seed', type=int, default=123, help='random seed to reproduce performance/dataset')

        # Dataset specification
        self.parser.add_argument('--dataset', default='caveman_small', help='Select dataset to train the model')
        self.parser.add_argument('--num_graphs', type=int, default=None, help='take complete dataset(None) | part of dataset')
        self.parser.add_argument('--produce_graphs', default=True, action='store_true', help='Whether to reproduce graphs from a known distribution (e.g. SBM)')
        self.parser.add_argument('--label', default=False, action='store_true', help='Whether to use label infomation in dataset')

        #Specific to transformer in DAGG generative model 
        self.parser.add_argument('--embd_pdrops', type=float, default=0.1,help='dropout rate')
        self.parser.add_argument('--position', default=False, action='store_true',
                                 help='Whether to use postion')
        self.parser.add_argument('--n_ctx', type=int, default=401, help='?')
        self.parser.add_argument('--n_embd', type=int, default=512, help='node embedding dimension')
        self.parser.add_argument('--layer_norm_epsilon', type=float, default=1e-5, help='cd gralayer_norm_epsilon')
        self.parser.add_argument('--n_layer', type=float, default=2, help='layer of attention')
        self.parser.add_argument('--withz', default=False, help='use z or not')
        self.parser.add_argument('--embd_pdrop', type=float, default=0.0, help='embd drop out rate')
        self.parser.add_argument('--n_head', type=int, default=4, help='transformer head')
        self.parser.add_argument('--attn_pdrop', type=float, default=0.0, help='attention drop rate')
        self.parser.add_argument('--resid_pdrop', type=float, default=0.0, help='residual drop rate')
        self.parser.add_argument('--activation_function', type=str, default="relu", help='activation type in transformer')
        self.parser.add_argument('--initializer_range', type=float, default=0.02,
                                 help='activation type in transformer')
        self.parser.add_argument('--hidden_size_node_level_transformer', type=int, default=256,
                                 help='hidden size for node level transformer')
        self.parser.add_argument('--embedding_size_node_level_transformer', type=int, default=256,
                                 help='the size for node level transformer input')
        self.parser.add_argument('--embedding_size_node_output', type=int, default=256,
                                 help='the size of node output embedding')
        self.parser.add_argument('--hidden_size_edge_level_transformer', type=int, default=256,
                                 help='hidden size for edge level transformer')
        self.parser.add_argument('--embedding_size_edge_level_transformer', type=int, default=256,
                                 help='the size for edge level transformer input')
        self.parser.add_argument('--embedding_size_edge_output', type=int, default=256,
                                 help='the size of edge output embedding')
        self.parser.add_argument('--num_layers', type=int, default=2, help='layers of rnn')
        self.parser.add_argument('--nobfs', default=True, action='store_true',
                                 help='whether to use bfs constraint during sampling')

        # Specify to the GNN for the q distribution
        self.parser.add_argument('--q_gnn_type', default='gcn', help='type of GNN for q model: { gat | gcn | appnp}')
        self.parser.add_argument('--gnn_hidden_dim', type=int, default=256, help='gnn hidden dimension')
        self.parser.add_argument('--gnn_out_dim', type=int, default=32, help='gnn output dimension')
        self.parser.add_argument('--gnn_in_feat_dropout', type=float, default=0.0, help='gnn input feature dropout rate')
        self.parser.add_argument('--gnn_dropout', type=float, default=0.0, help='gnn hidden feature dropout rate')
        self.parser.add_argument('--gnn_num_layers', type=int, default=3, help='number of layers of gnn')
        self.parser.add_argument('--gnn_batch_norm',  default=True, action='store_true', help='whether to use batchnorm in gnn')
        self.parser.add_argument('--gnn_residual',  default=True, action='store_true', help='whether to residual connection in gnn')
        self.parser.add_argument('--gnn_n_classes',  type=int, default=1, help='')
        self.parser.add_argument('--gat_nheads',  type=int, default=6, help='only applied for GAT model')
        self.parser.add_argument('--gnn_pretrain_path', default='', help='petrained gnn path')

        # Specific to sampler of orders from the q distribution
        self.parser.add_argument('--sample_size', type=int, default=2, help='sample size for gradient estimator')
        self.parser.add_argument('--use_mp_sampler',  default=True, action='store_true', help='Whether to multi-process for permutation sampler')
        self.parser.add_argument('--mp_num_workers', type=int, default=4, help='number of workers for permutation sampling')


        # Training config
        self.parser.add_argument('--batch_size', type=int, default=1, help='batchsize')
        self.parser.add_argument('--num_workers', type=int, default=1, help='number of workers for dataloader')
        self.parser.add_argument('--epochs', type=int, default=1001, help='epochs')

        self.parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.3, help='Learning rate decay factor')
        self.parser.add_argument('--clip', default=True, action='store_true', help='whether to use clip gradient for generation model')

        #Evaluation config
        self.parser.add_argument('--record', default='DAGG_Rout',
                                 help='foloder name for evaluation')
        self.parser.add_argument('--eval_epoch', default=400, help='which epoch to evaluate')
        self.parser.add_argument('--count', default=30, help='number of graphs to be sampled')
        self.parser.add_argument('--metric_eval_batch_size', default=30, help='batch size for evaluation')

        # Model load parameters
        self.parser.add_argument('--load_model',  default=False, action='store_true', help='whether to load model')
        self.parser.add_argument('--load_model_path', default='output/exp_name/model_save/', help='load model path')
        self.parser.add_argument('--load_device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='load device: cuda:[d] | cpu')
        self.parser.add_argument('--epochs_end', type=int, default=100, help='model in which epoch to load')
        self.parser.add_argument('--current_dataset_path', default='datasets/caveman_small/graphs/', help='model in which epoch to load')
        self.parser.add_argument('--sample_size_likelihood', type=int, default=10, help='sample size for likelihood estimation')





    def update_args(self):
        """
        Update args when load a trained model: use settings from the saved model 
        """
        args = self.parser.parse_args()
        if args.load_model:
            old_args = args
            fname = os.path.join(args.load_model_path, "model_save", "epoch_{}.dat".format(old_args.epochs_end))
            args = get_model_attribute(
                'saved_args', fname, args.load_device)
            args.device = old_args.load_device
            args.load_model = True
            args.load_model_path = old_args.load_model_path
            args.epochs = old_args.epochs
            args.epochs_end = old_args.epochs_end

            args.clean_tensorboard = False
            args.clean_temp = False
            args.produce_graphs = False

            return args


        args.milestones = [args.epochs//5, args.epochs//5*2, args.epochs//5*3, args.epochs//5*4]  # List of milestones

        args.time = '{0:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())

        bfs = "nobfs" if args.nobfs else "bfs"
        if args.enable_qdist:
            type = args.q_gnn_type
        else:
            type = "unif"

        if args.task == 'train':
            args.fname = 'DAGG' + '_' + args.q_gnn_type + "_" + type + "_" + bfs + "_" + args.time
        elif args.task == 'evaluate':
            args.fname = args.record
        args.dir_input = 'output/'
        args.experiment_path = args.dir_input + args.fname
        args.logging_path = args.experiment_path + '/' + 'logging/'
        args.logging_iter_path = args.logging_path + 'iter_history.csv'
        args.logging_epoch_path = args.logging_path + 'epoch_history.csv'
        args.model_save_path = args.experiment_path + '/' + 'model_save/'
        args.tensorboard_path = args.experiment_path + '/' + 'tensorboard/'
        args.dataset_path = 'datasets/'
        args.temp_path = args.experiment_path + '/' + 'tmp/'

        args.current_model_save_path = args.model_save_path

        #args.load_model_path = None


        args.current_processed_dataset_path = None
        args.current_temp_path = args.temp_path
        args.current_graphs_save_path = args.experiment_path + '/' +'predictions/'

        # noise argument for the barabasi_noise graph
        args.noise = 1
        return args
