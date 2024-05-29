import os 
import json
import numpy as np
import torch
import dgl
from args import Args
from utils import create_dirs,load_model
from models.DAGG.model import DAGG
from models.Rout.model import Rout
from train import train
from evaluate import evaluate
import datasets.process_dataset as gdata

if __name__ == '__main__':

    # preparation for model traing 
    args = Args()
    args = args.update_args()
    create_dirs(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    graph_dataset = gdata.load_graph_dataset(args)
    data_statistics = gdata.get_data_statistics(graph_dataset)


    dataset_train, dataset_validate, dataset_test = dgl.data.utils.split_dataset(graph_dataset, frac_list=[0.8, 0.1, 0.1])


    if args.task == "train":

        # prepare the data

        dataloader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.num_workers, collate_fn=lambda _: _)

        dataloader_validate = torch.utils.data.DataLoader(
            dataset_validate, batch_size=args.batch_size, shuffle=False, drop_last=True,
            num_workers=args.num_workers, collate_fn=lambda _: _)


        # save args
        with open(os.path.join(args.experiment_path, "configuration.txt"), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # the autoregressive graph generative model
        p_model = DAGG(args, data_statistics).to(args.device)

        # the q distributions of node orders given training graphs  
        q_model = Rout(args, data_statistics).to(args.device)

        # minimize the variational lower bounds of training graphs under the p model
        train(args, p_model, q_model, dataloader_train, dataloader_validate)
        
    elif args.task == "evaluate":


        # load the p and q models
        p_model,qmodel = load_model(args, args.eval_epoch)


        # load test set.
        graph_dataset = gdata.load_graph_dataset(args)
        dataset_train, dataset_validate, dataset_test = dgl.data.utils.split_dataset(graph_dataset, frac_list=[0.8, 0.1, 0.1])
        
        # compute MMD values from multiple graphs statistics 
        # compute the approximate log-likelihood from importance sampling
        
        evaluate(args, p_model, qmodel, dataset_train)

    else:

        raise Exception("No such task in args.task:" + args.task)

