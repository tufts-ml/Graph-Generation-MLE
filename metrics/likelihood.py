import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from data import NumpyTupleDataset



def model_likelihood(args, p_model, q_model, graphs, sample_size):
    '''
    This function is to estimate likehood of the given graphs with DAGG.
    Args:
            p_model: nn.Module, generative model
            q_model: nn.Module, inference model
            graphs: [nx.graph()]

        Return:

            pg: Tensor, estiamted log-likelihodd for the graphs
    '''

    dataloader_test= DataLoader(graphs, batch_size=1, shuffle=False, drop_last=False,
        num_workers=args.num_workers, collate_fn=NumpyTupleDataset.collate_batch)

    ll_p, ll_q= _get_log_likelihood(args, dataloader_test, p_model, q_model, sample_size)
    pg = _statistic(ll_p, ll_q)

    return pg

def _get_log_likelihood(args, dataloader_test, p_model, q_model, sample_size):
    with torch.no_grad():
        ll_p = torch.empty((len(dataloader_test), sample_size), device=args.device)
        ll_q = torch.empty((len(dataloader_test), sample_size), device=args.device)
        for id, graph in enumerate(dataloader_test):
            pis, log_q = q_model(graph, sample_size)
            log_joint = p_model(graph, pis)
            ll_p[id] = log_joint
            ll_q[id] = log_q

    return ll_p, ll_q




def _statistic(ll_p,ll_q):

    mllp = torch.logsumexp(ll_p-ll_q, 1) - torch.log(torch.tensor(ll_q.size()[1], device=ll_p.device))
    return torch.mean(mllp).cpu().item()









