import torch
import dgl
import time
import pandas as pd
from collections import defaultdict
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter
from utils import save_model, load_model, get_model_attribute, get_last_checkpoint



# Main training function

def train(args, p_model, q_model, dataloader_train, dataloader_valid):
    """
    maximize the elbo using `p_model` as the generative model $p(A)$ and `q_model` as the inference model $p(\pi | G)$ 
    Args:
       args:
       p_model: nn.Module, the generative model
       q_model: nn.Module, the inference model
       dataloader_train: Dataloader
       dataloader_valid: Dataloader
    """

    optimizer = optim.Adam([{"params": p_model.parameters()}, {"params": q_model.parameters()}], lr=args.lr)


    log_history = defaultdict(list)


    if args.log_tensorboard:
        writer = SummaryWriter(
            log_dir=args.tensorboard_path+ ' ' + args.time, flush_secs=5)
    else:
        writer = None


    for epoch in range(args.epochs):
        # train
        loss= train_epoch(args, p_model, q_model,dataloader_train,optimizer, log_history, epoch)

        epoch += 1

        if args.log_tensorboard:
            writer.add_scalar('{} {} Loss/train'.format(args.note, args.dataset), loss, epoch)

        print('Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))
        if epoch%40 == 0:
            save_model(args, epoch, p_model, q_model)
        print('Model Saved - Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))

        log_history['train_elbo'].append(loss)

        df_iter = pd.DataFrame()
        df_epoch = pd.DataFrame()
        df_iter['batch_elbo'] = log_history['batch_elbo']
        df_iter['batch_time'] = log_history['batch_time']

        df_epoch['train_elbo'] = log_history['train_elbo']

        df_iter.to_csv(args.logging_iter_path, index=False)
        df_epoch.to_csv(args.logging_epoch_path, index=False)


# remove the epoch argument from the argument, and move the print clause out 
def train_epoch(args, p_model, q_model, dataloader_train, optimizer, log_history, epoch):
    """
    One training epoch 
    """

    q_model.train()
    p_model.train()

    train_size = len(dataloader_train)
    total_loss = 0.0

    for batch_id, graphs in enumerate(dataloader_train):

        # currently this function can only train with batch size 1
        # assert batch size is 1
        assert(len(graphs) == 1)

        st = time.time()

        elbo = train_batch(args, p_model, q_model, optimizer, graphs)

        total_loss = total_loss + elbo

        spent = time.time() - st

        #if batch_id % args.print_interval == 0:
        #    print('epoch {} batch {}: elbo is {}, time spent is {}.'.format(epoch, batch_id, elbo,spent), flush=True)

        log_history['batch_elbo'].append(elbo)
        log_history['batch_time'].append(spent)




    avg_loss = total_loss / train_size

    return avg_loss




def train_batch(args, p_model, q_model,optimizer, graph):
    """
        compute the elbo and execute one gradient-descent step 
    """
    # Evaluate model, get costs and log probabilities
    pis, pi_log_likelihood = q_model(graph, args.sample_size)
    
    log_joint = p_model(graph, pis)

    # for the gradient of q dist 
    fake_nll_q = -torch.mean(torch.mean((log_joint.detach() - pi_log_likelihood.detach()) * pi_log_likelihood))

    # for the gradient of p model
    nll_p = -torch.mean(log_joint)

    loss = fake_nll_q + nll_p

    # Perform backward pass and optimization step
    optimizer.zero_grad()

    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    if args.clip == True:
        clip_grad_value_(p_model.parameters(), 1.0)

    optimizer.step()

    elbo = torch.mean(log_joint.detach() - pi_log_likelihood.detach())

    return elbo.item()

def validate_elbo(args, p_model, q_model, dataloader_validate):

    p_model.eval()
    q_model.eval()

    batch_count = len(dataloader_validate)
    with torch.no_grad():
        total_elbo = 0.0

        for _, graphs in enumerate(dataloader_validate):

            log_likelihood, pis = q_model(graphs, args.sample_size, return_pi=True)

            log_joint = p_model(graphs, pis)
            elbo = -torch.mean(log_joint.detach() - log_likelihood.detach())
            total_elbo = total_elbo + elbo

    return total_elbo / batch_count


