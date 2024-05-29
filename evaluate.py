import numpy as np
import random
from statistics import mean
from metrics.likelihood import model_likelihood
from metrics.stats_mmd import mmd 
import pandas as pd


def evaluate(args, p_model, q_model, dataset_test):
    """
    This function evalutes the generative model by computing MMD metrics from 
    generated graphs and its log-likelihood estimation over the test set. 
    """
    #generate graphs
    graphs = p_model.sample(args.count)

    graphs_test_indices = [i for i in range(len(dataset_test))]

    node_count_avg_ref, node_count_avg_pred = [], []
    edge_count_avg_ref, edge_count_avg_pred = [], []

    degree_mmd, clustering_mmd, orbit_mmd = [], [], []

    for i in range(0, len(graphs), args.metric_eval_batch_size):
        batch_size = min(args.metric_eval_batch_size,
                         len(graphs) - i)

        graphs_ref_indices = random.sample(graphs_test_indices, batch_size)
        graphs_ref = [dataset_test[i] for i in  graphs_ref_indices]

        graphs_pred = graphs[i: i + batch_size]

        # record basic statistics from test graphs and generated graphs
        node_count_avg_ref.append(mean([len(G.nodes()) for G in graphs_ref]))
        node_count_avg_pred.append(mean([len(G.nodes()) for G in graphs_pred]))

        edge_count_avg_ref.append(mean([len(G.edges()) for G in graphs_ref]))
        edge_count_avg_pred.append(mean([len(G.edges()) for G in graphs_pred]))

        # compute mmd metrics
        batch_degree_mmd, batch_clustering_mmd, batch_orbit_mmd = mmd(graphs_ref, graphs_pred)

        degree_mmd.append(batch_degree_mmd)
        clustering_mmd.append(batch_clustering_mmd)
        orbit_mmd.append(batch_orbit_mmd)


    # print('Evaluating {}, run at {}, epoch {}'.format(
    #     args.fname, args.time, args.num_epochs))

    print_stats(
        node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
        degree_mmd, clustering_mmd, orbit_mmd)

    save_stats(
        node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref,
        edge_count_avg_pred, degree_mmd, clustering_mmd, orbit_mmd, args)


    graph_likelihood = model_likelihood(args, p_model, q_model, dataset_test, args.sample_size_likelihood)

    print('Estimated log likelihood per graph is:')
    print(graph_likelihood)


def print_stats(
    node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref,
    edge_count_avg_pred, degree_mmd, clustering_mmd, orbit_mmd
):
    print('Node count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(node_count_avg_ref), mean(node_count_avg_pred)))
    print('Edge count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(edge_count_avg_ref), mean(edge_count_avg_pred)))

    print('MMD Degree - {:.6f}, MMD Clustering - {:.6f}, MMD Orbits - {:.6f}'.format(
        mean(degree_mmd), mean(clustering_mmd), mean(orbit_mmd)))



#save the rsult to the csv
def save_stats(
    node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref,
    edge_count_avg_pred, degree_mmd, clustering_mmd, orbit_mmd, args):
    stats = {}
    stats['Node count avg Test'] = np.array([mean(node_count_avg_ref)])
    stats['Node count avg Generated'] = np.array([mean(node_count_avg_pred)])

    stats['Edge count avg avg Test'] = np.array([mean(edge_count_avg_ref)])
    stats['Edge count avg Generated'] = np.array([mean(edge_count_avg_pred)])

    stats['MMD Degree'] = np.array([mean(degree_mmd)])
    stats['MMD Clustering'] = np.array([mean(clustering_mmd)])
    stats['MMD Orbits'] = np.array([mean(orbit_mmd)])
    print(stats)
    hist = pd.DataFrame.from_dict(stats)
    hist.to_csv('output/'+ args.fname+ '/stats.csv')

