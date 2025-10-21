import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def CAPEntriesFromBaseline(tpm_s):
    """
    baseline state is state 0; returns probability to enter any other state from baseline,
    as a np array of shape (n_states,)
    :param tpm_s: transition probability matrix (n_states x n_states)
    :return: np.array of probabilities
    """
    tpm_s = np.array(tpm_s)
    return {i:v for i,v in enumerate(tpm_s[0, :])}


def CAPExitsToBaseline(tpm_s):
    """
    baseline state is state 0; returns probability to enter baseline from any other state,
    as a np array of shape (n_states,)
    :param tpm_s: transition probability matrix (n_states x n_states)
    :return: np.array of probabilities
    """
    tpm_s = np.array(tpm_s)
    return {i:v for i,v in enumerate(tpm_s[:, 0])}


def CAPResilience(tpm_s):
    """
    for all states returns probability to remain in that state,
    as a np array of shape (n_states,)
    :param tpm_s: transition probability matrix (n_states x n_states)
    :return: np.array of probabilities
    """
    tpm_s = np.array(tpm_s)
    return {i:v for i,v in enumerate(np.diag(tpm_s))}



def CAPInDegree(tpm_s):
    """
    returns for each state the probability to enter that state from any other state,
    as a np array of shape (n_states,)
    :param tpm_s: transition probability matrix (n_states x n_states)
    :return: np.array of probabilities
    """
    tpm_s = np.array(tpm_s, copy=True)
    np.fill_diagonal(tpm_s, 0)
    return {i:v for i,v in enumerate(np.sum(tpm_s, axis=0))}


def CAPOutDegree(tpm_s):
    """
    returns for each state the probability to exit that state to any other state,
    as a np array of shape (n_states,)
    :param tpm_s: transition probability matrix (n_states x n_states)
    :return: np.array of probabilities
    """
    tpm_s = np.array(tpm_s, copy=True)
    np.fill_diagonal(tpm_s, 0)
    return {i:v for i,v in enumerate(np.sum(tpm_s, axis=1))}

def BetweennessCentrality(tpm_s, graph_plot_savepath=True, self_connections=False):
    tpm_s = np.array(tpm_s, copy=True)

    if not self_connections:
        np.fill_diagonal(tpm_s, 0)

    n = tpm_s.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # Add edges for nonzero probabilities with cost = -log(p)
    rows, cols = np.where( tpm_s > 0)
    probs = tpm_s[rows, cols]
    costs = -np.log(probs)  # higher prob => lower cost

    for i, j, w in zip(rows.tolist(), cols.tolist(), costs.tolist()):
        G.add_edge(i, j, weight=w)

    if graph_plot_savepath:
        weights = [100*np.exp(-d['weight']) for (_, _, d) in G.edges(data=True)]
        # Plot with arrows and width proportional to weight
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(
            G, pos,
            ax=ax,
            with_labels=True,
            node_color='lightblue',
            node_size=700,
            width=weights,
            font_size=10,
            arrowsize=20,
            connectionstyle='arc3,rad=0.1'
        )
        plt.savefig(graph_plot_savepath)
        plt.close(fig)
    # Weighted, directed betweenness centrality on shortest paths
    # normalized=True returns values in [0,1]
    bc = nx.betweenness_centrality(G, k=None, normalized=True, weight="weight", endpoints=False)
    return bc, G

