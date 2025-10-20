import numpy as np
import networkx as nx


def CAPEntriesFromBaseline(tpm_s):
    """
    baseline state is state 0; returns probability to enter any other state from baseline,
    as a np array of shape (n_states,)
    :param tpm_s: transition probability matrix (n_states x n_states)
    :return: np.array of probabilities
    """
    tpm_s = np.array(tpm_s)
    return tpm_s[0, :]


def CAPExitsToBaseline(tpm_s):
    """
    baseline state is state 0; returns probability to enter baseline from any other state,
    as a np array of shape (n_states,)
    :param tpm_s: transition probability matrix (n_states x n_states)
    :return: np.array of probabilities
    """
    tpm_s = np.array(tpm_s)
    return tpm_s[:, 0]


def CAPResilience(tpm_s):
    """
    for all states but baseline (state 0) returns probability to remain in that state,
    as a np array of shape (n_states-1,)
    :param tpm_s: transition probability matrix (n_states x n_states)
    :return: np.array of probabilities
    """
    tpm_s = np.array(tpm_s)
    return np.diag(tpm_s)



def BaselineResilience(tpm_s):
    """
    returns the probability to remain in baseline state (state 0)
    :param tpm_s:
    :return:
    """
    return tpm_s[0, 0]


def CAPInDegree(tpm_s):
    """
    returns for each state the probability to enter that state from any other state,
    as a np array of shape (n_states,)
    :param tpm_s: transition probability matrix (n_states x n_states)
    :return: np.array of probabilities
    """
    tpm_s = np.array(tpm_s, copy=True)
    np.fill_diagonal(tpm_s, 0)
    return np.sum(tpm_s, axis=0)


def CAPOutDegree(tpm_s):
    """
    returns for each state the probability to exit that state to any other state,
    as a np array of shape (n_states,)
    :param tpm_s: transition probability matrix (n_states x n_states)
    :return: np.array of probabilities
    """
    tpm_s = np.array(tpm_s, copy=True)
    np.fill_diagonal(tpm_s, 0)
    return np.sum(tpm_s, axis=1)

def BetweennessCentrality(tpm_s):
    tpm_s = np.array(tpm_s, copy=True)
    np.fill_diagonal(tpm_s, 0)
    # transform higher probabilities to lower distances

    n = tpm_s.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # Add edges for nonzero probabilities with cost = -log(p)
    rows, cols = np.where( tpm_s > 0)
    probs = tpm_s[rows, cols]
    costs = -np.log(probs)  # higher prob => lower cost

    for i, j, w in zip(rows.tolist(), cols.tolist(), costs.tolist()):
        G.add_edge(i, j, weight=w)

    # Weighted, directed betweenness centrality on shortest paths
    # normalized=True returns values in [0,1]
    bc = nx.betweenness_centrality(G, k=None, normalized=True, weight="weight", endpoints=False)
    return bc

