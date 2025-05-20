import torch
from torch_geometric.data import Data
import random
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data as GraphData
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_s":
            return self.x_s.size(0)
        if key == "edge_index_t":
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def get_sbm(n, nc, ratio, P):
    nbpc = torch.round(n * ratio).type(torch.int64)
    n = torch.sum(nbpc).item()
    C = torch.zeros(n, n)

    if not nc == 1:
        for c1 in range(nc):
            for c2 in range(c1 + 1):
                if c1 == c2:
                    for i in range(torch.sum(nbpc[:c1]), torch.sum(nbpc[: c1 + 1])):
                        for j in range(torch.sum(nbpc[:c2]), i):
                            if torch.rand(1) <= P[c1, c2]:
                                C[i, j] = 1
                else:
                    for i in range(torch.sum(nbpc[:c1]), torch.sum(nbpc[: c1 + 1])):
                        for j in range(torch.sum(nbpc[:c2]), torch.sum(nbpc[: c2 + 1])):
                            if torch.rand(1) <= P[c1, c2]:
                                C[i, j] = 1
    else:
        for i in range(n):
            for j in range(i):
                if torch.rand(1) <= P[0]:
                    C[i, j] = 1
    return C + C.T


def get_graph_sbm(n, nc, ratio, p_intra, p_inter, noise, clusters):
    if nc == 1:
        P = torch.tensor([p_intra])
    elif nc == 2 and clusters == [1, 3]:
        P = torch.tensor([[p_intra, 0.0], [0.0, p_intra]])
    elif nc == 2 and (clusters == [1, 2] or clusters == [2, 3]):
        P = torch.tensor([[p_intra, p_inter], [p_inter, p_intra]])
    elif nc == 3:
        P = torch.tensor(
            [
                [p_intra, p_inter, 0.0],
                [p_inter, p_intra, p_inter],
                [0.0, p_inter, p_intra],
            ]
        )
    C1 = get_sbm(n, nc, ratio, P)
    n = C1.shape[0]

    n_feat = 3  # dimension of the features
    feat_C1 = []  # features

    if nc == 1 and clusters == [1]:
        labels_g1 = torch.zeros(n).type(torch.LongTensor)
    elif nc == 1 and clusters == [2]:
        labels_g1 = torch.ones(n).type(torch.LongTensor)
    elif nc == 1 and clusters == [3]:
        labels_g1 = 2 * torch.ones(n).type(torch.LongTensor)
    elif nc == 2 and clusters == [1, 2]:
        labels1 = torch.zeros(n // 2)
        labels2 = torch.ones(n // 2)
        labels_g1 = torch.hstack([labels1, labels2]).type(torch.LongTensor)
    elif nc == 2 and clusters == [2, 3]:
        labels2 = torch.ones(n // 2)
        labels3 = torch.ones(n // 2) * 2
        labels_g1 = torch.hstack([labels2, labels3]).type(torch.LongTensor)
    elif nc == 2 and clusters == [1, 3]:
        labels1 = torch.zeros(n // 2)
        labels3 = torch.ones(n // 2) * 2
        labels_g1 = torch.hstack([labels1, labels3]).type(torch.LongTensor)

    elif nc == 3:
        labels1 = torch.zeros(n // 3)
        labels2 = torch.ones(n // 3)
        labels3 = torch.ones(n - 2 * (n // nc)) * 2
        labels_g1 = torch.hstack([labels1, labels2, labels3]).type(torch.LongTensor)

    for i in range(n):  # one hot encoding for the features
        feat = torch.zeros(n_feat)
        feat[labels_g1[i]] = 1
        feat_C1.append(feat)

    feat_C1 = torch.stack(feat_C1)
    feat_C1 = feat_C1 + noise * torch.normal(
        torch.zeros(n, n_feat), torch.ones(n, n_feat)
    )  # noise on the features

    edge_index1 = C1.nonzero().t().contiguous()

    G1 = GraphData(
        x=feat_C1,
        edge_index=edge_index1,
        y=labels_g1,
        num_features=n_feat,
        num_classes=3,
    )

    return G1, C1


class SBM(Data):
    def __init__(
        self,
        n_graphs=10000,
        nc1_i=3,
        nc2_i=3,
        clusters1=[1, 2, 3],
        clusters2=[1, 2, 3],
        random_graph=True,
        random_n_nodes=True,
        noise=0.4,
        p_intra=0.6,
        p_inter=0.05,
        n_nodes1=30,
        n_nodes2=30,
    ):
        """
        Parameters
        ----------
        n_pairs : int
            Number of pairs to generate.
        nc1_i : int
            Number of clusters for graph 1.
        nc2_i : int
            Number of clusters for graph 2.
        clusters1 : list
            List of clusters for graph 1.
        clusters2 : list
            List of clusters for graph 2.
        random_graph : bool
            If True, generate random graphs.
        random_n_nodes : bool
            If True, generate random number of nodes.
        noise : float   
            Noise level.
        p_intra : float
            Probability of intra-cluster edges.
        p_inter : float
            Probability of inter-cluster edges.
        n_nodes1 : int
            Number of nodes for graph 1.
        n_nodes2 : int
            Number of nodes for graph 2.
        """
        super().__init__()

        clusters = [[1, 2, 3], [2, 3], [1, 2]]

        n_min = 30
        n_max = 60

        self.all_pairs = []

        for i in range(n_graphs):
            if random_graph:
                noise = np.random.uniform(low=0.0, high=0.6)
                n_nodes1_i = np.random.randint(low=n_min, high=n_max - 1)
                n_nodes2_i = np.random.randint(low=n_min, high=n_max - 1)
            elif random_n_nodes:
                n_nodes1_i = np.random.randint(low=n_min, high=n_max - 1)
                n_nodes2_i = np.random.randint(low=n_min, high=n_max - 1)
            else:
                n_nodes1_i = n_nodes1
                n_nodes2_i = n_nodes2
            if random_graph:
                clusters1_i = random.choice(clusters)
                nc1_i = len(clusters1_i)

                clusters2_i = random.choice(clusters)
                nc2_i = len(clusters2_i)
            else:
                nc1_i = nc1_i
                clusters1_i = clusters1
                nc2_i = nc2_i
                clusters2_i = clusters2

            if nc1_i == 1:
                ratio1 = torch.tensor([1.0])
            if nc1_i == 2:
                ratio1 = torch.tensor([0.5, 0.5])
            if nc1_i == 3:
                ratio1 = torch.tensor([0.333, 0.333, 0.334])

            G1, C1 = get_graph_sbm(
                n_nodes1_i, nc1_i, ratio1, p_intra, p_inter, noise, clusters1_i
            )
            G1_nx = to_networkx(G1, to_undirected=True)
            connected_components = list(nx.connected_components(G1_nx))
            i = 0
            while len(connected_components) > 1:
                if i > 10:
                    raise ValueError("probabilities too low")
                G1, C1 = get_graph_sbm(
                    n_nodes1_i, nc1_i, ratio1, p_intra, p_inter, noise, clusters1_i
                )
                G1_nx = to_networkx(G1, to_undirected=True)
                connected_components = list(nx.connected_components(G1_nx))
                i += 1

            C1_csr = csr_matrix(C1)
            dist_matrix1 = torch.tensor(shortest_path(C1_csr))
            pad1 = torch.zeros((dist_matrix1.shape[0], n_max))
            pad1[:, : dist_matrix1.shape[1]] = dist_matrix1

            if nc2_i == 1:
                ratio2 = torch.tensor([1.0])
            if nc2_i == 2:
                ratio2 = torch.tensor([0.5, 0.5])
            if nc2_i == 3:
                ratio2 = torch.tensor([0.333, 0.333, 0.334])

            G2, C2 = get_graph_sbm(
                n_nodes2_i, nc2_i, ratio2, p_intra, p_inter, noise, clusters2_i
            )
            G2_nx = to_networkx(G2, to_undirected=True)
            connected_components = list(nx.connected_components(G2_nx))
            i = 0
            while len(connected_components) > 1:
                if i > 10:
                    raise ValueError("probabilities too low")
                G2, C2 = get_graph_sbm(
                    n_nodes2_i, nc2_i, ratio2, p_intra, p_inter, noise, clusters2_i
                )
                G2_nx = to_networkx(G2, to_undirected=True)
                connected_components = list(nx.connected_components(G2_nx))
                i += 1

            C2_csr = csr_matrix(C2)
            dist_matrix2 = torch.tensor(shortest_path(C2_csr))
            pad2 = torch.zeros((dist_matrix2.shape[0], n_max))
            pad2[:, : dist_matrix2.shape[1]] = dist_matrix2

            pair = PairData(
                x_s=G1.x,
                edge_index_s=G1.edge_index,
                x_t=G2.x,
                edge_index_t=G2.edge_index,
                y_s=G1.y,
                y_t=G2.y,
                connectivity_s=pad1,
                connectivity_t=pad2,
            )

            self.all_pairs.append(pair)

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        return self.all_pairs[idx]



