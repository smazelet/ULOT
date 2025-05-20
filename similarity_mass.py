#%%
import json
import torch
import pylab as pl
import numpy as np
from torch_geometric.data import Batch
from ulot.utils_data import SBM
from torch_geometric.data import Data 
from ulot.utils import (
    get_filename,
    get_model,
)
from tqdm import tqdm
from sklearn.manifold import  MDS

#%% Pair data class

class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_s":
            return self.x_s.size(0)
        if key == "edge_index_t":
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)

#%% Load the model

param_file="parameter_files/params_SBM_50000.json"

with open(param_file, "r") as file:
    parameters = json.load(file)

model = get_model(parameters, 3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=model.to(device)
path_trained_model = get_filename(parameters, "trained_model")
checkpoint = torch.load(
    path_trained_model, weights_only=False, map_location=torch.device("cpu")
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
#%% Generate SBMs of different class and compute the ULOT transport plan between them

n_graphs=20
dataset_22=SBM(n_graphs=n_graphs,nc1_i=2,nc2_i=2,clusters1=[1,2],clusters2=[1,2], random_graph=False, noise=0.3, p_intra=0.5, p_inter=0.05) 
dataset_22_bis=SBM(n_graphs=n_graphs,nc1_i=2,nc2_i=2,clusters1=[2,3],clusters2=[2,3], random_graph=False, noise=0.3, p_intra=0.5, p_inter=0.05) 
dataset_33=SBM(n_graphs=n_graphs,nc1_i=3,nc2_i=3,clusters1=[1,2,3],clusters2=[1,2,3], random_graph=False, noise=0.3, p_intra=0.5, p_inter=0.05) 
alphas=torch.tensor([0.5])
rhos=torch.tensor([0.01])

graphs=[]
graphs_embedding=torch.zeros(3*n_graphs,3*n_graphs)

for i in range(n_graphs):
    graph_pair=dataset_22.__getitem__(i)
    graph=Data(x=graph_pair.x_s, edge_index=graph_pair.edge_index_s)
    graphs.append(graph)

for i in range(n_graphs):
    graph_pair=dataset_22_bis.__getitem__(i)
    graph=Data(x=graph_pair.x_s, edge_index=graph_pair.edge_index_s)
    graphs.append(graph)


for i in range(n_graphs):
    graph_pair=dataset_33.__getitem__(i)
    graph=Data(x=graph_pair.x_s, edge_index=graph_pair.edge_index_s)
    graphs.append(graph)


for i in tqdm(range(3*n_graphs)):
    for j in range(i,3*n_graphs):
        graph_i=graphs[i]
        graph_j=graphs[j]
        graph_pair=PairData(x_s=graph_i.x,x_t=graph_j.x,edge_index_s=graph_i.edge_index,edge_index_t=graph_j.edge_index)
        graph_pair_batch=Batch.from_data_list([graph_pair],follow_batch=['x_s', 'x_t'])
        batch_indices_s =graph_pair_batch.x_s_batch
        batch_indices_t =graph_pair_batch.x_t_batch
        n_nodes_target=graph_pair_batch.x_t.shape[0]
        validation_nodes=np.random.randint(0,n_nodes_target,int(0.5*n_nodes_target))
        n_validation_nodes=len(validation_nodes)
        batched_rhos_s = rhos[batch_indices_s].reshape(
            len(rhos[batch_indices_s]), 1
        )
        batched_rhos_t = rhos[batch_indices_t].reshape(
            len(rhos[batch_indices_t]), 1
        )
        batched_alphas_s = alphas[batch_indices_s].reshape(
            len(alphas[batch_indices_s]), 1
        )
        batched_alphas_t =alphas[batch_indices_t].reshape(
            len(alphas[batch_indices_t]), 1
        )

        P, mask1, mask2 = model(
            graph_pair_batch.x_s,
            graph_pair_batch.x_t,
            graph_pair_batch.edge_index_s,
            graph_pair_batch.edge_index_t,
            graph_pair_batch.x_s_batch,
            graph_pair_batch.x_t_batch,
            batched_rhos_s,
            batched_rhos_t,
            batched_alphas_s,
            batched_alphas_t,
        )
        mass=torch.sum(P[0])
        graphs_embedding[i,j]=mass
        graphs_embedding[j,i]=mass

#%% Compute MDS using the ULOT transport plan mass distance matrix

pl.figure(figsize=(3,3))
distance_matrix=1-graphs_embedding.detach()
mds=MDS(n_components=2, dissimilarity='precomputed')
graph_embed = mds.fit_transform(distance_matrix)

pl.figure(figsize=(3,3))
c=torch.zeros(3*n_graphs)
c[:n_graphs]=1
c[n_graphs:2*n_graphs]=2
c[2*n_graphs:]=3

unique_c = np.unique(c)

for val in unique_c:
    mask = c == val
    if val==1:
      pl.scatter(graph_embed[mask, 0], graph_embed[mask, 1], label="(1,2) clusters ")
    if val==2:
      pl.scatter(graph_embed[mask, 0], graph_embed[mask, 1], label="(2,3) clusters")
    if val==3:
      pl.scatter(graph_embed[mask, 0], graph_embed[mask, 1], label="(1,2,3) clusters")

pl.legend()
pl.xticks([])
pl.yticks([])
pl.title("MDS on the ULOT plan mass")



#%% Visualize the ULOT transport plan mass similarity matrix

pl.figure(figsize=(5,5))
pl.imshow(graphs_embedding.detach(), interpolation="nearest")
cbar = pl.colorbar()
cbar.set_label('ULOT transport plan mass')
pl.xticks([10,30,50], labels=["(1,2)","(2,3)","(1,2,3)"])
pl.yticks([10,30,50], labels=["(1,2)","(2,3)","(1,2,3)"])
pl.title("Similarity matrix")
