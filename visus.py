# %%
import torch
import pylab as pl
import json
from ulot.utils import get_filename
import warnings
import numpy as np
from scipy.stats import pearsonr
import networkx as nx
import random
from torch_geometric.data import Data
import matplotlib.ticker as ticker
from torch_geometric.utils import to_networkx,remove_self_loops

torch.manual_seed(42)


cmap = pl.get_cmap('tab10')
color0 = cmap(0)    
color1 = cmap(1)    
color2 = cmap(2)
color3 = cmap(3)
color4 = cmap(4)
color5 = cmap(5)
color6 = cmap(6)
color7 = cmap(7)

cmap_paired = pl.get_cmap('Paired')
color0_paired = cmap_paired(0)    
color3_paired = cmap_paired(4)    
color4_paired0 = cmap_paired(8)    
color4_paired = cmap_paired(9)    


#%% visu plans for increasing alpha
##################################################################################################
##################################################################################################

torch.manual_seed(42)
np.random.seed(42)
random.seed(41)
fig, axs = pl.subplots(2, 3, figsize=(5, 3), dpi=600)  # 1 row, 5 columns
fig.suptitle(r"OT plan with respect to $\alpha$ for $(1,2,3) \rightarrow (1,2,3)$",y=1)

pos_g1=torch.load("results/figure_files/pos_g1_alpha")
pos_g2=torch.load("results/figure_files/pos_g2_alpha")
G1_nx=torch.load("results/figure_files/G1_nx_alpha")
G2_nx=torch.load("results/figure_files/G2_nx_alpha")
rgb1=torch.load("results/figure_files/rgb1_alpha")
rgb2=torch.load("results/figure_files/rgb2_alpha")
graph_pair=torch.load("results/figure_files/graph_pair_alpha")
all_P_solver=torch.load("results/figure_files/all_P_solver_alpha")
all_P_ulot=torch.load("results/figure_files/all_P_ulot_alpha")

 
rhos = torch.tensor([0.01])
alphas=torch.linspace(0,1,3)
n = 4
num = 0
cmap = pl.get_cmap('Reds')

max=np.max([torch.max(graph_pair.x_s),torch.max(graph_pair.x_t)])
min=np.min([torch.min(graph_pair.x_s),torch.min(graph_pair.x_t)])
for i, alpha in enumerate(alphas):
        P=all_P_ulot[i]
        ax = axs[0,i]
        

        for u in range(P[0].shape[1]):
            for v in range(P[0].shape[0]):
                    ax.plot(
                        [pos_g1[u][0], pos_g2[v][0]],
                        [pos_g1[u][1], pos_g2[v][1]],
                        "r",
                        alpha=P[0][u, v].item() * 20,
                    )
        nx.draw_networkx(
            G1_nx,
            pos=pos_g1,
            with_labels=False,
            node_color=rgb1,
            cmap="tab10",
            node_size=30,
            ax=ax,
            linewidths=0.8,
            edgecolors='black',
            width=0.5
        )


        nx.draw_networkx(
            G2_nx,
            pos=pos_g2,
            with_labels=False,
            node_color=rgb2,
            cmap="tab10",
            node_size=30,
            ax=ax,
            linewidths=0.8,
            edgecolors='black',
            width=0.5
        )

        #   ax.set_title(f"rho={rho:.5f},alpha={alpha:.5f}, volume={torch.sum(solver_plan):.2f}, loss={solver_loss:.5f}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_aspect('equal')       # <-- Same scale for both axes
        ax.set_ylim(-3.5,1.0)
        ax.set_xlim(-1.5,5)


        ax=axs[1,i]
        P=all_P_solver[i]
        for u in range(P.shape[1]):
            for v in range(P.shape[0]):
                    ax.plot(
                        [pos_g1[u][0], pos_g2[v][0]],
                        [pos_g1[u][1], pos_g2[v][1]],
                        "r",
                        alpha=P[u, v].item() * 20,
                    )
        nodes=nx.draw_networkx(
            G1_nx,
            pos=pos_g1,
            with_labels=False,
            node_color=rgb1,
            cmap="tab10",
            node_size=30,
            ax=ax,
            linewidths=0.8,
            edgecolors='black',
            width=0.5

        )

        nx.draw_networkx(
            G2_nx,
            pos=pos_g2,
            with_labels=False,
            node_color=rgb2,
            cmap="tab10",
            node_size=30,
            ax=ax,
            linewidths=0.8,
            edgecolors='black',
            width=0.5
        )

        #   ax.set_title(f"rho={rho:.5f},alpha={alpha:.5f}, volume={torch.sum(solver_plan):.2f}, loss={solver_loss:.5f}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_aspect('equal') 

        ax.set_ylim(-3.5,1.0)
        ax.set_xlim(-1.5,5)
        ax.text(0.5, 0, r"$\alpha$"+f"={alpha}", transform=ax.transAxes, ha='center', va='top') 




fig.tight_layout()
axs[0,0].set_title("ULOT",loc='left')
axs[1,0].set_title("Solver",loc='left')


#%% visu plans for increasing rho
##################################################################################################
##################################################################################################

torch.manual_seed(42)
np.random.seed(42)
random.seed(41)
fig, axs = pl.subplots(2, 3, figsize=(5, 3), dpi=600)  # 1 row, 5 columns
fig.suptitle(r"OT plan with respect to $\rho$ for $(1,2) \rightarrow (2,3)$",y=1)

pos_g1=torch.load("results/figure_files/pos_g1_rho")
pos_g2=torch.load("results/figure_files/pos_g2_rho")
G1_nx=torch.load("results/figure_files/G1_nx_rho")
G2_nx=torch.load("results/figure_files/G2_nx_rho")
rgb1=torch.load("results/figure_files/rgb1_rho")
rgb2=torch.load("results/figure_files/rgb2_rho")
graph_pair=torch.load("results/figure_files/graph_pair_rho")
all_P_solver=torch.load("results/figure_files/all_P_solver_rho")
all_P_ulot=torch.load("results/figure_files/all_P_ulot_rho")

 
rhos = np.exp(-np.linspace(5, 7, 3))
alpha=torch.tensor([[0.8]])
n = 4
num = 0
cmap = pl.get_cmap('Reds')

max=np.max([torch.max(graph_pair.x_s),torch.max(graph_pair.x_t)])
min=np.min([torch.min(graph_pair.x_s),torch.min(graph_pair.x_t)])
for i, rho in enumerate(rhos[::-1]):
        P=all_P_ulot[i]

        ax = axs[0,i]

        for u in range(P[0].shape[1]):
            for v in range(P[0].shape[0]):
                    ax.plot(
                        [pos_g1[u][0], pos_g2[v][0]],
                        [pos_g1[u][1], pos_g2[v][1]],
                        "r",
                        alpha=P[0][u, v].item() * 20,
                    )
        nx.draw_networkx(
            G1_nx,
            pos=pos_g1,
            with_labels=False,
            node_color=rgb1,
            cmap="tab10",
            node_size=30,
            ax=ax,
            linewidths=0.8,
            edgecolors='black',
            width=0.5
        )


        nx.draw_networkx(
            G2_nx,
            pos=pos_g2,
            with_labels=False,
            node_color=rgb2,
            cmap="tab10",
            node_size=30,
            ax=ax,
            linewidths=0.8,
            edgecolors='black',
            width=0.5
        )

        #   ax.set_title(f"rho={rho:.5f},alpha={alpha:.5f}, volume={torch.sum(solver_plan):.2f}, loss={solver_loss:.5f}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_aspect('equal')       # <-- Same scale for both axes
        ax.set_ylim(-3.5,1.0)
        ax.set_xlim(-1.5,5)


        ax=axs[1,i]
        P=all_P_solver[i]
        for u in range(P.shape[1]):
            for v in range(P.shape[0]):
                    ax.plot(
                        [pos_g1[u][0], pos_g2[v][0]],
                        [pos_g1[u][1], pos_g2[v][1]],
                        "r",
                        alpha=P[u, v].item() * 20,
                    )
        nodes=nx.draw_networkx(
            G1_nx,
            pos=pos_g1,
            with_labels=False,
            node_color=rgb1,
            cmap="tab10",
            node_size=30,
            ax=ax,
            linewidths=0.8,
            edgecolors='black',
            width=0.5

        )

        nx.draw_networkx(
            G2_nx,
            pos=pos_g2,
            with_labels=False,
            node_color=rgb2,
            cmap="tab10",
            node_size=30,
            ax=ax,
            linewidths=0.8,
            edgecolors='black',
            width=0.5
        )

        #   ax.set_title(f"rho={rho:.5f},alpha={alpha:.5f}, volume={torch.sum(solver_plan):.2f}, loss={solver_loss:.5f}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_aspect('equal') 

        ax.set_ylim(-3.5,1.0)
        ax.set_xlim(-1.5,5)
        ax.text(0.5, 0, r"$\rho$"+f"={rho:.3f}", transform=ax.transAxes, ha='center', va='top') 



fig.tight_layout()
axs[0,0].set_title("ULOT",loc='left')
axs[1,0].set_title("Solver",loc='left')


#%% Load data SBM
##################################################################################################
##################################################################################################

param_file="parameter_files/params_SBM_10000.json"
with open(param_file, "r") as file:
    parameters = json.load(file)


path_test_performances_sbm = get_filename(parameters, "test_performances")
path_true_losses_sbm = get_filename(parameters, "true_losses")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    test_performances_sbm = torch.load(
        path_test_performances_sbm, map_location=torch.device("cpu"))
    true_losses_all_sbm = np.load(path_true_losses_sbm, allow_pickle=True).item()


#%% Scatter plot GNN VS solver - SBM
####### test
##################################################################################################
##################################################################################################

true_losses = true_losses_all_sbm["losses_test_solver"]
color = [
    torch.sum(test_performances_sbm["plans_test"][i]).detach().numpy()
    for i in range(len(test_performances_sbm["losses_test"]))
]
losses = [x["total"][0].detach().numpy() for x in test_performances_sbm["losses_test"]]

color = [
    torch.sum(test_performances_sbm["plans_test"][i]).detach().numpy()
    for i in range(len(test_performances_sbm["losses_test"]))
]

losses_uniform = test_performances_sbm["losses_uniform_test"]
mask = ~np.isnan(true_losses)
true_losses = np.array(true_losses)
losses = np.array(losses)

pl.figure(figsize=(4, 3))
pl.scatter(
    true_losses,
    losses,
    alpha=0.5,
    c=color,
    label=f"pearson correlation={pearsonr(true_losses[mask], losses[mask]).statistic:.2f}",
    vmin=0,
    vmax=1,
)
pl.plot([0, 0.05], [0, 0.05], color="red", linestyle="--")
pl.xlabel("Solver")
pl.ylabel("ULOT")
pl.colorbar(label="volume of the learned plan")
pl.legend()
pl.grid()
pl.title("FUGW loss for ULOT and solver")
pl.tight_layout()


#%% label propagation
##################################################################################################
##################################################################################################

all_alphas_2=torch.load("results/all_alphas_2_no_weight_bis")
all_rhos_2=torch.load("results/all_rhos_2_no_weight_bis")

all_alphas_3=torch.load("results/all_alphas_3_no_weight")
all_rhos_3=torch.load("results/all_rhos_3_no_weight")

pl.figure(1,figsize=(3,3))
pl.hist2d(all_alphas_2[:,-1],all_rhos_2[:,-1],cmap="Blues",bins=[10,10], range=np.array([[0,1],[0,1]]), cmin=0.0000001, density=True)
pl.colorbar()

pl.xlabel(r"$\alpha$")
pl.ylabel(r"$\rho$")

pl.title("2 clusters")
pl.figure(2,figsize=(3,3))
pl.hist2d(all_alphas_3[:,-1],all_rhos_3[:,-1],cmap="Reds",bins=[10,10], range=np.array([[0,1],[0,1]]), cmin=0.0000001, density=True)
pl.colorbar()
pl.title("3 clusters")


pl.xlabel(r"$\alpha$")
pl.ylabel(r"$\rho$")


#%% Functional minimization
##################################################################################################
##################################################################################################

torch.manual_seed(40)
np.random.seed(6)
F_along_step_01=torch.load("results/figure_files/F_along_step_01_sub")
F_along_step_05=torch.load("results/figure_files/F_along_step_05_sub")
G_along_step_01=torch.load("results/figure_files/G_along_step_01_sub")
G_along_step_05=torch.load("results/figure_files/G_along_step_05_sub")
G_2=torch.load("results/figure_files/G2_gradient_flow")
steps_01=[0,100,300,1000,3000]
steps_05=[0,100,300,1000,3000]

G_flows_01=[]
G_flows_05=[]
for i in range(len(steps_05)):
    F=torch.tensor(F_along_step_05[i])
    G=torch.tensor(G_along_step_05[i])
    mask = (G > 0) & (G < 2)
    edge_index_flow = mask.nonzero(as_tuple=False).t()
    edge_index_flow = remove_self_loops(edge_index_flow)[0]
    output_graph=Data(x=F, edge_index=edge_index_flow)
    G_flows_05.append(output_graph)

for i in range(len(steps_01)):
    F=torch.tensor(F_along_step_01[i])
    G=torch.tensor(G_along_step_01[i])
    mask = (G > 0) & (G < 2)
    edge_index_flow = mask.nonzero(as_tuple=False).t()
    edge_index_flow = remove_self_loops(edge_index_flow)[0]
    output_graph=Data(x=F, edge_index=edge_index_flow)
    G_flows_01.append(output_graph)


G_0=G_flows_01[0]
G_0_nx= to_networkx(G_0, to_undirected=True)

min_rgb=np.min([np.min([torch.min(G_flows_01[i].x) for i in range(len(steps_01))])])

max_rgb=np.max([np.max([torch.max(G_flows_01[i].x) for i in range(len(steps_01))])])

rgb=(G_0.x - min_rgb) / (max_rgb - min_rgb)

pos_0 = nx.spring_layout(G_0_nx)
min_rgb=np.min([torch.min(G_0.x),np.min([torch.min(G_flows_01[i].x) for i in range(len(steps_01))]) ,np.min([torch.min(G_flows_05[i].x) for i in range(len(steps_01))]),torch.min(G_2.x)])

max_rgb=np.max([torch.max(G_0.x),np.max([torch.max(G_flows_01[i].x) for i in range(len(steps_01))]),np.max([torch.max(G_flows_05[i].x) for i in range(len(steps_05))]),torch.max(G_2.x)])

fig, axs = pl.subplots(2, 4, figsize=(8, 3), dpi=400)  # 1 row, 5 columns

G_1=G_flows_01[1]
graph=Data(x=G_1.x, edge_index=G_1.edge_index)
G_1_nx= to_networkx(G_1, to_undirected=True)
pos_1 = nx.spring_layout(G_1_nx, iterations=0, pos=pos_0)
rgb=(graph.x - min_rgb) / (max_rgb - min_rgb)


nx.draw_networkx(
    G_1_nx,
    with_labels=False,
    cmap="tab10",
    node_size=100,
    node_color=rgb,
    linewidths=0.8,
    edgecolors='black',
    width=0.5,
    pos=pos_1,
    ax=axs[0,0])
ax=axs[0,0]
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title(r"$\alpha=0.5$",loc='left')

for i in range(2, len(steps_05)):
    t = steps_05[i]
    G_flow = G_flows_05[i]

    G_nx = to_networkx(G_flow, to_undirected=True)

    # Identify isolated nodes before removing
    isolated_nodes = list(nx.isolates(G_nx))

    # Remove isolated nodes
    G_nx.remove_nodes_from(isolated_nodes)

    # Create positions using spring layout, with initial pos
    pos = nx.spring_layout(G_nx, iterations=10, pos=pos_0)

    ax = axs[0, i - 1]

    # Extract node features for remaining nodes only
    node_indices = list(G_nx.nodes)
    rgb = (G_flow.x[node_indices] - min_rgb) / (max_rgb - min_rgb)

    # Draw graph
    nx.draw_networkx(
        G_nx,
        with_labels=False,
        cmap="tab10",
        node_size=100,
        node_color=rgb,
        linewidths=0.8,
        edgecolors='black',
        width=0.5,
        pos=pos,
        ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
G_1=G_flows_05[1]
graph=Data(x=G_1.x, edge_index=G_1.edge_index)
G_1_nx= to_networkx(G_1, to_undirected=True)
pos_1 = nx.spring_layout(G_1_nx, iterations=0, pos=pos_0)
rgb=(graph.x - min_rgb) / (max_rgb - min_rgb)


nx.draw_networkx(
    G_1_nx,
    with_labels=False,
    cmap="tab10",
    node_size=100,
    node_color=rgb,
    linewidths=0.8,
    edgecolors='black',
    width=0.5,
    pos=pos_1,
    ax=axs[1,0])
ax=axs[1,0]
ax.text(0.5, -0.2, "step 200", transform=ax.transAxes, ha='center', va='top') 
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title(r"$\alpha=1$",loc='left')

    
for i in range(2, len(steps_01)):
    t = steps_01[i]
    G_flow = G_flows_01[i]

    G_nx = to_networkx(G_flow, to_undirected=True)

    # Identify isolated nodes before removing
    isolated_nodes = list(nx.isolates(G_nx))

    # Remove isolated nodes
    G_nx.remove_nodes_from(isolated_nodes)

    # Create positions using spring layout, with initial pos
    pos = nx.spring_layout(G_nx, iterations=10, pos=pos_0)

    ax = axs[1, i - 1]

    # Extract node features for remaining nodes only
    node_indices = list(G_nx.nodes)
    rgb = (G_flow.x[node_indices] - min_rgb) / (max_rgb - min_rgb)

    # Draw graph
    nx.draw_networkx(
        G_nx,
        with_labels=False,
        cmap="tab10",
        node_size=100,
        node_color=rgb,
        linewidths=0.8,
        edgecolors='black',
        width=0.5,
        pos=pos,
        ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, -0.2, f"step {t}", transform=ax.transAxes, ha='center', va='top') 

pl.subplots_adjust(hspace=0.5)

#%% SBM mass similarity matrix
##################################################################################################
##################################################################################################


graphs_embedding=torch.load("results/graphs_embedding")
pl.figure(figsize=(3,3))
pl.imshow(graphs_embedding.detach())
cbar = pl.colorbar(fraction=0.046, pad=0.04, shrink=0.8)
cbar.set_label('ULOT plan mass')
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

pl.xticks([10,30,50], labels=["(1,2)","(2,3)","(1,2,3)"])
pl.yticks([10,30,50], labels=["(1,2)","(2,3)","(1,2,3)"])
pl.title("Similarity matrix")

#%% Accuracy surfaces for clusters (1,2,3) -> (1,2,3)
##################################################################################################
##################################################################################################

all_accuracy_3=torch.load("results/figure_files/all_accuracy_3")
rhos=torch.load("results/figure_files/rhos")
alphas=torch.load("results/figure_files/alphas")
pl.figure(figsize=(3, 3))
#pl.imshow(all_accuracy_3[-1], vmin=0, vmax=1)
pl.imshow(torch.mean(all_accuracy_3, axis=0),vmin=0, vmax=1)

n_rho=len(rhos)
n_alpha=len(alphas)
cbar = pl.colorbar()
cbar.set_label("Accuracy")
pl.ylabel(r"$\alpha$")
pl.xlabel(r"$\rho$")
pl.xticks(
    ticks=np.arange(0, n_rho, 15), labels=[f"{x[0].item():.3f}" for x in rhos[::15]]
)

pl.yticks(
    ticks=np.arange(0, n_alpha, 9), labels=[f"{x[0].item():.2f}" for x in alphas[::9]]
)
pl.title("Accuracy for\n$(1,2,3) \\rightarrow (1,2,3)$")
pl.gca().invert_xaxis()
pl.gca().invert_yaxis()

#%% Accuracy surfaces for clusters (1,2,3) -> (1,2)
##################################################################################################
##################################################################################################

all_accuracy_3=torch.load("results/figure_files/all_accuracy_2")
rhos=torch.load("results/figure_files/rhos")
alphas=torch.load("results/figure_files/alphas")
pl.figure(figsize=(3, 3))
pl.imshow(all_accuracy_3[0], vmin=0, vmax=1)
#pl.imshow(torch.mean(all_accuracy_3, axis=0).T)

n_rho=len(rhos)
n_alpha=len(alphas)
cbar = pl.colorbar()
cbar.set_label("Accuracy")
pl.ylabel(r"$\alpha$")
pl.xlabel(r"$\rho$")
pl.xticks(
    ticks=np.arange(0, n_rho, 15), labels=[f"{x[0].item():.3f}" for x in rhos[::15]]
)

pl.yticks(
    ticks=np.arange(0, n_alpha, 9), labels=[f"{x[0].item():.2f}" for x in alphas[::9]]
)
pl.gca().invert_xaxis()
pl.gca().invert_yaxis()
pl.title("Accuracy for\n$(1,2,3) \\rightarrow (1,2)$")
