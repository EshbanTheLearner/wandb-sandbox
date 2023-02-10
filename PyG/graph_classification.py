import os
import torch
import json
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import scipy.sparse as sp
import wandb
from torch import Tensor
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset
import networkx as nx
from networkx.algorithms import community
from tqdm.auto import trange
from visualize import GraphVisualization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_wandb = True
wandb_project = "PyG_Intro"
wandb_run_name = "upload_and_analyze_dataset"

if use_wandb:
    wandb.init(
        project=wandb_project,
        name=wandb_run_name
    )

dataset_path = "../data/TUDataset"
dataset = TUDataset(root=dataset_path, name="MUTAG")
dataset.download()

data_details = {
    "num_node_features": dataset.num_node_features,
    "num_edge_features": dataset.num_edge_features,
    "num_classes": dataset.num_classes,
    "num_node_labels": dataset.num_node_labels,
    "num_edge_labels": dataset.num_edge_labels
}

if use_wandb:
    wandb.log(data_details)
else:
    print(json.dumps(data_details, sort_keys=True, indent=4))

def create_graph(graph):
    g = to_networkx(graph)
    pos = nx.spring_layout(g)
    vis = GraphVisualization(
        g, 
        pos, 
        node_text_position="top_left",
        node_size=20
    )
    fig = vis.create_figure()
    return fig

fig = create_graph(dataset[0])
fig.show()

if use_wandb:
    table = wandb.Table(
        columns=[
            "Graph",
            "Number of Nodes",
            "Number of Edges",
            "Label"
        ]
    )
    for graph in dataset:
        fig = create_graph(graph)
        n_nodes = graph.num_nodes
        n_edges = graph.num_edges
        label = graph.y.item()
        table.add_data(
            wandb.Html(plotly.io.to_html(fig)),
            n_nodes,
            n_edges,
            label
        )
    wandb.log({
        "data": table
    })

if use_wandb:
    dataset_artifact = wandb.Artifact(
        name="MUTAG",
        type="dataset",
        metadata=data_details
    )
    dataset_artifact.add_dir(dataset_path)
    wandb.log_artifact(dataset_artifact)
    wandb.finish()