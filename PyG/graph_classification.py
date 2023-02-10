import os
import torch
import json
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import plotly
import scipy.sparse as sp
import wandb
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
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

dataset_path = "./data/TUDataset"
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
        node_text_position="top left",
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

torch.manual_seed(42)

dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

print(f"Number of Training Graphs: {len(train_dataset)}")
print(f"Number of Test Graphs: {len(test_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

for step, data in enumerate(train_loader):
    print(f"Step {step + 1}:")
    print("=======")
    print(f"Number of graphs in current batch: {data.num_graphs}")
    print(data)
    print()

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

wandb_project = "PyG_Intro"
wandb_run_name = "upload_and_analyze_dataset"

if use_wandb:
    wandb.init(
        project=wandb_project
    )
    wandb.use_artifact("eshban9492/PyG_Intro/MUTAG:v0")

model = GCN(hidden_channels=64)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    for data in train_loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(loader, create_table=False):
    model.eval()
    table = wandb.Table(
        columns=[
            "graph",
            "ground truth",
            "prediction"
        ]
    ) if use_wandb else None
    correct = 0
    loss_ = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss_ += loss.item()
        pred = out.argmax(dim=1)
        if create_table and use_wandb:
            table.add_data(
                wandb.Html(
                    plotly.io.to_html(create_graph(data))
                ),
                data.y.item(),
                pred.item()
            )
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset), loss_ / len(loader.dataset), table

