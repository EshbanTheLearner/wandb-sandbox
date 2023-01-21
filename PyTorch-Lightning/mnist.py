import numpy as np

import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split

from torchvision.datasets import MNIST
from torchvision import transforms

MNIST.mirrors = [mirror for mirror in MNIST.mirrors
                 if not mirror.startswith("http://yann.lecun.com")]

import pytorch_lightning as pl
import torchmetrics
pl.seed_everything(hash("Setting Random Seed") % 2**32 - 1)

import wandb
from pytorch_lightning.loggers import WandbLogger
wandb.login()

class LitMLP(pl.LightningModule):
    def __init__(self, in_dims, n_classes=10,
                n_layer_1=128, n_layer_2=256, lr=1e-4):
        super().__init__()
        self.layer_1 = nn.Linear(np.prod(in_dims), n_layer_1)
        self.layer_2 = nn.Linear(n_layer_1, n_layer_2)
        self.layer_3 = nn.Linear(n_layer_2, n_classes)

        self.save_hyperparameters()

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        batch_size, *dims = x.size()
        x = x.view(batch_size, -1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, xs, ys):
        logits = self(xs)
        loss = F.null_loss(logits, ys)
        return logits, loss

