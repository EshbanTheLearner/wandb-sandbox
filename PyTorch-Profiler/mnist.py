import glob
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from torch.profiler import tensorboard_trace_handler
import wandb

torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors if not mirror.startswith("http://yann.lecun.com")]

wandb.login()

OPTIMIZERS = {
    "Adadelta": optim.Adadelta,
    "Adagrad": optim.Adagrad,
    "SGD": optim.SGD
}

class Net(pl.LightningModule):
    def __init__(self, optimizer="Adadelta"):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.optimizer = self.set_optimizer(optimizer)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    def set_optimizer(self, optimizer):
        return OPTIMIZERS[optimizer]
    

def training_step(self, batch, idx):
    inputs, labels = batch
    outputs = self(inputs)
    loss = F.nll_loss(outputs, labels)
    return {
        "loss": loss
    }

def configure_optimizers(self):
    return self.optimizer(self.parameters(), lr=0.1)

