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

