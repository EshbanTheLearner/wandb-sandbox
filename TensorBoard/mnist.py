import tensorflow as tf
import os
import datetime

import wandb

wandb.init(
    project="MNIST-TF",
    sync_tensorboard=True
)