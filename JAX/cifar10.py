import jax
import jax.numpy as jnp
import optax

from flax import linen as nn
from flax.training import train_state
from flax.serialization import (
    to_state_dict, msgpack_serialize, from_bytes
)

import os
import wandb
import numpy as np
from typing import Callable
from tqdm.auto import tqdm

import tensorflow as tf
import tensorflow_datasets as tfds

wandb.init(
    project="cifar10-jax",
    entity="jax",
    job_type="training-loop"
)

config = wandb.config
config.seed = 42
config.batch_size = 64
config.validation_split = 0.2
config.pooling = "avg"
config.learning_rate = 1e-4
config.epochs = 15

MODULE_DICT = {
    "avg": nn.avg_pool,
    "max": nn.max_pool
}