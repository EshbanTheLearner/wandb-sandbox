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