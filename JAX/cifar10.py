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

(full_train_set, test_dataset), ds_info = tfds.load(
    "cifar10",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

def normalize_img(image, label):
  image = tf.cast(image, tf.float32) / 255.
  return image, label

full_train_set = full_train_set.map(
    normalize_img,
    num_parallel_calls=tf.data.AUTOTUNE
)

num_data = tf.data.experimental.cardinality(
    full_train_set
).numpy()

print(f"Total number of data points: {num_data}")

train_dataset = full_train_set.take(
    num_data * (1 - config.validation_split)
)

val_dataset = full_train_set.take(
    num_data * (config.validation_split)
)

print(f"Number of train examples: {tf.data.experimental.cardinality(train_dataset).numpy()}")
print(f"Number of validation examples: {tf.data.experimental.cardinality(val_dataset).numpy()}")

train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(
    tf.data.experimental.cardinality(train_dataset).numpy()
)

train_dataset = train_dataset.batch(config.batch_size)

val_dataset = val_dataset.cache()
val_dataset = val_dataset.shuffle(
    tf.data.experimental.cardinality(val_dataset).numpy()
)

val_dataset = val_dataset.batch(config.batch_size)

test_dataset = test_dataset.map(
    normalize_img,
    num_parallel_calls=tf.data.AUTOTUNE
)

print(f"Number of test records: {tf.data.experimental.cardinality(test_dataset).numpy()}")

test_dataset = test_dataset.cache()
test_dataset = test_dataset.batch(config.batch_size)