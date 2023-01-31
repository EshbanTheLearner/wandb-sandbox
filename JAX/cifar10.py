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

class CNN(nn.Module):
  pool_module: Callable = nn.avg_pool

  def setup(self):
    self.conv_1 = nn.Conv(features=32, kernel_size=(3, 3))
    self.conv_2 = nn.Conv(features=32, kernel_size=(3, 3))
    self.conv_3 = nn.Conv(features=64, kernel_size=(3, 3))
    self.conv_4 = nn.Conv(features=64, kernel_size=(3, 3))
    self.conv_5 = nn.Conv(features=128, kernel_size=(3, 3))
    self.conv_6 = nn.Conv(features=128, kernel_size=(3, 3))
    self.dense_1 = nn.Dense(features=1024)
    self.dense_1 = nn.Dense(features=512)
    self.dense_output = nn.Dense(features=10)

  @nn.compact
  def __call__(self, x):
    x = nn.relu(self.conv_1(x))
    x = nn.relu(self.conv_2(x))
    x = self.pool_module(x, window_shape=(2, 2), stride=(2, 2))
    x = nn.relu(self.conv_3(x))
    x = nn.relu(self.conv_4(x))
    x = self.pool_module(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.relu(self.conv_5(x))
    x = nn.relu(self.conv_6(x))
    x = self.pool_module(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))
    x = nn.relu(self.dense_1(x))
    x = nn.relu(self.dense_2(x))
    return self.dense_output(x)

rng = jax.random.PRNGKey(config.seed)
x = jnp.ones(shape=(config.batch_size, 32, 32, 3))
model = CNN(pool_module=MODULE_DICT[config.pooling])
params = model.init(rng, x)
jax.tree_map(lambda x: x.shape, params)

nn.tabulate(model, rng)(x)

def init_train_state(model, random_key, shape, learning_rate) -> train_state.TrainState:
  variables = model.init(random_key, jnp.ones(shape))
  optimizer = optax.adam(learning_rate)
  return train_state.TrainState.create(
      apply_fn=model.apply,
      tx=optimizer,
      params=variables["params"]
  )

state = init_train_state(
    model, 
    rng,
    (config.batch_size, 32, 32, 3),
    config.learning_rate
)

print(type(state))

def cross_entropy_loss(*, logits, labels):
  one_hot_encoded_labels = jax.nn.one_hot(labels, num_classes=10)
  return optax.softmax_cross_entropy(
      logits=logits, labels=one_hot_encoded_labels
  ).mean()

def compute_metrics(*, logits, labels):
  loss = cross_entropy_loss(logits=logits, labels=labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      "loss": loss,
      "accuracy": accuracy
  }
  return metrics

@jax.jit
def train_step(state: train_state.TrainState, batch: jnp.ndarray):
  image, label = batch
  def loss_fn(params):
    logits = state.apply_fn({
        "params": params
    }, image)
    loss = cross_entropy_loss(logits=logits, labels=label)
    return loss, logits
  
  gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = gradient_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits=logits, labels=label)
  return state, metrics

@jax.jit
def eval_step(state, batch):
  image, label = batch
  logits = state.apply_fn({
      "params": params
  }, image)
  return compute_metrics(logits=logits, labels=label)