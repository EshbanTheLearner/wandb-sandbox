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

def save_checkpoint(ckpt_path, state, epoch):
  with open(ckpt_path, "wb") as outfile:
    outfile.write(msgpack_serialize(to_state_dict(state)))
  artifact = wandb.Artifact(
      f"{wandb.run.name}-checkpoint", type="dataset"
  )
  artifact.add_file(ckpt_path)
  wandb.log_artifact(artifact, aliases=["latest", f"epoch_{epoch}"])

def load_checkpoint(ckpt_file, state):
  artifact = wandb.use_artifact(
      f"{wandb.run.name}-checkpoint:latest"
  )
  artifact_dir = artifact.download()
  ckpt_path = os.path.join(artifact_dir, ckpt_file)
  with open(ckpt_path, "rb") as data_file:
    byte_data = data_file.read()
  return from_bytes(state, byte_data)

def accumulate_metrics(metrics):
  metrics = jax.device_get(metrics)
  return {
      k: np.mean([metrics[k] for metric in metrics])
      for k in metrics[0]
  }

def train_and_evaluate(train_dataset, eval_dataset, test_dataset, state: train_state.TrainState, epochs: int):
  num_train_batches = tf.data.experimental.cardinality(train_dataset)
  num_eval_batches = tf.data.experimental.cardinality(eval_dataset)
  num_test_batches = tf.data.experimental.cardinality(test_dataset)

  for epoch in tqdm(range(1, epochs + 1)):
    best_eval_loss = 1e6
    
    train_batch_metrics = []
    train_datagen = iter(tfds.as_numpy(train_dataset))
    for batch_idx in range(num_train_batches):
      batch = next(train_datagen)
      state, metrics = train_step(state, batch)
      train_batch_metrics.append(metrics)
    train_batch_metrics = accumulate_metrics(train_batch_metrics)
    print(f"TRAIN {epoch}/{epochs}: Loss: {train_batch_metrics['loss']}, Accuracy: {train_batch_metrics['accuracy'] * 100}")
    
    eval_batch_metrics = []
    eval_datagen = iter(tfds.as_numpy(eval_dataset))
    for batch_idx in range(num_eval_batches):
      batch = next(eval_datagen)
      metrics = eval_step(state, batch)
      eval_batch_metrics.append(metrics)
    eval_batch_metrics  = accumulate_metrics(eval_batch_metrics)
    print(f"EVAL {epoch}/{epochs} Loss: {eval_batch_metrics['loss']}, Accuracy: {eval_batch_metrics['accuracy'] * 100}")

    wandb.log({
        "Train Loss": train_batch_metrics["loss"],
        "Train Accuracy": train_batch_metrics["accuracy"],
        "Validation Loss": eval_batch_metrics["loss"],
        "Validation Accuracy": eval_batch_metrics["accuracy"]
    }, step=epoch)

    if eval_batch_metrics["loss"] < best_eval_loss:
      save_checkpoint("checkpoint.msgpack", state, epoch)
  
  restored_state = load_checkpoint("checkpoint.msgpack", state)
  test_batch_metrics = []
  test_datagen = iter(tfds.as_numpy(test_dataset))
  for batch_idx in range(num_test_batches):
    batch = next(test_datagen)
    metrics = eval_step(restored_state, batch)
    test_batch_metrics.append(metrics)

  test_batch_metrics = accumulate_metrics(test_batch_metrics)
  print(f"TEST Loss: {test_batch_metrics['loss']}, Accuracy: {test_batch_metrics['accuracy']*100}")
  wandb.log({
      "Test Loss": test_batch_metrics["loss"],
      "Test Accuracy": test_batch_metrics["accuracy"]
  })

  return state, restored_state

state, best_state = train_and_evaluate(
    train_dataset,
    val_dataset,
    test_dataset,
    state,
    epochs=config.epochs
)

wandb.finish()