import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.datasets import cifar10

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
#from wandb.keras import WandbCallback

wandb.login()

BATCH_SIZE = 64

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(BATCH_SIZE)

def make_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x1 = keras.layers.Dense(64, activation="relu")(inputs)
    x2 = keras.layers.Dense(64, activation="relu")(x1)
    outputs = keras.layers.Dense(10, name="prediction")(x2)
    return keras.Model(inputs=inputs, outputs=outputs)

def train_step(x, y, model, optimizer, loss_fn, train_acc_metric):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value

def test_step(x, y, model, loss_fn, val_acc_metric):
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    val_acc_metric.update_state(y, val_logits)
    return loss_value

def train(train_dataset, val_dataset, model, 
        optimizer, train_acc_metric, val_acc_metric, 
        epochs=10, log_step=200, val_log_step=50):
        for epoch in range(epochs):
            print(f"\nStart of epoch {epoch}")
            train_loss = []
            val_loss =[]
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                loss_value = train_step(x_batch_train, y_batch_train,
                                        model, optimizer,
                                        loss_fn, train_acc_metric)
                train_loss.append(float(loss_value))
            for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                val_loss_value = test_step(x_batch_val, y_batch_val,
                                            model, loss_fn,
                                            val_acc_metric)
                val_loss.append(float(val_loss_value))
            train_acc = train_acc_metric.result()
            print(f"Training acc over epoch: {float(train_acc):.4f}")
            val_acc = val_acc_metric.result()
            print(f"Validation acc: {float(val_acc):.4f}")
            train_acc_metric.reset_states()
            val_acc_metric.reset_states()

            wandb.log({
                "epochs": epoch,
                "loss": np.mean(train_loss),
                "acc": float(train_acc),
                "val_loss": np.mean(val_loss),
                "val_acc": float(val_acc)
            })

config = {
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 64,
    "log_step": 200,
    "val_log_step": 50,
    "architecture": "CNN",
    "dataset": "MNIST"
}

run = wandb.init(project="tf-mnist-demo", config=config)
config = wandb.config

model = make_model()

optimizer = keras.optimizers.SGD(learning_rate=config.learning_rate)

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

train(train_dataset, val_dataset, model, 
    optimizer, train_acc_metric, 
    val_acc_metric, epochs=config.epochs, 
    log_step=config.log_step, val_log_step=config.val_log_step)