import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras import models
from tensorflow.keras.datasets import cifar10
# import tensorflow_datasets as tfds

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbEvalCallback

wandb.login()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train[::55] / 255., x_test / 255.
y_train = y_train[::5]

CLASS_NAMES = [
    "airplane", "automobile", "bird",
    "cat", "deer", "dog", "frog",
    "horse", "ship", "truck"
]

print("Shape of x_train: ", x_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of x_test: ", x_test.shape)
print("Shape of y_test: ", y_test.shape)

def Model():
    inputs = keras.layers.Input(shape=(32, 32, 3))
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(inputs)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return keras.models.Model(inputs=inputs, outputs=outputs)

run = wandb.init(
    project="keras-cifar10",
    config = {
        "learning_rate": 0.005,
        "epochs": 5,
        "batch_size": 1024,
        "loss_function": "sparse_categorical_crossentropy",
        "architecture": "CNN",
        "dataset": "CIFAR-10"
    }
)

config = wandb.config

tf.keras.backend.clear_session()

model = Model()
model.summary()

optimizer = tf.keras.optimizer.Adam(config.learning_rate)
model.compile(optimizer, config.loss_function, metrics=["acc"])

wandb_callbacks = [
    WandbMetricsLogger(),
    WandbModelCheckpoint(filepath="keras_cifar10_{epoch:02d}")
]

model.fit(
    x_train, y_train,
    epochs=config.epochs,
    batch_size=config.batch_size,
    validation_data=(x_test, y_test),
    callbacks=wandb_callbacks
)

loss, accuracy = model.evaluate(x_test, y_test)
print("Test Error Rate:", round((1 - accuracy) * 100, 2))

wandb.log({
    "Test Error Rate": round((1 - accuracy) * 100, 2)
})

run.finish()