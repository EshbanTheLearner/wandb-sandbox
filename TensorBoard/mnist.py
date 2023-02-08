import tensorflow as tf
import os
import datetime

import wandb

wandb.init(
    project="MNIST-TF",
    sync_tensorboard=True
)

mnist = tf.keras.datasts.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0