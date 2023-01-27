import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import datasets, cluster
from sklearn.model_selection import train_test_split

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import wandb
wandb.login()

iris = datasets.load_iris()

X, y = iris.data, iris.target
names = iris.target_names

def get_label_ids(classes):
    return np.array([names[aclass] for aclass in classes])

labels = get_label_ids(y)

kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(X)

run = wandb.init(
    project="sklearn-clustering",
    name="clustering"
)

wandb.sklearn.plot_elbow_curve(kmeans, X)
wandb.sklearn.plot_silhouette(kmeans, X, labels)
wandb.sklearn.plot_cluster(kmeans, X, cluster_labels, labels, "Kmeans")

wandb.finish()