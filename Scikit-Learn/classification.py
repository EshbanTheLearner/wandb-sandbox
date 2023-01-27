import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import wandb
wandb.login()

wbcd = datasets.load_breast_cancer()

feature_names = wbcd.feature_names
labels = wbcd.target_names

X_train, X_test, y_train, y_test = train_test_split(wbcd.data, wbcd.target, test_size=0.2)

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_probas = model.predict_proba(X_test)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

run = wandb.init(
    project="sklearn-classification",
    name="classification"
)

wandb.sklearn.plot_class_proportions(y_train, y_test, labels)
wandb.sklearn.plot_learning_curve(model, X_train, y_train)
wandb.sklearn.plot_roc(y_test, y_probas, labels)
wandb.sklearn.plot_precision_recall(y_test, y_probas)
wandb.sklearn.plot_feature_importances(model)

wandb.sklearn.plot_classifier(
    model, X_train, X_test,
    y_train, y_test, y_pred,
    y_probas, labels, is_binary=True,
    model_name="RandomForest"
)

wandb.finish()