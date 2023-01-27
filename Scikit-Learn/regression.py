import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn import datasets
from sklearn.model_selection import train_test_split

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import wandb
wandb.login()

housing = datasets.fetch_california_housing()

X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

X, y = X[::2], y[::2]

wandb.errors.term._show_warnings = False

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#reg = Ridge()
reg = LinearRegression()

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

run = wandb.init(
    project="scikit-regression",
    name="regerssion"
)

wandb.sklearn.plot_residuals(reg, X_train, y_train)
wandb.sklearn.plot_outlier_candidates(reg, X_train, y_train)
wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test, model_name='Ridge')

wandb.finish()