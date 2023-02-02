import ast
import sys
import json
from pathlib import Path
from dill.source import getsource
from dill import detect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly

from scipy.stats import ks_2samp
from sklearn import metrics
from sklearn import model_selection
import xgboost as xgb

import wandb
wandb.login()

data_dir = Path(".")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

id_vars = ["UniqueID"]
targ_var = "loan_default"

def function_to_string(fn):
    return getsource(detect.code(fn))

run = wandb.init(
    project="xgboost-credit-scorecard",
    config={
        "wandb_nb": "wandb_credit_soc"
    }
)

dataset_art = run.use_artifact(
    "morgan/credit_scorecard/vehicle_loan_defaults:latest", 
    type="dataset"
)
dataset_dir = dataset_art.download(data_dir)

