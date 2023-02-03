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

data_dir = Path("./")
model_dir = Path("./models")
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

from data_utils import (
    describe_data_g_targ,
    one_hot_encode_data,
    create_feature_interaction_constraints,
    get_monotonic_constraints,
    load_training_data,
    calculate_credit_scores
)
from scorecard import generate_scorecard

dataset = pd.read_csv(data_dir/"vehicle_loans_subset.csv")

dataset, p_vars = one_hot_encode_data(dataset, id_vars, targ_var)

processed_data_path = data_dir/"proc_ds.csv"
dataset.to_csv(processed_data_path, index=False)

processed_ds_art = wandb.Artifact(
    name="vehicle_defaults_processed",
    type="processed_dataset",
    description="One-Hot Encoded Dataset",
    metadata={
        "preprocessing_fn": function_to_string(one_hot_encode_data)
    }
)

processed_ds_art.add_file(processed_data_path)
run.log_artifact(processed_ds_art)
run.finish()

with wandb.init(project="xgboost-credit-scorecard", job_type='train-val-split', config={"wandb_nb":"wandb_credit_soc"}) as run:
    dataset_art = run.use_artifact(
        "vehicle_defaults_processed:latest", 
        type="processed_dataset"
    )
    dataset_dir = dataset_art.download(data_dir)
    dataset = pd.read_csv(processed_data_path)

    test_size = 0.25
    random_state = 42

    run.config.update({
        "test_size": test_size,
        "random_state": random_state
    })

    train_data, val_data = model_selection.train_test_split(
        dataset, 
        test_size=test_size,
        random_state=random_state, 
        stratify=dataset[[targ_var]]
    )

    print(f"Train dataset size: {train_data[targ_var].value_counts()} \n")
    print(f"Validation dataset sizeL {val_data[targ_var].value_counts()}")

    train_path = data_dir/"train.csv"
    val_path = data_dir/"val.csv"