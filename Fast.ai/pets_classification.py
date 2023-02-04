from fastai.vision.all import *
from fastai.callback.wandb import *

import wandb
wandb.login()

path = untar_data(URLs.PETS)/"images"
SEED = 42

def is_cat(x):
    return x[0].isupper()

dls = ImageDataLoaders.from_name_func(
    path,
    get_image_files(path),
    valid_pct=0.2,
    seed=SEED,
    bs=32,
    label_func=is_cat,
    item_tfms=Resize(128)
)

learn = vision_learner(
    dls, 
    "convnext_tiny",
    metrics=error_rate
)
learn.fine_tune(1)

learn = vision_learner(
    dls,
    "convnext_tiny",
    metrics=error_rate,
    cbs=WandbCallback()
)

wandb.init(
    project="fastai"
)

learn.fine_tune(1)

wandb.finish()