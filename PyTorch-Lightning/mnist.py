import numpy as np

import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split

from torchvision.datasets import MNIST
from torchvision import transforms

MNIST.mirrors = [mirror for mirror in MNIST.mirrors
                 if not mirror.startswith("http://yann.lecun.com")]

import pytorch_lightning as pl
import torchmetrics
pl.seed_everything(hash("Setting Random Seed") % 2**32 - 1)

import wandb
from pytorch_lightning.loggers import WandbLogger
wandb.login()

class LitMLP(pl.LightningModule):
    def __init__(self, in_dims, n_classes=10,
                n_layer_1=128, n_layer_2=256, lr=1e-4):
        super().__init__()
        self.layer_1 = nn.Linear(np.prod(in_dims), n_layer_1)
        self.layer_2 = nn.Linear(n_layer_1, n_layer_2)
        self.layer_3 = nn.Linear(n_layer_2, n_classes)

        self.save_hyperparameters()

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        batch_size, *dims = x.size()
        x = x.view(batch_size, -1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, xs, ys):
        logits = self(xs)
        loss = F.null_loss(logits, ys)
        return logits, loss

def training_step(self, batch, batch_idx):
    xs, ys = batch
    logits, loss = self.loss(xs, ys)
    preds = torch.argmax(logits, 1)
    self.log("train/loss", loss, on_epoch=True)
    self.train_acc(preds, ys)
    self.log("train/acc", self.train_acc, on_epoch=True)
    return loss

def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

LitMLP.training_step = training_step
LitMLP.configure_optimizers = configure_optimizers

def test_step (self, batch, batch_idx):
    xs, ys = batch
    logits, loss = self.loss(xs, ys)
    preds = torch.argmax(logits, 1)
    self.test_acc(preds, ys)
    self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
    self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)

LitMLP.test_step = test_step

def test_epoch_end(self, test_step_outputs):
    dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
    model_filename = "lit-mnist.onnx"
    self.to_onnx(model_filename, dummy_input, export_params=True)
    artifact = wandb.Artifact(name="model.ckpt", type="model")
    artifact.add_file(model_filename)
    wandb.log_artifact(artifact)

LitMLP.test_epoch_end = test_epoch_end

def validation_step(self, batch, batch_idx):
    xs, ys = batch
    logits, loss = self.loss(xs, ys)
    preds = torch.argmax(logits, 1)
    self.valid_acc(preds, ys)
    self.log("valid/loss_epoch", loss)
    self.log("valid/acc_epoch", self.valid_acc)
    return logits

def validation_epoch_end(self, validation_step_outputs):
    dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
    model_filename = f"model_{str(self.global_step).zfill(5)}.onnx"
    torch.onnx.export(self, dummy_input, model_filename, opset_version=11)
    artifact = wandb.Artifact(name="model.ckpt", type="model")
    artifact.add_file(model_filename)
    self.logger.experiment.log_artifact(artifact)
    flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
    self.logger.experiment.log({
        "valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
        "global_step": self.global_step
    })

LitMLP.validation_step = validation_step
LitMLP.validation_epoch_end = validation_epoch_end

class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, 1)
        trainer.logger.experiment.log({
            "examples": [wandb.Image(x, caption=f"Pred: {pred}, Label: {y}")
                            for x, pred, y in zip(val_imgs, preds, self.val_labels)],
            "global_step": trainer.global_step
        })

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./", batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist, [55000, 5000])
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_val, batch_size=10*self.batch_size)
        return mnist_val
    
    def test_dataloader(self):
        mnist_test = DataLoader(self.mnist_test, batch_size=10*self.batch_size)
        return mnist_test

mnist = MNISTDataModule()
mnist.prepare_data()
mnist.setup()

samples = next(iter(mnist.val_dataloader))

wandb_logger = WandbLogger(project="lit-mnist")

trainer = pl.Trainer(
    logger=wandb_logger,
    log_every_n_steps=50,
    gpus=-1,
    max_epochs=5,
    deterministic=True,
    callbacks=[
        ImagePredictionLogger(samples)
    ]
)

model = LitMLP(in_dims=(1, 28, 28))

trainer.fit(model, mnist)

trainer.test(datamodule=mnist, ckpt_path=None)

wandb.finish()