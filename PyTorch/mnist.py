import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

torch.backends.cudnn.deterministic = True
random.seed(hash("Setting random seed") % 2**32 - 1)
np.random.seed(hash("Improves Reproducibility") % 2**32 - 1)
torch.manual_seed(hash("By removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("So runs are repeatable") % 2**32 - 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors 
                                    if not mirror.startswith("http://yann.lecun.com")]

import wandb
wandb.login()

config = dict(
    epochs=5,
    classes=10,
    kernels=[16, 32],
    batch_size=128,
    learning_rate=0.005,
    dataset="MNIST",
    architecture="CNN"
)

def model_pipeline(hyperparameters):
    with wandb.init(project="pytorch-mnist-demo", config=hyperparameters):
        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)
        train(model, train_loader, criterion, optimizer, config)
        test(model, test_loader)
    return model

def make(config):
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)
    model = ConvNet(config.kernels, config.classes).to(device)
    criterion = nn.CrossEntropy()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate
    )
    return model, train_loader, test_loader, criterion, optimizer

def get_data(slice=5, train=True):
    full_dataset = torchvision.datasets.MNIST(
        root=".",
        train=train,
        transform=transforms.ToTensor(),
        download=True
    )
    sub_dataset = torch.utils.data.Subset(
        full_dataset,
        indices=range(0, len(full_dataset) ,slice)
    )
    return sub_dataset

def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2
    )
    return loader

class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                1,
                kernels[0],
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                16,
                kernels[1],
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )
        self.fc = nn.Linear(7 * 7 * kernels[-1], classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def train(model, loader, criterion, optimizer, config):
    wandb.watch(model, criterion, log="all", log_freq=10)
    total_batches = len(loader) * config.epochs
    example_ct = 0
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):
            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct += 1
            batch_ct += 1
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)

def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backwards()
    optimizer.step()
    return loss

def train_log(loss, example_ct, epoch):
    wandb.log({
        "epoch": epoch,
        "loss": loss
    }, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")

def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) 
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Accuracy of the model on the {total} " +
            f"test images: {correct/total:%}")
        wandb.log({
            "test_accuracy": correct / total
        })
    torch.onnx.export(model, images, "mnist-model.onnx")
    wandb.save("mnist-model.onnx")

model = model_pipeline(config)