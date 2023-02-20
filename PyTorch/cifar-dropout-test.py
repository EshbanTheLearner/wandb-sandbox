import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

import wandb
wandb.login()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root="../data", train=True, download=True, transforms=transform)
testset = torchvision.datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

CLASS_NAMES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship",  "truck"
)

def train(model, device, train_loader, optimizer, criterion, epoch, steps_per_epoch=20):
    model.train()
    train_loss = 0
    train_total = 0
    train_correct = 0

    for batch_idx, (data, target) in enumerate(train_loader, start=0):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        scores, predictions = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += int(sum(predictions == target))
        optimizer.zero_grad()
        loss.backwards()
        optimizer.step()
    
    acc = round((train_correct / train_total) * 100, 2)
    print(f"Epoch [{epoch}], Loss: {train_loss/train_total}, Accuracy: {acc}")
    wandb.log({
        "Train Loss": train_loss/train_total,
        "Train Accuracy": acc,
        "Epoch": epoch
    })

def test(model, device, test_loader, criterion, classes):
    model.eval()
    test_loss = 0
    test_total = 0
    test_correct = 0

    example_images = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            scores, predictions = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += int(sum(predictions == target))
    acc = round((test_correct / test_total) * 100, 2)
    print(f"Test Loss: {test_loss/test_total}, Test Accuracy: {acc}")
    wandb.log({
        "Test Loss": test_loss/test_total,
        "Test Accuracy": acc
    })

