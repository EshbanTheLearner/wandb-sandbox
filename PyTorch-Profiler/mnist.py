import glob
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from torch.profiler import tensorboard_trace_handler
import wandb

torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors if not mirror.startswith("http://yann.lecun.com")]

wandb.login()

OPTIMIZERS = {
    "Adadelta": optim.Adadelta,
    "Adagrad": optim.Adagrad,
    "SGD": optim.SGD
}

class Net(pl.LightningModule):
    def __init__(self, optimizer="Adadelta"):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.optimizer = self.set_optimizer(optimizer)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    def set_optimizer(self, optimizer):
        return OPTIMIZERS[optimizer]
    

def training_step(self, batch, idx):
    inputs, labels = batch
    outputs = self(inputs)
    loss = F.nll_loss(outputs, labels)
    return {
        "loss": loss
    }

def configure_optimizers(self):
    return self.optimizer(self.parameters(), lr=0.1)

Net.training_step = training_step
Net.configure_optimizers = configure_optimizers

class TorchTensorboardProfilerCallback(pl.Callback):
    def __init__(self, profiler):
        super().__init__()
        self.profiler = profiler
    
    def on_train_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self.profiler.step()
        pl_module.log_dict(outputs)

config = {
    "batch_size": 32,
    "num_workers": 0,
    "pin_memory": False,
    "precision": 32,
    "optimizer": "Adadelta"
}

with wandb.init(project="mnist-trace", config=config) as run:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=wandb.config.batch_size,
        num_workers=wandb.config.num_workers,
        pin_memory=wandb.config.pin_memory
    )

    model = Net(optimizer=wandb.config["optimizer"])

    wait, warmup, active, repeat = 1, 1, 2, 1
    total_steps = (wait + warmup + active) * (1 + repeat)
    schedule = torch.profiler.schedule(
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat
    )
    profiler = torch.profiler.profile(
        schedule=schedule,
        on_trace_ready=tensorboard_trace_handler("wandb/latest-run/tbprofile"),
        with_stack=False
    )

    with profiler:
        profiler_callback = TorchTensorboardProfilerCallback(profiler)
        trainer = pl.Trainer(
            gpus=0,
            max_epochs=1,
            max_steps=total_steps,
            logger=pl.loggers.WandbLogger(
                log_model=True, 
                save_code=True
            ),
            callbacks=[profiler_callback],
            precision=wandb.config.precision
        )
        trainer.fit(model, trainloader)
    profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
    profile_art.add_file(glob.glob("wandb/latest-run/tbprofile/*.pt.trace.json")[0], "trace.pt.trace.json")
    run.log_artifact(profile_art)