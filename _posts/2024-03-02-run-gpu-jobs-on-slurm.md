---
title: Run Jobs on Slurm systems with GPUs
author: yuanjian
date: 2024-03-02 20:48:00 -0500
categories: [Blogging, Tutorial]
tags: [Slurm, GPU]
pin: true
---

First of all, when you are on a login node, you usually have no access to a GPU. If you want to test PyTorch with a GPU environment, it is better to enter a GPU session first.

```bash
# To get access to GPU, do this
srun --gres=gpu:1 -p gpu2 --pty ~bin/bash
```

I'll show you an SBATCH script example to train a simple Convolutional Neural Network (CNN) with PyTorch.

```bash
#!/bin/bash
#SBATCH --job-name=simplecnn
#SBATCH --output=simplecnn.out
#SBATCH --error=simplecnn.err
#SBATCH --account=pi-foster
#SBATCH --time=00:10:00
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=ALL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yuanjian@uchicago.edu  # Where to send email

module load Anaconda3/2022.10
module load pytorch/1.2

python train_simeplecnn.py
```

And the Python script is shown below. We train a simple CNN network on the CIFAR10 dataset. Note that in supercomputers, the computation nodes sometimes have no access to the Internet. It is better to prepare the datasets beforehand and store them in your supercomputer's filesystem.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import time

import logging
import os
from datetime import datetime
from pathlib import Path
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)     
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

current_time = datetime.now()
log_prefix = f"logs/{current_time.month}-{current_time.day}-{current_time.year}_{current_time.hour}-{current_time.minute}-{current_time.second}"
os.makedirs(log_prefix)

logger = setup_logger('app_logger', f'{log_prefix}/app.log')



# Check if GPU is available
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

print(device)
logger.info(f"Using device: {device}")

# Define the transformation for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Define a simple convolutional neural network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model
model = SimpleCNN().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
start_time = time.time()
for epoch in range(20):  # More epochs can be added to extend training time
    running_loss = 0.0
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 200 == 199:  # Print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            
            model.eval()
            with torch.inference_mode():
                correct = 0
                total = 0
                for data in testloader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                accuracy = 100 * correct / total
            print(f"epoch: {epoch + 1}, accuracy: {accuracy}, total: {total}, correct: {correct}")
            logger.info(f"epoch: {epoch + 1}, loss: {running_loss / 200:.3f}, accuracy: {accuracy}, total: {total}, correct: {correct}")
            running_loss = 0.0

print("Training finished. Total training time:", time.time() - start_time, "seconds")
logger.info(f"Training finished. Total training time: {time.time() - start_time} seconds")

MODEL_PATH = "trained_model.pth"
# Save the trained model
torch.save(model.state_dict(), MODEL_PATH)
print("Trained model saved.")
logger.info(f"Trained model saved to {Path(os.getcwd()) / MODEL_PATH}")
```