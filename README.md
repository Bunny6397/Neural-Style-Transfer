# Neural-Style-Transfer
# 🎨 Neural Style Transfer with PyTorch

This project implements a **Neural Style Transfer** model using **PyTorch**, which blends the content of one image with the style of another. It generates an image that looks like the content photo painted in the style of the artwork.

## 🧠 Concept

Neural Style Transfer (NST) is a deep learning technique that uses a **pre-trained convolutional neural network** (VGG19) to extract style and content representations and optimize a new image to combine both.

## 🛠️ Requirements

Make sure you have Python 3 and the following libraries:

```bash
pip install torch torchvision matplotlib pillow
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import copy

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loading and preprocessing
def image_loader(image_path, max_size=400):
    image = Image.open(image_path).convert('RGB')
    size = max(max(image.size), max_size)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image.to(device, torch.float)

# Display tensor as image
def imshow(t
