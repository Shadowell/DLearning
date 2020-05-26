import torch
from torch import nn
from torchvision import models
print(torch.cuda.is_available())
vgg = models.vgg16()
print(len(vgg.features), len(vgg.classifier))
