import torch
from dataset import MNIST
from vit import ViT
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

DEVICE='cuda' if torch.cuda.is_available() else 'cpu' 

dataset=MNIST()
model=ViT().to(DEVICE)
model.load_state_dict(torch.load('model.pth'))
model.eval()

img,label=dataset[0]
print('Correct:',label)
plt.imshow(img.permute(1,2,0))
plt.show()

logits=model(img.unsqueeze(0).to(DEVICE)) #[1, C, H, W]
print('Prediction:',logits.argmax(-1).item())

