# coding: utf-8
import matplotlib.pyplot as plt
import torch
from prepare import get_model

device = ('cuda' if torch.cuda.is_available() else 'cpu')

noise = torch.load('noise_load_vgg16.pth')
model = get_model('vgg16', device)
plt.figure(dpi=200)
plt.imshow(noise[0].detach().permute(1, 2, 0).cpu().numpy())
plt.title('Perturbation')
plt.axis('off')
plt.show()