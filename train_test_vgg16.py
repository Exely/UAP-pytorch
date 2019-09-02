# coding: utf-8
import torch
from universal import universal_adversarial_perturbation
from prepare import fooling_rate, get_model

device = ('cuda' if torch.cuda.is_available() else 'cpu')
path_train_imagenet='/data/dongcheng/imgnet/train/'
path_test_imagenet='/data/dongcheng/imgnet/val/'

# noise=torch.load('noise_load.pth')

model = get_model('vgg16', device)

# compute noise
noise = universal_adversarial_perturbation(path_train_imagenet, model, device)
torch.save(noise, 'noise_load_vgg16.pth')
print("saving noise_load.pth")

print("vgg16:")
model = get_model('vgg16', device)
fr = fooling_rate(path_test_imagenet, noise, model, device)

print("vgg19:")
model = get_model('vgg19', device)
fr = fooling_rate(path_test_imagenet, noise, model, device)

'''
print("resnet18:")
model=get_model('resnet18',device)
fr,model = fooling_rate(path_test_imagene,tnoise,model, device)
'''

print("resnet152:")
model = get_model('resnet152', device)
fr = fooling_rate(path_test_imagenet, noise, model, device)

print("alexnet:")
model = get_model("alexnet", device)
fr = fooling_rate(path_test_imagenet, noise, model, device)