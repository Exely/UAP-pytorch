# coding: utf-8
import torch
import torchvision.models as models
from tqdm import tqdm
import numpy as np
import os
from scipy.misc import imread, imresize


def get_model(model, device):
    if model == 'vgg16':
        net = models.vgg16(pretrained=True)
    elif model == 'vgg19':
        net = models.vgg19(pretrained=True)
    elif model == 'resnet18':
        net = models.resnet18(pretrained=True)
    elif model == 'resnet152':
        net = models.resnet152(pretrained=True)
    elif model == 'alexnet':
        net = models.alexnet(pretrained=True)
    elif model == 'inception':
        net = models.inception_v3(pretrained=True)
    elif model == 'googlenet':
        net = models.googlenet(pretrained=True)
    # pytorch0.4 has no googlenet
    net.eval()
    net = net.to(device)
    return net


def preprocess_image_batch(image_paths, img_size=None, crop_size=None, color_mode="rgb", out=None):
    img_list = []

    for im_path in image_paths:
        img = imread(im_path, mode='RGB')
        if img_size:
            img = imresize(img, img_size)

        img = img.astype('float32')
        # We normalize the colors (in RGB space) with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        # We permute the colors to get them in the BGR order
        # if color_mode=="bgr":
        #    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]

        if crop_size:
            img = img[(img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2, (img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2, :]

        img_list.append(img)

    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        raise ValueError('when img_size and crop_size are None, images'
                         ' in image_paths must have the same shapes.')

    if out is not None and hasattr(out, 'append'):
        out.append(img_batch)
    else:
        return img_batch


def undo_image_avg(img):
    img_copy = np.copy(img)
    img_copy[:, :, 0] = img_copy[:, :, 0] + 123.68
    img_copy[:, :, 1] = img_copy[:, :, 1] + 116.779
    img_copy[:, :, 2] = img_copy[:, :, 2] + 103.939
    return img_copy


def fooling_rate(path_test_imagenet, v, model, device):
    """
    :path_test_imagenet: path to test dataset
    :v: Noise Matrix
    :model: target network
    :device: PyTorch device
    """
    fooled = 0.0
    files = os.walk(path_test_imagenet).next()[2]
    torch.cuda.empty_cache()
        
    for img in tqdm(files):
        path_img = os.path.join(path_test_imagenet,img)
        image = preprocess_image_batch([path_img],img_size=(256,256), crop_size=(224,224), color_mode="rgb")
        image = np.transpose(image, (0, 3, 1, 2))
        image = torch.from_numpy(image)
        image = image.to(device)
        _, pred = torch.max(model(image),1)
        _, adv_pred = torch.max(model(image+v),1)
        
        if pred != adv_pred:
            fooled += 1

    num_images = len(files)
    # Compute the fooling rate
    fr = fooled/num_images
    print('Fooling Rate = ', fr, 'fooled=', fooled, 'num_images=', num_images)
    return fr