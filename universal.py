# coding: utf-8
from deepfool import deepfool
import numpy as np
import torch
import os
from tqdm import tqdm
from prepare import fooling_rate, preprocess_image_batch

def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v

def universal_adversarial_perturbation(path_train_imagenet, model, device, xi=10, delta=0.2, max_iter_uni=10, p=np.inf,
                                       num_classes=10, overshoot=0.02, max_iter_df=10, t_p=0.2):
    """
        :param path_train_imagenet: path to train dataset
        :param model: target network
        :param device: PyTorch Device
        :param xi: controls the l_p magnitude of the perturbation
        :param delta: controls the desired fooling rate (default = 80% fooling rate)
        :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = 10*num_images)
        :param p: norm to be used (default = np.inf)
        :param num_classes: For deepfool: num_classes (limits the number of classes to test against, by default = 10)
        :param overshoot: For deepfool: used as a termination criterion to prevent vanishing updates (default = 0.02).
        :param max_iter_df:For deepfool: maximum number of iterations for deepfool (default = 10)
        :param t_p:For deepfool: truth perentage, for how many flipped labels in a batch.(default = 0.2)

        :return: the universal perturbation matrix.
    """

    v = torch.zeros(1, 3, 224, 224).to(device)
    v.requires_grad_()

    fr = 0.0
    itr = 0
    files = os.walk(path_train_imagenet).next()[2]

    while fr < 1 - delta and itr < max_iter_uni:
        torch.cuda.empty_cache()
        np.random.shuffle(files)
        # Iterate over the dataset and compute the purturbation incrementally
        pbar = tqdm(files)
        pbar.set_description('Starting pass number ' + str(itr))
        for img in pbar:
            path_img = os.path.join(path_train_imagenet,img)
            image = preprocess_image_batch([path_img],img_size=(256,256), crop_size=(224,224), color_mode="rgb")
            image = np.transpose(image, (0, 3, 1, 2))
            with torch.no_grad():
                image = torch.from_numpy(image)
                image = image.to(device)
                _, pred = torch.max(model(image), 1)
                _, adv_pred = torch.max(model(image + v), 1)

            if pred == adv_pred:
                dr, iter, _, _, _ = deepfool((image + v).detach()[0], model, device, num_classes=num_classes,
                                             overshoot=overshoot, max_iter=max_iter_df)
                if iter < max_iter_df - 1:
                    with torch.no_grad():
                        v = v + torch.from_numpy(dr).to(device)
                        v = proj_lp(v, xi, p).to(device)
            del _, pred, adv_pred, img
            pbar.set_description('Norm of v: ' + str(torch.norm(v).detach().cpu().numpy()))
            torch.cuda.empty_cache()
        with torch.no_grad():
            fr = fooling_rate(path_train_imagenet, v, model, device)
        itr = itr + 1

    return v