# UAP-pytorch
A simple Pytorch implementation of Universal Adversarival Pertubation proposed in [[1]](https://arxiv.org/pdf/1610.08401.pdf)      
The code is adapted from [LTS4](https://github.com/LTS4/universal) and [ferjad](https://github.com/ferjad/Universal_Adversarial_Perturbation_pytorch). Test passed on python2.7 and Pytorch0.4 .
## Usage 
### Dataset Preparation
- __Training set__: Selected 10,000 images from [ILSVRC 2012](http://www.image-net.org/challenges/LSVRC/2012/) training set in average 10 images per class.    
- __Validation set__: ILSVRC 2012 validation set (50,000 images) .    

Please modify the dataset path in [train_test_vgg16.py](train_test_vgg16.py) .
### Compute and Evaluate
```sh
python train_test_vgg16.py
```
This will download a pre-trained vgg16 model, and compute the universal perturbation on training set and evaluate fooling rate on several different models. 
### Show Saved Universal Perturbation
```sh
python show_v.py
```
## Reference
[1] S. Moosavi-Dezfooli\*, A. Fawzi\*, O. Fawzi, P. Frossard:
[*Universal adversarial perturbations*](http://arxiv.org/pdf/1610.08401), CVPR 2017
