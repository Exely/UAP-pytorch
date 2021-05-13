# UAP-pytorch
A simple and UNOFFICIAL Pytorch implementation of Universal Adversarial Perturbation proposed in [[1]](https://arxiv.org/pdf/1610.08401.pdf).      
The code is adapted from [LTS4](https://github.com/LTS4/universal) and [ferjad](https://github.com/ferjad/Universal_Adversarial_Perturbation_pytorch). Test passed on python2.7 and Pytorch0.4 .
## Usage
### Dataset preparation.
- __Training set__: Random 10,000 images in 1000 classes from [ILSVRC 2012](http://www.image-net.org/challenges/LSVRC/2012/) training set.    
- __Validation set__: ILSVRC 2012 validation set (50,000 images).    

Please modify the dataset path in [train_test_vgg16.py](train_test_vgg16.py) .
### Traing and evalutaion.
```sh
python train_test_vgg16.py
```
This generates the universal perturbation on a pretrained VGG16 model and evaluates misclassifcation rate on multiple different models. 
### Visualization of generated noise.
```sh
python show_v.py
```
## Reference
[1] S. Moosavi-Dezfooli\*, A. Fawzi\*, O. Fawzi, P. Frossard:
[*Universal adversarial perturbations*](http://arxiv.org/pdf/1610.08401), CVPR 2017
