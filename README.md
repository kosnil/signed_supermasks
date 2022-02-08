# Signing the Supermask: Keep, Hide, Invert

## Background
This repository implements the experiments conducted in our paper [*Signing the Supermask: Hide and Invert*](https://openreview.net/forum?id=e0jtGTfPihs), ICLR 2022 or on [arXiv](https://arxiv.org/abs/2201.13361).
We present a novel approach that trains a given neural network by simply selecting important weights, possibly inverting their sign and dropping the rest.
This proposition extends the work of [Zhou et al.](https://arxiv.org/abs/1905.01067) ([Code](https://github.com/uber-research/deconstructing-lottery-tickets)) and [Ramanujan et al.](https://arxiv.org/abs/1911.13299) ([Code](https://github.com/allenai/hidden-networks)) who consider masking but not sign-inverting the initial weights and follow the findings of the Lottery Ticket Hypothesis [Frankle2018](https://arxiv.org/abs/1803.03635) ([Code](https://github.com/google-research/lottery-ticket-hypothesis)).
Through this extension and adaptations of initialization methods, we concurrently gain a pruning rate of 96% to 99% while matching or exceeding the performance of various baseline models and the current literature. 

Experiments can be run on all investigated architectures in the paper, including a simple feed-forward neural network, four differently sized VGG-like CNNs as well as three ResNets. The table below shows the architecture of FCN, and the different VGG-like CNN architectures.

|             | FCN | Conv2 | Conv4 | Conv6 | Conv8 |
|------------ | :-: | :---: | :---: | :---: | :---: |
| Conv <br/> Layers | | 64,64,pool | 64,64,pool <br/> 128,128,pool | 64,64,pool <br/> 128,128,pool <br/> 256,256,pool | 64,64,pool <br/> 128,128,pool <br/> 256,256,pool <br/> 512,512,pool |
| FC Layers | 300,100,10 | 256,256,10 | 256,256,10 | 256,256,10 | 256,256,10 |
| Parameter Count | 266.200 | 4.300.992 | 2.425.024 | 2.261.184 | 5.275.840 |

While FCN was trained on MNIST only, the CNN architectures (Conv2 - Conv8) were trained on CIFAR10.
Furthermore, we trained ResNet20s with variying width on CIFAR10, as well as ResNet56 and ResNet110 on CIFAR100. 

The following table lists our results in terms of mean test accuracy and mean remaining weights over all conducted experiments. 
Normally trained architectures act as baseline.

| Model         | Dataset   | Baseline Acc. \[%\] | Signed Supermask Acc. \[%\]   | Params    | Rem. Weights     |
| :------------ | :-------: | :-----------------: | :--------------------------:  | :-----:   | :--------------: |
| FCN           | MNIST     | 97.42               | 97.48                         | 0.27 M    | 3.77 % / 10.1 K  |
| Conv2         | CIFAR10   | 68.79               | 68.37                         | 4.3 M     | 0.60 % / 25.8 K  |
| Conv4         | CIFAR10   | 74.50               | 77.40                         | 2.4 M     | 2.91 % / 69.8 K  | 
| Conv6         | CIFAR10   | 75.91               | 79.17                         | 2.3 M     | 2.36 % / 54.3 K  | 
| Conv8         | CIFAR10   | 72.24               | 80.91                         | 5.3 M     | 1.17 % / 62.0 K  |
| ResNet20      | CIFAR10   | 84.91               | 81.68                         | 0.25 M    | 21.13 % / 52.8 K | 
| ResNet20x2    | CIFAR10   | 86.80               | 84.42                         | 1.0 M     | 7.69 / 76.9 K        | 
| ResNet20x3    | CIFAR10   | 87.32               | 84.89                         | 2.2 M     | 4.06 / 89.3 K        | 
| ResNet56      | CIFAR100  | 68.04               | 60.01                         | .84 M     | 29.39 / 247 K        | 
| ResNet110     | CIFAR100  | 62.70               | 46.42                         | 1.7 M     | 20.64 / 351 K        | 


For further and more detailed results as well as comparisons to existing literature, we refer to our paper. For instance, we vary weight and mask initialization, investigate the influence of batch normalization for the investigated ResNets and prune the networks to the extreme.

## Python requirements

The code was written and tested on

- Python 3.7
- Tensorflow 2.7 (we strongly recommend a GPU)
- NumPy 1.21


## Structure

We shortly explain the main structure of the repository and explain each file. Generally, the code is structured such that it is easy to reproduce the experiments. All parameters are defined in a .yaml-config (all configs can be found in the configs folder).


- `custom_layers.py` holds, as the name suggests, all customized layers. These not only include the Dense and Conv2D layers modified for signed Supermasks, but also some standard layers which are just slightly modified to fit the whole pipeline.
- `dense_networks.py` contains only the two FCN variants (that is the baseline and the signed Supermask version)
- `conv_networks.py` contains all CNN architectures
- `resnet_networks.py` contains all ResNet architectures as well as the ResNet blocks
- `data_preprocessor.py` holds all functionality regarding data handling and preprocessing
- `weight_initializer.py` includes all common initialization schemes (i.e. He, Xavier) as well as ELU/S for weights and masks. It is possible to initialize models directly with newly created weights and to save weights for later use as well as set weights from a specific file/array. `init_example.py` provides a code snippet, that lets you create and save a defined model's weights and masks. If you'd like to create weights on the fly, set the parameters accordingly - see e.g. `resnet20_elu_baseline.yaml`.
- `model_trainer.py` is used to train the models, both baselines and signed Supermasks.
- `experiment_looper.py` stitches all previously mentioned files together to a single pipeline, such that training becomes easy. A user merely passes the path to the config file and the model is then trained. Models are not being saved, due to the high number of experiments in the paper.
- `minimal_working_example.py` is a very minimalistic working example. The config files should be self-explanatory after a brief examination.



