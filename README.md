# Signing the Supermask: Keep, Hide, Invert

## Background
This repository implements the experiments conducted in the paper *Signing the Supermask: Hide and Invert*, ICLR 2022 or on arXiv.
They present a novel approach that trains a given neural network by simply selecting important weights, possibly inverting their sign and dropping the rest.
This proposition extends the work of [Zhou et al.](https://arxiv.org/abs/1905.01067) ([Code](https://github.com/uber-research/deconstructing-lottery-tickets)) and [Ramanujan et al.](https://arxiv.org/abs/1911.13299) ([Code](https://github.com/allenai/hidden-networks)) who consider masking but not sign-inverting the initial weights and follow the findings of the Lottery Ticket Hypothesis [Frankle2018](https://arxiv.org/abs/1803.03635) ([Code](https://github.com/google-research/lottery-ticket-hypothesis)).
Through this extension and adaptations of initialization methods, they concurrently gain a pruning rate of 96% to 99% while matching or exceeding the performance of various baseline models and the current literature.

Experiments can be run on the five different neural network architectures investigated in the paper. The table below shows the architecture of FCN, a simple feed-forward neural network as well as different VGG-like CNN architectures.

|             | FCN | Conv2 | Conv4 | Conv6 | Conv8 |
|------------ | :-: | :---: | :---: | :---: | :---: |
| Conv <br/> Layers | | 64,64,pool | 64,64,pool <br/> 128,128,pool | 64,64,pool <br/> 128,128,pool <br/> 256,256,pool | 64,64,pool <br/> 128,128,pool <br/> 256,256,pool <br/> 512,512,pool |
| FC Layers | 300,100,10 | 256,256,10 | 256,256,10 | 256,256,10 | 256,256,10 |
| Parameter Count | 266.200 | 4.300.992 | 2.425.024 | 2.261.184 | 5.275.840 |

While FCN was trained on MNIST only, the CNN architectures (Conv2 - Conv8) were trained on CIFAR10.
Furthermore, we trained ResNet20s with variying width on CIFAR10, as well as ResNet56 and ResNet110 on CIFAR100. 

The following table lists our results in terms of mean test accuracy and mean remaining weights over all conducted experiments. 
Normally trained architectures act as baseline.

| Model         | Dataset   | Baseline Acc. \[%\] | Signed Supermask Acc. \[%\]   | Params. | Rem. Weights [%]  |
| :------------ | :-------: | :-----------------: | :--------------------------:  | :-----: | :---------------: |
| FCN           | MNIST     | 97.42               | 97.48                         | .27 M   | 3.77              |
| Conv2         | CIFAR10   | 68.79               | 68.37                         | 4.3 M   | .60               |
| Conv4         | CIFAR10   | 74.50               | 77.40                         | 2.4 M   | 2.91              | 
| Conv6         | CIFAR10   | 75.91               | 79.17                         | 2.3 M   | 2.36              | 
| Conv8         | CIFAR10   | 72.24               | 80.91                         | 5.3 M   | 1.17              |
| ResNet20      | CIFAR10   | 84.91               | 81.68                         | .25 M   | 21.13             | 
| ResNet20x2    | CIFAR10   | 86.80               | 84.42                         | 1.0 M   | 7.69              | 
| ResNet20x3    | CIFAR10   | 87.32               | 84.89                         | 2.2 M   | 4.06              | 
| ResNet56      | CIFAR100  | 68.04               | 60.01                         | .84 M   | 29.39             | 
| ResNet110     | CIFAR100  | 62.70               | 46.42                         | 1.7 M   | 20.64             | 


For further and more detailed results as well as comparisons to existing literature, we refer to our paper. For instance, we vary weight and mask initialization, investigate the influence of batch normalization for the investigated ResNets and prune the networks to the extreme.

## Python requirements

The code was written and tested on

- Python 3.7
- Tensorflow 2.7 (we strongly recommend a GPU)
- NumPy 1.21


## Structure

The code is structured such that we only need to specify a few parameters such as the network architecture (among the available FCN, Conv2, Conv4, Conv6 and Conv8) and learning hyperparameters along with the path to the previously initialized weights. This way, the actual training process is speed up and we the different all base on the same initial weights/masks.

### Weights / Masks

Since each weight file is at least 30MB (stored as pickle), we do not provide those in this codebase. However, the file `init_example.py` shows how to easily create initial weight and mask values for each architecture considered.


