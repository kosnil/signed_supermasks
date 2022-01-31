# Signing the Supermask: Keep, Hide, Invert

## Background
This repository implements the experiments conducted in the paper *Signing the Supermask: Hide and Invert*, ICLR 2022 or on arXiv.
They present a novel approach that trains a given neural network by simply selecting important weights, possibly inverting their sign and dropping the rest.
This proposition extends the work of [Zhou2019](https://arxiv.org/abs/1905.01067) ([Code](https://github.com/uber-research/deconstructing-lottery-tickets)) and [Ramanujan2019](https://arxiv.org/abs/1911.13299) ([Code](https://github.com/allenai/hidden-networks)) who consider masking but not sign-inverting the initial weights and follow the findings of the Lottery Ticket Hypothesis [Frankle2018](https://arxiv.org/abs/1803.03635) ([Code](https://github.com/google-research/lottery-ticket-hypothesis)).
Through this extension and adaptations of initialization methods, they concurrently gain a pruning rate of 96% to 99% while matching or exceeding the performance of various baseline models and the current literature.

Experiments can be run on the five different neural network architectures investigated in the paper. The table below shows the architecture of FCN, a simple feed-forward neural network as well as different VGG-like CNN architectures.

|             | FCN | Conv2 | Conv4 | Conv6 | Conv8 |
|------------ | :-: | :---: | :---: | :---: | :---: |
| Conv <br/> Layers | | 64,64,pool | 64,64,pool <br/> 128,128,pool | 64,64,pool <br/> 128,128,pool <br/> 256,256,pool | 64,64,pool <br/> 128,128,pool <br/> 256,256,pool <br/> 512,512,pool |
| FC Layers | 300,100,10 | 256,256,10 | 256,256,10 | 256,256,10 | 256,256,10 |
| Parameter Count | 266.200 | 4.300.992 | 2.425.024 | 2.261.184 | 5.275.840 |

While FCN was trained on MNIST only, the CNN architectures (Conv2 - Conv8) were trained on CIFAR-10.
Furthermore, we trained ResNet20s with variying width on CIFAR-10, as well as ResNet56 and ResNet110 on CIFAR-100. 

The following table lists our results in terms of test accuracy and remaining weights in comparison to the most related work:

|       |                      |                |      |       |                      |         |                    |       |                     |
|:------|:--------------------:|:--------------:|:----:|:-----:|:--------------------:|:-------:|:------------------:|:-----:|:-------------------:|
|       |                      | Accuracy \[%\] |      |       |                      |         | Rem. Weights \[%\] |       |                     |
|       |       Baseline       |      [Zhou](https://arxiv.org/abs/1905.01067)      | [Ram.](https://arxiv.org/abs/1911.13299) | [Diff.](https://arxiv.org/abs/2103.09377) |    Sig. Supermask    |  Zhou   |        Ram.        | Diff. |   Sig. Supermask    |
|:------|:--------------------:|:--------------:|:----:|:-----:|:--------------------:|:-------:|:------------------:|:-----:|:-------------------:|
| FCN   | 97.43 \[97.4, 97.5\] |      98.0      |  \-  |  \-   | 97.48 \[97.3, 97.7\] | 11 - 93 |         \-         |  \-   | 3.77 \[3.7,3.8\]    |
|:------|:--------------------:|:--------------:|:----:|:-----:|:--------------------:|:-------:|:------------------:|:-----:|:-------------------:|
| Conv2 | 68.79 \[68.4, 69.2\] |      66.0      |  65  |  70   | 68.37 \[67.7, 69.0\] | 11 - 93 |         10         |  10   | 0.60 \[0.58, 0.62\] |
| Conv4 | 74.50 \[74.7, 75.3\] |      72.5      |  74  |  79   | 77.40 \[76.7, 78.3\] | 11 - 93 |         10         |  10   |  2.91 \[2.9, 3.0\]  |
| Conv6 | 75.91 \[75.4, 76.4\] |      76.5      |  77  |  82   | 79.17 \[78.1, 80.5\] | 11 - 93 |         10         |  10   |  2.36 \[1.9, 2.6\]  |
| Conv8 | 72.24 \[71.4, 73.0\] |       \-       |  70  |  85   | 80.91 \[79.9, 81.7\] |   \-    |         10         |  10   |  1.17 \[1.1, 1.2\]  |

For further and more detailed results, we refer to our paper. For instance, we vary weight and mask initialization, investigate the influence of batch normalization for the investigated ResNets and prune the networks to the extreme.

## Python requirements

The code was written and tested on

- Python 3.7
- Tensorflow 2.7 (we strongly recommend a GPU)
- NumPy 1.21


## Structure

The code is structured such that we only need to specify a few parameters such as the network architecture (among the available FCN, Conv2, Conv4, Conv6 and Conv8) and learning hyperparameters along with the path to the previously initialized weights. This way, the actual training process is speed up and we the different all base on the same initial weights/masks.

### Weights / Masks

Since each weight file is at least 30MB (stored as pickle), we do not provide those in this codebase. However, the file `init_example.py` shows how to easily create initial weight and mask values for each architecture considered.


