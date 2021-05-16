# Signing the Supermask: Hide and Invert

## Authors

This codebase was developed by Nils Koster in 2020.

## Background
This repository implements the experiments conducted in the paper *Signing the Supermask: Hide and Invert*.
They present a novel approach that trains a given neural network by simply selecting important weights, possibly inverting their sign and dropping the rest.
This proposition extends the work of [Zhou2019](https://arxiv.org/abs/1905.01067) ([Code](https://github.com/uber-research/deconstructing-lottery-tickets)) and [Ramanujan2019](https://arxiv.org/abs/1911.13299) ([Code](https://github.com/allenai/hidden-networks)) who consider masking but not sign-inverting the initial weights and follow the findings of the Lottery Ticket Hypothesis [Frankle2018](https://arxiv.org/abs/1803.03635) ([Code](https://github.com/google-research/lottery-ticket-hypothesis)).
Through this extension and adaptations of initialization methods, they concurrently gain a pruning rate of 96% to 99% while matching or exceeding the performance of various baseline models and the current literature.
This approach has three main advantages.
First, it yields very sparse networks that can be stored more efficiently, in addition to faster inference.
Second, by reducing the neural network to its very foundation, we gain insights into which weights matter for performance, thus improving the network’s interpretability. 
What stands out, however, is that while signed Supermask models heavily reduce a model's complexity, they nonetheless perform well on given tasks.
The code for the training algorithm and the conducted experiments can be found online:

## Structure

The code is structured such that we only need to specify a few parameters such as the network architecture (among the available FCN, Conv2, Conv4, Conv6 and Conv8) and learning hyperparameters along with the path to the previously initialized weights. This way, the actual training process is speed up and we the different all base on the same initial weights/masks.

             | FCN | Conv2 | Conv4 | Conv6 | Conv 8
------------ | --- | ----- | ----- | ----- | ----- 
Conv <\br> Layers | | 64,64,pool | 64,64,pool <\br> 128,128,pool | 64,64,pool <\br> 128,128,pool <\br> 256,256,pool <\br> 64,64,pool <\br> 128,128,pool <\br> 256,256,pool <\br> 512,512,pool 
FC Layers | 300,100,10 | 256,256,10 | 256,256,10 | 256,256,10 | 256,256,10 
Parameter Count | 266.200 | 4.300.992 | 2.425.024 | 2.261.184 | 5.275.840
