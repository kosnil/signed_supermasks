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

While FCN was trained on MNIST only, the CNN architectures (Conv2 - Conv8) were trained on CIFAR-10.
Furthermore, we trained ResNet20s with variying width on CIFAR-10, as well as ResNet56 and ResNet110 on CIFAR-100. 

The following table lists our results in terms of test accuracy and remaining weights in comparison to the most related work:


<table>
<colgroup>
<col style="width: 5%" />
<col style="width: 16%" />
<col style="width: 11%" />
<col style="width: 4%" />
<col style="width: 5%" />
<col style="width: 16%" />
<col style="width: 6%" />
<col style="width: 14%" />
<col style="width: 5%" />
<col style="width: 15%" />
</colgroup>
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th style="text-align: center;">Baseline</th>
<th style="text-align: center;"><a
href="https://arxiv.org/abs/1905.01067">Zhou et al.</a></th>
<th style="text-align: center;"><a
href="https://arxiv.org/abs/1911.13299">Ramanujan et al.</a></th>
<th style="text-align: center;"><a
href="https://arxiv.org/abs/2103.09377">Diffenderfer &amp;
Kailkhura</a></th>
<th style="text-align: center;">Sig. Supermask (Ours)</th>
<th style="text-align: center;"><a
href="https://arxiv.org/abs/1905.01067">Zhou et al.</a></th>
<th style="text-align: center;"><a
href="https://arxiv.org/abs/1911.13299">Ramanujan et al.</a></th>
<th style="text-align: center;"><a
href="https://arxiv.org/abs/2103.09377">Diffenderfer &amp;
Kailkhura</a></th>
<th style="text-align: center;">Sig. Supermask (Ours)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr class="even">
<td style="text-align: left;">FCN</td>
<td style="text-align: center;">97.43</td>
<td style="text-align: center;">98.0</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">97.48</td>
<td style="text-align: center;">11 - 93</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">3.77</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Conv2</td>
<td style="text-align: center;">68.79</td>
<td style="text-align: center;">66.0</td>
<td style="text-align: center;">65</td>
<td style="text-align: center;">70</td>
<td style="text-align: center;">68.37</td>
<td style="text-align: center;">11 - 93</td>
<td style="text-align: center;">10</td>
<td style="text-align: center;">10</td>
<td style="text-align: center;">0.60</td>
</tr>
<tr class="even">
<td style="text-align: left;">Conv4</td>
<td style="text-align: center;">74.50</td>
<td style="text-align: center;">72.5</td>
<td style="text-align: center;">74</td>
<td style="text-align: center;">79</td>
<td style="text-align: center;">77.40</td>
<td style="text-align: center;">11 - 93</td>
<td style="text-align: center;">10</td>
<td style="text-align: center;">10</td>
<td style="text-align: center;">2.91</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Conv6</td>
<td style="text-align: center;">75.91</td>
<td style="text-align: center;">76.5</td>
<td style="text-align: center;">77</td>
<td style="text-align: center;">82</td>
<td style="text-align: center;">79.17</td>
<td style="text-align: center;">11 - 93</td>
<td style="text-align: center;">10</td>
<td style="text-align: center;">10</td>
<td style="text-align: center;">2.36</td>
</tr>
<tr class="even">
<td style="text-align: left;">Conv8</td>
<td style="text-align: center;">72.24</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">70</td>
<td style="text-align: center;">85</td>
<td style="text-align: center;">80.91</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">10</td>
<td style="text-align: center;">10</td>
<td style="text-align: center;">1.17</td>
</tr>
<tr class="odd">
<td style="text-align: left;">ResNet20</td>
<td style="text-align: center;">84.91</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">81.68</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">21.13</td>
</tr>
<tr class="even">
<td style="text-align: left;">ResNet56</td>
<td style="text-align: center;">68.04</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">60.01</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">29.39</td>
</tr>
<tr class="odd">
<td style="text-align: left;">ResNet110</td>
<td style="text-align: center;">62.7</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">46.42</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">-</td>
<td style="text-align: center;">20.64</td>
</tr>
</tbody>
</table>

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


