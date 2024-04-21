Official PyTorch Implementation for the ["Embarrassingly Simple Dataset Distillation"](https://openreview.net/forum?id=PLoWVP7Mjc) paper. Published at the *Twelfth International Conference on Learning Representations, 2024*.

**Embarrassingly Simple Dataset Distillation**

Yunzhen Feng, Ramakrishna Vedantam, Julia Kempe

**Abstract**: Dataset distillation extracts a small set of synthetic training samples from a large dataset with the goal of achieving competitive performance on test data when trained on this sample. In this work, we tackle dataset distillation at its core by treating it directly as a bilevel optimization problem. Re-examining the foundational back-propagation through time method, we study the pronounced variance in the gradients, computational burden, and long-term dependencies. We introduce an improved method: Random Truncated Backpropagation Through Time (RaT-BPTT) to address them. RaT-BPTT incorporates a truncation coupled with a random window, effectively stabilizing the gradients and speeding up the optimization while covering long dependencies. This allows us to establish new state-of-the-art for a variety of standard dataset benchmarks. A deeper dive into the nature of distilled data unveils pronounced intercorrelation. In particular, subsets of distilled datasets tend to exhibit much worse performance than directly distilled smaller datasets of the same size. Leveraging RaT-BPTT, we devise a boosting mechanism that generates distilled datasets that contain subsets with near optimal performance across different data budgets.


# RaT-BPTT
Random Truncated Backpropagation through Time for dataset distillation.

**TL;DR** We propose RaT-BPTT, a new algorithm for Dataset distillation, which sets SOTA across various benchmarks. The main idea is to return to the core of the bilevel optimization and to carefully study the metagradients through the unrolling step. This leads to the **Ra**ndom **T**runcated **B**ackpropagation **T**hrough **T**ime (RaT-BPTT).

## Project Structure
This project consists of:
- [`main.py`](./main.py) - Main entry of the code.
- [`framework/base.py`](./framework/base.py) - Worker function for the distillation.
- [`framework/distill_higher.py`](./framework/distill_higher.py) - RaT-BPTT distillation function class.
- [`framework/config.py`](./framework/config.py) - Config functions for data handling and network classes.
- [`framework/metric.py`](./framework/metric.py) - Metric functions.
- [`framework/convnet.py`](./framework/convnet.py) - Convnets.
- [`framework/model.py`](./framework/model.py) - ResNets.
- [`framework/vgg.py`](./framework/metric.py) - VGG and AlexNet.

## Installation

```
> conda env create -f environment.yml
> conda activate ffcv
```

## Example Usage 

Before running the script, please install the environment in environment.yml. The key package here is the Higher package (https://github.com/facebookresearch/higher).

To distill on CIFAR-10 with 10 images per class:

```
python main.py --dataset cifar10 --num_per_class 10 --batch_per_class 10 --num_train_eval 8 \
 --world_size 1 --rank 0 --batch_size 5000 --ddtype curriculum --cctype 2 --epoch 60000 \ 
 --test_freq 25 --print_freq 10 --arch convnet --window 60 --minwindow 0 --totwindow 200 \ 
 --inner_optim Adam --inner_lr 0.001 --lr 0.001 --zca --syn_strategy flip_rotate \ 
 --real_strategy flip_rotate --fname 60_200 --seed 0
```

In the above script, we use batch size 5000, window size 60, unroll length 200, and the Adam optimizer with 0.001 learning rate in both the inner loop and the outer loop. In your GPU setting, you should always select the largest 

To distill on CIFAR-100 with 10 images per class

```
python main.py --dataset cifar100 --num_per_class 10 --batch_per_class 1 --train_y \ 
--task_sampler_nc 100 --num_train_eval 8 --world_size 1 --rank 0 --batch_size 5000 \ 
--ddtype curriculum --cctype 2 --epoch 60000 --test_freq 25 --print_freq 10 --arch convnet \ 
--window 100 --minwindow 0 --totwindow 300 --inner_optim Adam --inner_lr 0.001 --lr 0.001 \ 
--zca --syn_strategy flip_rotate --real_strategy flip_rotate --fname train_y
```

To distill on Tiny-ImageNet with 10 images per class

```
python main.py --dataset tiny-imagenet-200 --num_per_class 10 --batch_per_class 1 --task_sampler_nc 50 \ 
--train_y --num_train_eval 8 --world_size 1 --rank 0 --batch_size 1000 \ 
--ddtype curriculum --cctype 2 --epoch 60000 --test_freq 10 --print_freq 10 --arch convnet4 \ 
--window 100 --minwindow 0 --totwindow 300 --inner_optim Adam --inner_lr 0.001 --lr 0.0003 \ 
--syn_strategy flip_rotate --real_strategy flip_rotate --fname test 
```

### Stabilizing the Optimization

We have conducted further analyses on optimizing the distilled dataset. The Bash scripts described previously initiate the back-propagation window from the zeroth step, corresponding to the random initialization of the inner network. However, the early stages of neural network training typically experience significant chaotic phases. During these phases, the gradient signal is notably noisy, suggesting that positioning the window at this initial stage may not be ideal. Instead, it might be beneficial to identify scaling constants through validation experiments, which can determine a more effective starting point. Such a starting point could potentially enhance performance and reduce optimization time in subsequent Bash scripts. Identifying these constants, however, is not straightforward, and the more general random approach remains a versatile alternative.

To distill on CIFAR-100 with 1 images per class

```
python main.py --dataset cifar100 --num_per_class 1 --batch_per_class 1 --task_sampler_nc 100 --num_train_eval 8 \ 
--train_y --world_size 1 --rank 0 --batch_size 5000 \ 
--ddtype curriculum --cctype 0 --epoch 60000 --test_freq 50 --print_freq 10 --arch convnet \ 
--window 90 --minwindow 0 --totwindow 130 --inner_optim Adam --inner_lr 0.001 --lr 0.0003 \ 
--zca --syn_strategy flip_rotate --real_strategy flip_rotate 
```

To distill on Tiny-ImageNet with 10 images per class

```
python main.py --dataset tiny-imagenet-200 --num_per_class 10 --batch_per_class 1 --task_sampler_nc 50 \ 
--train_y --num_train_eval 8 --world_size 1 --rank 0 --batch_size 2000 \ 
--ddtype curriculum --cctype 0 --epoch 60000 --test_freq 10 --print_freq 10 --arch convnet4 \ 
--window 90 --minwindow 0 --totwindow 270 --inner_optim Adam --inner_lr 0.001 --lr 0.0003 \ 
--syn_strategy flip_rotate --real_strategy flip_rotate 
```

To distill on ImageNet with 1 images per class

```
python main.py --dataset imagenet --train_y --num_per_class 1 --batch_per_class 1 --task_sampler_nc 50 \ 
--num_train_eval 8 --world_size 1 --rank 0 --batch_size 1500 \ 
--ddtype curriculum --cctype 0 --epoch 60000 --test_freq 1 --print_freq 10 --arch convnet4 \ 
--window 80 --minwindow 0 --totwindow 280 --inner_optim Adam --inner_lr 0.001 --lr 0.0003 \ 
--syn_strategy flip_rotate --real_strategy flip_rotate --workers 4
```

## Citation

If you find this useful for your research, please cite our paper:

```
@inproceedings{feng2023embarrassingly,
  title={Embarrassingly Simple Dataset Distillation},
  author={Feng, Yunzhen and Vedantam, Shanmukha Ramakrishna and Kempe, Julia},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```