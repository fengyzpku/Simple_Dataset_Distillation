# RaT-BPTT
Random Truncated Backpropagation through Time for dataset distillation.


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
python main.py --dataset cifar10 --num_per_class 10 --batch_per_class 10 --num_train_eval 8 --world_size 1 --rank 0 --batch_size 5000 --ddtype curriculum --cctype 2 --epoch 60000 --test_freq 25 --print_freq 10 --arch convnet --window 60 --minwindow 0 --totwindow 200 --inner_optim Adam --inner_lr 0.001 --lr 0.001 --zca --syn_strategy flip_rotate --real_strategy flip_rotate --fname 60_200 --seed 0
```

In the above script, we use batch size 5000, window size 60, unroll length 200, and the Adam optimizer with 0.001 learning rate in both the inner loop and the outer loop. 

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