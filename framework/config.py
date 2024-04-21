'''Framework default config'''

from framework.model import ResNet18
from framework.vgg import VGG11, AlexNet
from framework.convnet import ConvNet, ConvNet2
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder

import numpy as np

import h5py

import torch
import os


class CIFAR10Dataset(CIFAR10):
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class CIFAR100Dataset(CIFAR100):
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class DistillDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_data, list_data):
        assert len(tensor_data) == len(list_data), "Both inputs must have the same length"
        self.tensor_data = tensor_data
        self.list_data = list_data

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, index):
        return self.tensor_data[index].view(3, 32, 32), self.list_data[index]


def get_config():
    config = {
            'root': '/home/fyz/dataset/',
            'num_workers_mnist': 1,
            'num_workers_cifar10': 4,
            'num_workers_imagenet': 4
    }
    return config


def get_arch(arch, num_classes, channel, im_size, width=64):
    if arch == 'resnet18':
        return ResNet18(channel=channel, num_classes=num_classes)
    if arch == 'vgg':
        return VGG11(channel=channel, num_classes=num_classes)
    if arch == 'alexnet':
        return AlexNet(channel=channel, num_classes=num_classes)
    if arch == 'convnet':
        net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
        return ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = im_size)
    if arch == 'convnet4':
        net_width, net_depth, net_act, net_norm, net_pooling = 128, 4, 'relu', 'instancenorm', 'avgpooling'
        return ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = im_size)    
    raise NotImplementedError


def get_dataset(dataset, root, transform_train, transform_test, zca=False):
    data_root = os.path.join(root, dataset)
    process_config = None
    if dataset == 'cifar10':
        if zca:
            print('Using ZCA')
            trainset = CIFAR10Dataset(
                    root=root, train=True, download=True, transform=None)
            trainset_test = CIFAR10Dataset(
                    root=root, train=True, download=True, transform=None)
            testset = CIFAR10Dataset(
                    root=root, train=False, download=True, transform=None)
            trainset.data, testset.data, process_config = preprocess(trainset.data, testset.data, regularization=0.1)
            trainset_test.data = trainset.data.clone()
        else:
            trainset = CIFAR10(
                    root=root, train=True, download=True, transform=transform_train)
            trainset_test = CIFAR10(
                    root=root, train=True, download=True, transform=transform_test)
            testset = CIFAR10(
                    root=root, train=False, download=True, transform=transform_test)
        num_classes = 10
        shape = [3, 32, 32]
    elif dataset == 'cifar100':
        if zca:
            print('Using ZCA')
            trainset = CIFAR100Dataset(
                    root=root, train=True, download=True, transform=None)
            testset = CIFAR100Dataset(
                    root=root, train=False, download=True, transform=None)
            trainset.data, testset.data, process_config = preprocess(trainset.data, testset.data, regularization=0.1)
            trainset_test = trainset
        else:
            trainset = CIFAR100(
                    root=root, train=True, download=True, transform=transform_train)
            trainset_test = CIFAR100(
                    root=root, train=True, download=True, transform=transform_test)
            testset = CIFAR100(
                    root=root, train=False, download=True, transform=transform_test)
        num_classes = 100
        shape = [3, 32, 32]
    elif dataset == 'tiny-imagenet-200':
        shape = [3, 64, 64]
        num_classes = 200
        if zca:
            print('Using ZCA')
            # preprocess the tiny-imagenet-200 with ZCA to save time.
            db = h5py.File('./dataset/tiny-imagenet-200/zca_pro.h5', 'r')
            train_data = torch.tensor(db['train'])
            test_data = torch.tensor(db['test'])
            train_label = torch.tensor(db['train_label'])
            test_label = torch.tensor(db['test_label'])
            trainset = TensorDataset(train_data, train_label)
            trainset_test = trainset
            testset = TensorDataset(test_data, test_label)
        else:
            raise NotImplementedError
    elif dataset == 'cub-200':
        shape = [3, 32, 32]
        num_classes = 200
        if zca:
            print('Using ZCA')
            db = h5py.File('./dataset/CUB_200_2011/zca_new.h5', 'r')
            train_data = torch.tensor(db['train'])
            test_data = torch.tensor(db['test'])
            train_label = torch.tensor(db['train_label'])
            test_label = torch.tensor(db['test_label'])
            trainset = TensorDataset(train_data, train_label)
            trainset_test = trainset
            testset = TensorDataset(test_data, test_label)
        else:
            raise NotImplementedError
    elif dataset == 'imagenet':
        print('Using ImageNet')
        shape = [3, 64, 64]
        im_size = (64, 64)
        num_classes = 1000
        data_path = '/imagenet/'

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(im_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        trainset = ImageFolder(os.path.join(data_path, "train"), transform=data_transforms['train']) # no augmentation
        testset = ImageFolder(os.path.join(data_path, "val"), transform=data_transforms['val'])
        class_names = trainset.classes
        class_map = {x:x for x in range(num_classes)}
        trainset_test = trainset
        
    elif dataset == 'mnist':
        trainset = MNIST(
                root=root, train=True, download=True, transform=transform_train)
        trainset_test = MNIST(
                root=root, train=True, download=True, transform=transform_test)
        testset = MNIST(
                root=root, train=False, download=True, transform=transform_test)
        num_classes = 10
        shape = [1, 28, 28]
    else:
        raise NotImplementedError
        
    return trainset, trainset_test, testset, num_classes, shape, process_config

# remove all the ToTensor() for cifar10
def get_transform(dataset):
    print(dataset)
    if dataset == 'cifar10':
        default_transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        default_transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        print('the dataset is cifar10')
    elif dataset == 'cifar100':
        default_transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        default_transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        print('the dataset is cifar100')
    elif dataset == 'tiny-imagenet-200':
        default_transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        default_transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        print('the dataset is tiny-imagenet-200')
    elif dataset == 'imagenet':
        print('the dataset is imagenet')
        default_transform_train = None
        default_transform_test = None
    elif dataset == 'cub-200':
        default_transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        default_transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        print('the dataset is cub-200-2011')
    elif dataset == 'mnist':
        default_transform_train = transforms.Compose([
                transforms.ToTensor(),
        ])
        default_transform_test = transforms.Compose([
                transforms.ToTensor(),
        ])
    else:
        raise NotImplementedError

    return default_transform_train, default_transform_test


def get_pin_memory(dataset):
    return dataset == 'imagenet'

import torch
import numpy as np
import os
from PIL import Image, TarIO
import pickle
import tarfile

class cub200(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        super(cub200, self).__init__()

        self.root = root
        self.train = train
        self.transform = transform


        if self._check_processed():
            print('Train file has been extracted' if self.train else 'Test file has been extracted')
        else:
            self._extract()

        if self.train:
            self.train_data, self.train_label = pickle.load(
                open(os.path.join(self.root, 'processed/train.pkl'), 'rb')
            )
        else:
            self.test_data, self.test_label = pickle.load(
                open(os.path.join(self.root, 'processed/test.pkl'), 'rb')
            )

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, idx):
        if self.train:
            img, label = self.train_data[idx], self.train_label[idx]
        else:
            img, label = self.test_data[idx], self.test_label[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def _check_processed(self):
        assert os.path.isdir(self.root) == True
        assert os.path.isfile(os.path.join(self.root, 'CUB_200_2011.tgz')) == True
        return (os.path.isfile(os.path.join(self.root, 'processed/train.pkl')) and
                os.path.isfile(os.path.join(self.root, 'processed/test.pkl')))

    def _extract(self):
        processed_data_path = os.path.join(self.root, 'processed')
        if not os.path.isdir(processed_data_path):
            os.mkdir(processed_data_path)

        cub_tgz_path = os.path.join(self.root, 'CUB_200_2011.tgz')
        images_txt_path = 'CUB_200_2011/images.txt'
        train_test_split_txt_path = 'CUB_200_2011/train_test_split.txt'

        tar = tarfile.open(cub_tgz_path, 'r:gz')
        images_txt = tar.extractfile(tar.getmember(images_txt_path))
        train_test_split_txt = tar.extractfile(tar.getmember(train_test_split_txt_path))
        if not (images_txt and train_test_split_txt):
            print('Extract image.txt and train_test_split.txt Error!')
            raise RuntimeError('cub-200-1011')

        images_txt = images_txt.read().decode('utf-8').splitlines()
        train_test_split_txt = train_test_split_txt.read().decode('utf-8').splitlines()

        id2name = np.genfromtxt(images_txt, dtype=str)
        id2train = np.genfromtxt(train_test_split_txt, dtype=int)
        print('Finish loading images.txt and train_test_split.txt')
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        print('Start extract images..')
        cnt = 0
        train_cnt = 0
        test_cnt = 0
        for _id in range(id2name.shape[0]):
            cnt += 1

            image_path = 'CUB_200_2011/images/' + id2name[_id, 1]
            image = tar.extractfile(tar.getmember(image_path))
            if not image:
                print('get image: '+image_path + ' error')
                raise RuntimeError
            image = Image.open(image)
            label = int(id2name[_id, 1][:3]) - 1

            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
            image_np = np.array(image)
            image.close()

            if id2train[_id, 1] == 1:
                train_cnt += 1
                train_data.append(image_np)
                train_labels.append(label)
            else:
                test_cnt += 1
                test_data.append(image_np)
                test_labels.append(label)
            if cnt%1000 == 0:
                print('{} images have been extracted'.format(cnt))
        print('Total images: {}, training images: {}. testing images: {}'.format(cnt, train_cnt, test_cnt))
        tar.close()
        pickle.dump((train_data, train_labels),
                    open(os.path.join(self.root, 'processed/train.pkl'), 'wb'))
        pickle.dump((test_data, test_labels),
                    open(os.path.join(self.root, 'processed/test.pkl'), 'wb'))

    
class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, target_tensor):
        assert data_tensor.size(0) == target_tensor.size(0), "Data and targets must have the same number of samples"
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

# ZCA preprocess
def preprocess(train, test, zca_bias=0, regularization=0, permute=True):
    origTrainShape = train.shape
    origTestShape = test.shape

    train = np.ascontiguousarray(train, dtype=np.float32).reshape(train.shape[0], -1).astype('float64')
    test = np.ascontiguousarray(test, dtype=np.float32).reshape(test.shape[0], -1).astype('float64')

    nTrain = train.shape[0]
    
    train_mean = np.mean(train, axis=1)[:,np.newaxis]
    
    # Zero mean every feature
    train = train - np.mean(train, axis=1)[:,np.newaxis]
    test = test - np.mean(test, axis=1)[:,np.newaxis]

    # Normalize
    train_norms = np.linalg.norm(train, axis=1)
    test_norms = np.linalg.norm(test, axis=1)

    # Make features unit norm
    train = train/train_norms[:,np.newaxis]
    test = test/test_norms[:,np.newaxis]

    trainCovMat = 1.0/nTrain * train.T.dot(train)

    (E,V) = np.linalg.eig(trainCovMat)

    E += zca_bias
    sqrt_zca_eigs = np.sqrt(E + regularization * np.sum(E) / E.shape[0])
    inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
    global_ZCA = V.dot(inv_sqrt_zca_eigs).dot(V.T)
    inverse_ZCA = V.dot(np.diag(sqrt_zca_eigs)).dot(V.T)
    
    train = (train).dot(global_ZCA)
    test = (test).dot(global_ZCA)

    train_tensor = torch.Tensor(train.reshape(origTrainShape).astype('float64'))
    test_tensor  = torch.Tensor(test.reshape(origTestShape).astype('float64'))
    if permute:
        train_tensor = train_tensor.permute(0,3,1,2).contiguous()
        test_tensor  = test_tensor.permute(0,3,1,2).contiguous()

    return train_tensor, test_tensor, (inverse_ZCA, train_norms, train_mean)
