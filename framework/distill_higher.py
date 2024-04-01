import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.multiprocessing as mp
import higher

from copy import deepcopy

import numpy as np
import random

import time


from framework.config import get_arch

def _weights_init(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        init.kaiming_normal_(m.weight)        

        
class Distill(nn.Module):
    def __init__(self, x_init, y_init, arch, window, lr, num_train_eval, img_pc, batch_pc, num_classes=2, task_sampler_nc=2, train_y=False, 
                 channel=3, im_size=(32, 32), inner_optim='SGD', syn_intervention=None, real_intervention=None, cctype=0):
        super(Distill, self).__init__()
        self.data = nn.Embedding(img_pc*num_classes, int(channel*np.prod(im_size)))
        self.train_y = train_y
        if train_y:
            self.label = nn.Embedding(img_pc*num_classes, num_classes)
            self.label.weight.data = y_init.float().cuda()
        else:
            self.label = y_init
        self.num_classes = num_classes
        self.channel = channel
        self.im_size = im_size
        self.net = get_arch(arch, self.num_classes, self.channel, self.im_size)
        self.img_pc = img_pc
        self.batch_pc = batch_pc
        self.arch = arch
        self.lr = lr
        self.window = window
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.num_train_eval = num_train_eval
        self.curriculum = window
        self.inner_optim = inner_optim
        self.batch_id = 0
        self.syn_intervention = syn_intervention
        self.real_intervention = real_intervention
        self.task_sampler_nc = task_sampler_nc
        self.cctype = cctype
        
    # shuffle the data 
    def shuffle(self):
        #True
        self.order_list = torch.randperm(self.img_pc)
        if self.img_pc >= self.batch_pc:
            self.order_list = torch.cat([self.order_list, self.order_list], dim=0)
    
    # randomly sample label sets from the full label set
    def get_task_indices(self):
        task_indices = list(range(self.num_classes))
        if self.task_sampler_nc < self.num_classes:
            random.shuffle(task_indices)
            task_indices = task_indices[:self.task_sampler_nc]
            task_indices.sort()
        return task_indices    
        
    def subsample(self):
        indices = []
        if self.task_sampler_nc == self.num_classes:
            for i in range(self.num_classes):
                ind = torch.randperm(self.img_pc)[:self.batch_pc].sort()[0] + self.img_pc * i
                indices.append(ind)
        else:
            task_indices = self.get_task_indices()
            for i in task_indices:
                ind = torch.randperm(self.img_pc)[:self.batch_pc].sort()[0] + self.img_pc * i
                indices.append(ind)
        indices = torch.cat(indices).cuda()
        imgs    = self.data(indices)
        imgs = imgs.view(
                   self.task_sampler_nc * min(self.img_pc, self.batch_pc),
                   self.channel,
                   self.im_size[0],
                   self.im_size[1]
               ).contiguous()
            
        if self.train_y:
            labels    = self.label(indices)
            labels = labels.view(
                       self.task_sampler_nc * min(self.img_pc, self.batch_pc),
                       self.num_classes
                   ).contiguous()
        else:
            labels = self.label[indices]
        
        return imgs, labels

    def forward(self, x):
        self.net = get_arch(self.arch, self.num_classes, self.channel, self.im_size).cuda()
        self.net.train()
            
        if self.inner_optim == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            # TODO: add decay rules for SGD
        elif self.inner_optim == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        if self.dd_type not in ['curriculum', 'standard']:
            print('The dataset distillation method is not implemented!')
            raise NotImplementedError()
        
        if self.dd_type == 'curriculum':
            for i in range(self.curriculum):
                self.optimizer.zero_grad()
                imgs, label = self.subsample()
                imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
                ratio = 0
                out, pres = self.net(imgs)
            
                loss = self.criterion(out, label)
                loss.backward()
                self.optimizer.step()
        with higher.innerloop_ctx(
                self.net, self.optimizer, copy_initial_weights=True
            ) as (fnet, diffopt):
            for i in range(self.window):
                imgs, label = self.subsample()
                imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
                ratio = 0
                out, pres = fnet(imgs)
            
                loss = self.criterion(out, label)
                diffopt.step(loss)
            x = self.real_intervention(x, dtype='real', seed=random.randint(0, 10000))
            return fnet(x)

    def init_train(self, epoch, init=False, lim=True):
        if init:
            self.net = get_arch(self.arch, self.num_classes, self.channel, self.im_size).cuda()
                    
            if self.inner_optim == 'SGD':
                self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            elif self.inner_optim == 'Adam':
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
            #self.shuffle()
        for i in range(epoch):
            self.optimizer.zero_grad()
            imgs, label = self.subsample()
            imgs = self.syn_intervention(imgs, dtype='syn', seed=random.randint(0, 10000))
            out, pres = self.net(imgs)
            loss = self.criterion(out, label)
            loss.backward()
            self.optimizer.step()
    
    # initialize the EMA with the currect data value
    def ema_init(self, ema_coef):
        self.shadow = -1e5
        self.ema_coef = ema_coef
    
    # update the EMA value
    def ema_update(self, grad_norm):
        if self.shadow == -1e5: 
            self.shadow = grad_norm
        else:
            self.shadow -= (1 - self.ema_coef) * (self.shadow - grad_norm)
        return self.shadow
    
    def test(self, x):
        with torch.no_grad():
            out = self.net(x)
        return out
        

def random_indices(y, nclass=10, intraclass=False, device='cuda'):
    n = len(y)
    if intraclass:
        index = torch.arange(n).to(device)
        for c in range(nclass):
            index_c = index[y == c]
            if len(index_c) > 0:
                randidx = torch.randperm(len(index_c))
                index[y == c] = index_c[randidx]
    else:
        index = torch.randperm(n).to(device)
    return index