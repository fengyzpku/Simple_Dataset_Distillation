import os
import logging
from enum import Enum

import torch
import torch.distributed as dist

import numpy as np

import torch.nn.functional as F


# directly initailize the tensors with Gaussian distribution
def init_gaussian(num_classes, ipc_per_class, tensor_length):
    # num_classes: number of classes
    # ipc_per_class: number of tensors per class
    # tensor_length: length of each tensor
    # mean: mean of the Gaussian distribution
    # variance: variance of the Gaussian distribution

    # initialize the tensors
    tensors = torch.zeros(num_classes * ipc_per_class, tensor_length)

    # initialize the class means with the standard Gaussian distribution
    # the variance is just identity matrix
    class_means = torch.normal(torch.zeros(num_classes, tensor_length), torch.ones(tensor_length))
    
    # calculate the minimum distance between the class means
    min_dist = float('inf')
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            dist = torch.dist(class_means[i], class_means[j])
            if dist < min_dist:
                min_dist = dist
    # calculate the variance
    variance = min_dist / 4

    # initialize the tensors
    for i in range(num_classes):
        for j in range(ipc_per_class):
            tensors[i * ipc_per_class + j] = torch.normal(class_means[i], variance * torch.ones(tensor_length))

    return tensors

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.max(1)
        correct = pred.eq(target).sum().item()

    return correct * 100.0 / batch_size

def accuracy_ind(output, target, topk=(1,)):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.max(1)
        correct = pred.eq(target)

    return correct
    


# class for intervening on data
class ImageIntervention(object):
    def __init__(self, name, strategy, phase, not_single=False):
        self.name = name
        self.phase = phase
        self.not_single = not_single
        self.flip = False
        self.color = False
        self.cutout = False
        if self.name in ['syn_aug', 'real_aug', 'pair_aug']:
            self.functions = {
                'scale': self.diff_scale,
                'flip': self.diff_flip,
                'rotate': self.diff_rotate,
                'crop': self.diff_crop,
                'color': [self.diff_brightness, self.diff_saturation, self.diff_contrast],
                'cutout': self.diff_cutout,
                'mixup': self.mixup,
            }
            self.prob_flip = 0.5
            self.ratio_scale = 1.2
            self.ratio_rotate = 15.0
            self.ratio_crop_pad = 0.125
            self.ratio_cutout = 0.5 # the size would be 0.5x0.5
            self.ratio_noise = 0.05
            self.brightness = 1.0
            self.saturation = 2.0
            self.contrast = 0.5
            self.mixup_alpha = None
            
            self.keys = list(strategy.split('_'))
            for key in self.keys:
                if 'mixup' in key:
                    self.mixup_alpha = float(key[5:])
                if key == 'flip' and not_single == True:
                    self.flip = True
                    self.keys.remove(key)
                elif key == 'color' and not_single == True:
                    self.color = True
                    self.keys.remove(key)
                elif key == 'cutout' and not_single == True:
                    self.cutout = True
                    self.keys.remove(key)

            
            self.seed = -1
        elif self.name != 'none':
            raise NotImplementedError

    def __call__(self, x, dtype, seed):
        if self.name == 'none':
            return x

        elif self.name == 'syn_aug':
            if dtype == 'real':
                return x
            elif dtype == 'syn':
                return self.do(x, seed)
            else:
                raise NotImplementedError

        elif self.name == 'real_aug':
            if dtype == 'syn':
                return x
            elif dtype == 'real':
                return self.do(x, seed)
            else:
                raise NotImplementedError

        elif self.name == 'pair_aug':
            return self.do(x, seed)

    def do(self, x, seed):
        if not self.not_single:
        #print('Adding augmentation for {}'.format(self.name))
            self.set_seed(seed)
            intervention = self.keys[np.random.randint(0, len(self.keys), size=(1,))[0]]

            if intervention == 'color':
                self.set_seed(seed)
                function = self.functions['color'][np.random.randint(0, len(self.functions['color']), size=(1,))[0]]
            else:
                function = self.functions[intervention]

            self.set_seed(seed)
            x = function(x)
            self.reset_seed()
        else:
            self.set_seed(seed)
            if self.flip:
                x = self.diff_flip(x)
            if self.color:
                for f in self.functions['color']:
                    self.set_seed(seed)
                    # function = self.functions['color'][np.random.randint(0, len(self.functions['color']), size=(1,))[0]]
                    x = f(x)
            if len(self.keys) > 0:
                intervention = self.keys[np.random.randint(0, len(self.keys), size=(1,))[0]]
                function = self.functions[intervention]
                x = function(x)
                
            if self.cutout:
                self.set_seed(seed)
                x = self.diff_cutout(x)

            self.reset_seed()
            
        return x

    def reset_seed(self):
        self.seed = -1

    def update_seed(self):
        self.seed += 1
        #torch.random.manual_seed(self.seed)
        np.random.seed(self.seed)

    def set_seed(self, seed):
        self.seed = seed
        #torch.random.manual_seed(self.seed)
        np.random.seed(self.seed)

    def diff_scale(self, x):
        # x>1, max scale
        # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
        ratio = self.ratio_scale
        #sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
        sx = torch.Tensor(np.random.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio)
        self.update_seed()
        #sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
        sy = torch.Tensor(np.random.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio)
        theta = [[[sx[i], 0,  0],
                [0,  sy[i], 0],] for i in range(x.shape[0])]
        theta = torch.tensor(theta, dtype=torch.float)
        if self.phase == 'train' and self.name == 'pair_aug':
            theta[:] = theta[0]
        grid = F.affine_grid(theta, x.shape, align_corners=False).to(x.device)
        x = F.grid_sample(x, grid, align_corners=False)
        return x
       
    def mixup(self, x, y):
        alpha = self.mixup_alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
    
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


        lam = np.random.beta()
        randf = torch.Tensor(np.random.rand(x.size(0), 1, 1, 1)).to(x.device)
        if self.phase == 'train' and self.name == 'pair_aug':
            randf[:] = randf[0]
        return torch.where(randf < prob, x.flip(3), x)
       
    def diff_flip(self, x):
        prob = self.prob_flip
        #randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
        randf = torch.Tensor(np.random.rand(x.size(0), 1, 1, 1)).to(x.device)
        if self.phase == 'train' and self.name == 'pair_aug':
            randf[:] = randf[0]
        return torch.where(randf < prob, x.flip(3), x)

    def diff_rotate(self, x):
        ratio = self.ratio_rotate
        #theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
        theta = torch.Tensor(np.random.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
        theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
            [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
        theta = torch.tensor(theta, dtype=torch.float)
        if self.phase == 'train' and self.name == 'pair_aug':
            theta[:] = theta[0]
        grid = F.affine_grid(theta, x.shape, align_corners=False).to(x.device)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def diff_crop(self, x):
        # The image is padded on its surrounding and then cropped.
        ratio = self.ratio_crop_pad
        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        #translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
        translation_x = torch.Tensor(np.random.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1])).to(x.device).long()
        self.update_seed()
        #translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
        translation_y = torch.Tensor(np.random.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1])).to(x.device).long()
        if self.phase == 'train' and self.name == 'pair_aug':
            translation_x[:] = translation_x[0]
            translation_y[:] = translation_y[0]
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x

    def diff_brightness(self, x):
        ratio = self.brightness
        #randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        randb = torch.Tensor(np.random.rand(x.size(0), 1, 1, 1)).to(x.device)
        if self.phase == 'train' and self.name == 'pair_aug':
            randb[:] = randb[0]
        x = x + (randb - 0.5)*ratio
        return x

    def diff_saturation(self, x):
        ratio = self.saturation
        x_mean = x.mean(dim=1, keepdim=True)
        #rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        rands = torch.Tensor(np.random.rand(x.size(0), 1, 1, 1)).to(x.device)
        if self.phase == 'train' and self.name == 'pair_aug':
            rands[:] = rands[0]
        x = (x - x_mean) * (rands * ratio) + x_mean
        return x

    def diff_contrast(self, x):
        ratio = self.contrast
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        #randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
        randc = torch.Tensor(np.random.rand(x.size(0), 1, 1, 1)).to(x.device)
        if self.phase == 'train' and self.name == 'pair_aug':
            randc[:] = randc[0]
        x = (x - x_mean) * (randc + ratio) + x_mean
        return x

    def diff_cutout(self, x):
        ratio = self.ratio_cutout
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        #offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_x = torch.Tensor(
                np.random.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1])).to(x.device).long()
        self.update_seed()
        #offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.Tensor(
                np.random.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1])).to(x.device).long()
        if self.phase == 'train' and self.name == 'pair_aug':
            offset_x[:] = offset_x[0]
            offset_y[:] = offset_y[0]
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        return x

    
    '''
def save_proto_np(proto_state, step, image_dir=None, use_pmap=False):
    if use_pmap:
        proto_state = flax.jax_utils.unreplicate(proto_state)

    x_proto, y_proto = proto_state.params['x_proto'], proto_state.params['y_proto']

    path = os.path.join(image_dir, 'np')
    if not os.path.exists(path):
        os.makedirs(path)

    save_path = os.path.join(path, 'step{}'.format(str(step).zfill(6)))
    np.savez('{}.npz'.format(save_path), image=x_proto, label=y_proto)
    logging.info('Save prototype to numpy! Path: {}'.format(save_path))


def load_proto_np(path):
    npzfile = np.load('{}.npz'.format(path))
    return npzfile['image'], npzfile['label']


def save_frepo_image(proto_state, step, num_classes=10, class_names=None, rev_preprocess_op=None, image_dir=None,
                    use_pmap=False, is_grey=False, save_np=False, save_img=False):
    def scale_for_vis(img, rev_preprocess_op=None):
        if rev_preprocess_op:
            img = rev_preprocess_op(img)
        else:
            img = img / img.std() * 0.2 + 0.5
        img = np.clip(img, 0, 1)
        return img

    if use_pmap:
        proto_state = flax.jax_utils.unreplicate(proto_state)

    x_proto, y_proto = proto_state.apply_fn(variables={'params': proto_state.params})

    if save_np and image_dir:
        path = os.path.join(image_dir, 'np')
        if not os.path.exists(path):
            os.mkdir(path)
        save_path = os.path.join(path, 'step{}'.format(str(step).zfill(6)))
        logging.info('Save prototype to numpy! Path: {}'.format(save_path))
        np.savez('{}.npz'.format(save_path), image=x_proto, label=y_proto)

    x_proto = scale_for_vis(x_proto, rev_preprocess_op)

    total_images = y_proto.shape[0]
    total_index = list(range(total_images))
    total_img_per_class = total_images // num_classes
    img_per_class = 100 // num_classes

    if num_classes <= 100:
        select_idx = []
        # always select the top to make it consistent
        for i in range(num_classes):
            select = total_index[i * total_img_per_class: (i + 1) * total_img_per_class][:img_per_class]
            select_idx.extend(select)
    else:
        select_idx = []
        # always select the top to make it consistent
        for i in range(100):
            select = total_index[i * total_img_per_class: (i + 1) * total_img_per_class][0]
            select_idx.append(select)

    row, col = len(select_idx) // 10, 10
    fig = plt.figure(figsize=(33, 33))

    for i, idx in enumerate(select_idx[: row * col]):
        img = x_proto[idx]
        ax = plt.subplot(row, col, i + 1)
        if class_names is not None:
            ax.set_title('{}'.format(class_names[y_proto[idx].argmax(-1)], y_proto[idx].argmax(-1)), x=0.5, y=0.9,
                         backgroundcolor='silver')
        else:
            ax.set_title('class_{}'.format(y_proto[idx].argmax(-1)), x=0.5, y=0.9, backgroundcolor='silver')

        if is_grey:
            plt.imshow(np.squeeze(img), cmap='gray')
        else:
            plt.imshow(np.squeeze(img))

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        plt.imshow(np.squeeze(img))
        plt.xticks([])
        plt.yticks([])

    fig.patch.set_facecolor('black')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    if save_img and image_dir:
        path = os.path.join(image_dir, 'png')
        if not os.path.exists(path):
            os.mkdir(path)
        save_path = os.path.join(path, 'step{}'.format(str(step).zfill(6)))
        logging.info('Save prototype to numpy! Path: {}'.format(save_path))
        fig.savefig('{}.png'.format(save_path), bbox_inches='tight')

    return fig
    '''