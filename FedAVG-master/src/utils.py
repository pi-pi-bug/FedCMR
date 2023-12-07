#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import os.path
import torchvision
import math
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.dataloader import default_collate
from PIL import ImageOps

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import svhn_iid


def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    return image.view(c, h, w)


def _colorize_grayscale_image(image):
    return ImageOps.colorize(image, (0, 0, 0), (255, 255, 255))


def fill_dataset(capacity, dataset):
    if capacity is not None and len(dataset) < capacity:
        print('fill complete')
        return ConcatDataset([
            copy.deepcopy(dataset) for _ in
            range(math.ceil(capacity / len(dataset)))
        ])

    else:
        return dataset


def get_dataset(args, name, permutation=None):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if name == 'permutated-mnist':
        data_dir = '../data/permutated-mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
        ])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    elif name == 'svhn':
        data_dir = '../data/svhn/'

        train_dataset = datasets.SVHN(data_dir, split='train', download=True,
                                      transform=transforms.Compose(_SVHN_TRAIN_TRANSFORMS),
                                      target_transform=transforms.Compose(_SVHN_TARGET_TRANSFORMS)
                                      )

        test_dataset = datasets.SVHN(data_dir, split='test', download=True,
                                     transform=transforms.Compose(_SVHN_TEST_TRANSFORMS),
                                     target_transform=transforms.Compose(_SVHN_TARGET_TRANSFORMS)
                                     )
        # print('svhn complete')

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from svhn
            user_groups = svhn_iid(train_dataset, args.num_users)
        # else:
        #     # Sample Non-IID user data from svhn
        #     user_groups = svhn_noniid(train_dataset, args.num_users)

    elif name == 'color-mnist':
        data_dir = '../data/color-mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: _colorize_grayscale_image(x)),
            transforms.Pad(2),
            transforms.ToTensor(),
        ])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        # print('color-mnist complete')

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


_SVHN_TRAIN_TRANSFORMS = _SVHN_TEST_TRANSFORMS = [
    transforms.ToTensor(),
]
_SVHN_TARGET_TRANSFORMS = [
    transforms.Lambda(lambda y: y % 10)
]


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def label_squeezing_collate_fn(batch):
    x, y = default_collate(batch)
    return x, y.long().squeeze()


def get_data_loader(dataset, batch_size, cuda=False, collate_fn=None):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=(collate_fn or default_collate),
        **({'num_workers': 0, 'pin_memory': True} if cuda else {})
    )


def save_checkpoint(model, model_dir):
    path = os.path.join(model_dir, model.name)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save({'state': model.state_dict()}, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))


def load_checkpoint(model, model_dir):
    path = os.path.join(model_dir, model.name)

    # load the checkpoint.
    checkpoint = torch.load(path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=path
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])


def test_model(model, sample_size, path, verbose=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(
        model.sample(sample_size).data,
        path + '.jpg',
        nrow=6,
    )
    if verbose:
        print('=> generated sample images at "{}".'.format(path))


def validate(model, dataset, test_size=1024,
             cuda=False, verbose=True, collate_fn=None):
    data_loader = get_data_loader(
        dataset, 128, cuda=cuda,
        collate_fn=(collate_fn or default_collate),
    )
    total_tested = 0
    total_correct = 0
    for data, labels in data_loader:
        # break on test size.
        if total_tested >= test_size:
            break
        # test the model.
        data = Variable(data).cuda() if cuda else Variable(data)
        labels = Variable(labels).cuda() if cuda else Variable(labels)
        scores = model(data)
        _, predicted = torch.max(scores, 1)

        # update statistics.
        total_correct += (predicted == labels).sum().item()
        total_tested += len(data)

    precision = total_correct / total_tested
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision


def xavier_initialize(model):
    modules = [m for n, m in model.named_modules() if 'conv' in n or 'fc' in n]
    parameters = [p for m in modules for p in m.parameters()]

    for p in parameters:
        if p.dim() >= 2:
            nn.init.xavier_normal(p)
        else:
            nn.init.constant(p, 0)


def gaussian_intiailize(model, std=.01):
    modules = [m for n, m in model.named_modules() if 'conv' in n or 'fc' in n]
    parameters = [p for m in modules for p in m.parameters()]

    for p in parameters:
        if p.dim() >= 2:
            nn.init.normal(p, std=std)
        else:
            nn.init.constant(p, 0)


class LambdaModule(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'mnist-color': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 3, 'classes': 100},
    'svhn': {'size': 32, 'channels': 3, 'classes': 10},

}