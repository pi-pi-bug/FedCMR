#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,        #10
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,    #100
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.3,      #0.1
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # Generative replay arguments
    parser.add_argument('--mnist-permutation-number', type=int, default=5)
    parser.add_argument('--mnist-permutation-seed', type=int, default=0)
    parser.add_argument(
        '--replay-mode', type=str, default='generative-replay',
        choices=['exact-replay', 'generative-replay', 'none'],
    )
    parser.add_argument('--generator-lambda', type=float, default=10.)
    parser.add_argument('--generator-z-size', type=int, default=100)
    parser.add_argument('--generator-c-channel-size', type=int, default=64)
    parser.add_argument('--generator-g-channel-size', type=int, default=64)
    parser.add_argument('--solver-depth', type=int, default=5)
    parser.add_argument('--solver-reducing-layers', type=int, default=3)
    parser.add_argument('--solver-channel-size', type=int, default=1024)
    parser.add_argument('--generator-c-updates-per-g-update', type=int, default=5)
    parser.add_argument('--generator-iterations', type=int, default=2000)               #3000
    parser.add_argument('--solver-iterations', type=int, default=1000)                  #1000
    parser.add_argument('--importance-of-new-task', type=float, default=.3)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-05)
    parser.add_argument('--batch-size', type=int, default=32)               #32
    parser.add_argument('--test-size', type=int, default=1024)
    parser.add_argument('--sample-size', type=int, default=36)
    parser.add_argument('--sample-log', action='store_true')
    parser.add_argument('--sample-log-interval', type=int, default=300)
    parser.add_argument('--image-log-interval', type=int, default=100)
    parser.add_argument('--eval-log-interval', type=int, default=50)
    parser.add_argument('--loss-log-interval', type=int, default=30)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--sample-dir', type=str, default='./samples')
    parser.add_argument('--no-gpus', action='store_false', dest='cuda')

    main_command = parser.add_mutually_exclusive_group(required=True)
    main_command.add_argument('--train', action='store_true')
    main_command.add_argument('--test', action='store_false', dest='train')


    # other arguments
    parser.add_argument('--dataset', type=str, default='permutated-mnist',
                        choices=['permutated-mnist', 'svhn-mnist', 'five-mnist'],
                        help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=1, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args
