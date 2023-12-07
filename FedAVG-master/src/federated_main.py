#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# 联邦中各个客户的数据分配（iid、 non-iid），还有服务器对客户端Solver权重的聚合（Generator权重只存储在本地）
# 1、Minist的10个类分为5组任务
# 2、SVHN-Minist
# 3、Minist 对像素进行5种不同的排列，形成5种任务
# GR 与 ER 与 none
# 8.17 进度：现在把svhn-mnist这个实验基本能跑通了，另一个划分mnist为五个任务的数据还是分的有点问题，体现不出来遗忘性，然后就是结果测试那块也没写完。
# 8.20 进度：现在实验两个都基本能跑通，但是准确率偏低，现在就在调参数，我怀疑是回合数太小，但是回合数设置大了程序又会报错，现在就在解决这个问题。还有mnist数据集改为划分5个类别的数据集。
# 8.23 进度：1、将服务器的8097端口映射到本地端口，实现在本地查看visdom。
#           2、每一全局回合后，对所有客户端的测试精度取平均（或者用最终的全局模型在全局测试集上测试精度）。
#           3、增加实验回合数。(补全数据集)
#           4、按类划分mnist为5个任务。
#           5、为什么fill补充数据集不执行？
# 10.22 进度 目前接着之前的进度，已经将三个实验的数据集处理好了，目前还在调参数，还有一些问题
#           1、现在写的框架对实验数据量少的情况下能运行，但是精度不高，大概在85左右，但是我参考别的论文填充数据集后运行会报错，现在正在解决这个问题。
# 10.26 进度： 现在三个实验基本都做完了，就剩下可视化了。
# 10.29 进度 现在还在改visdom这个框架的代码做可视化，这个里面的回调函数现在还在改写，对聚合后全局模型的测试结果进行表示（前后到达任务的性能测试）。
# 11.2  进度 这两天在写对聚合后最终模型的结果测试，之前是直接用原来的持续学习框架直接对每个客户端训练后的模型进行测试，现在要改写为对最终训练好的模型进行测试。
#           现在在写全局的测试集，还有改写原来的测试框架再可视化。
#           1、什么时候测试结果 ——> 对训练好后聚合的全局模型测试  2、在什么数据集上测试 ——> 全局测试集？
# 11.23 进度： 最近框架都写好了，各个模块实现了，
# 12.1 现在实验基本完成了，在我电脑上用一个小的数据集跑了一下实验，能跑通了，现在准备把另外两个数据集处理好之后就可以放在服务器上跑了。
# 12.3 这两天在处理另外两个数据集，同时还整理了一下参考文献，开始构思写相关工作这部分内容了，问是先写中文的，完了再改成英文的吗，还是直接写英文的。
# 目标完成时间：12月10日前做完实验，1月前写完论文初稿，1月中旬放假前改完！
# 12.17 进度：这两天在服务器上跑了一下实验，把第一个排序手写数字那个数据集都跑了一遍，现在能跑通了，结果出来，就是效果还有点不理想，新任务到达时旧任务的性能现在能从原始的20提高到60左右，
#           现在在跑另外两个数据集，同时也开始写引言部分了，意见：在相关工作部分对FedWeit介绍一下，自然的引出自己的工作，不能批评它，最后实验部分也要与FedWeit对比一下。
# 12.31 进度，现在把自己数据集调整好了，正在跑，应该这两天结果就能出来，同时引言部分我也写了一部分了。


import os
import copy
import time
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter

from gen_data.gene_data import DataGenerator
from dgr import Scholar
from options import args_parser
from models import CNN, WGAN
from utils import average_weights, exp_details, get_dataset, DATASET_CONFIGS, gaussian_intiailize, \
    label_squeezing_collate_fn, fill_dataset
from train import train

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    cuda = torch.cuda.is_available() and args.cuda

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    exp_details(args)

    if args.gpu:
        # torch.cuda.set_device(args.gpu)
        torch.cuda.set_device(0)
    device = 'cuda' if args.gpu else 'cpu'

    # generate data
    data_g = DataGenerator(args)
    # user_id -> List[dataset1, ... dataset2] 客户端本地train/test数据集
    local_dataset_train, local_dataset_test, dataset_config = data_g.generate_data()

    # define the models.
    cnn = CNN(
        image_size=dataset_config['size'],
        image_channel_size=dataset_config['channels'],
        classes=dataset_config['classes'],
        depth=args.solver_depth,
        channel_size=args.solver_channel_size,
        reducing_layers=args.solver_reducing_layers,
    )
    wgan = WGAN(
        z_size=args.generator_z_size,
        image_size=dataset_config['size'],
        image_channel_size=dataset_config['channels'],
        c_channel_size=args.generator_c_channel_size,
        g_channel_size=args.generator_g_channel_size,
    )
    label = '{experiment}-{replay_mode}-r{importance_of_new_task}'.format(
        experiment=args.dataset,
        replay_mode=args.replay_mode,
        importance_of_new_task=(
            1 if args.replay_mode == 'none' else
            args.importance_of_new_task
        ),
    )

    # use cuda if needed
    # device_ids = [0, 1]
    scholar = Scholar(label, generator=wgan, solver=cnn)
    if cuda:
        scholar.cuda()
        # scholar = torch.nn.DataParallel(scholar, device_ids=device_ids)

    # initialize the model.
    gaussian_intiailize(scholar, std=.02)

    # determine whether we need to train the generator or not.
    train_generator = (
            args.replay_mode == 'generative-replay' or
            args.sample_log
    )

    global_model = scholar.solver

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            print(f'\n | Local Training Users : {idx} |\n')
            w = train(  # return
                scholar, local_dataset_train[idx], local_dataset_test[idx],
                replay_mode=args.replay_mode,
                generator_lambda=args.generator_lambda,
                generator_iterations=(
                    args.generator_iterations if train_generator else 0
                ),
                generator_c_updates_per_g_update=(
                    args.generator_c_updates_per_g_update
                ),
                solver_iterations=args.solver_iterations,
                importance_of_new_task=args.importance_of_new_task,
                batch_size=args.batch_size,
                test_size=args.test_size,
                sample_size=args.sample_size,
                lr=args.lr, weight_decay=args.weight_decay,
                beta1=args.beta1, beta2=args.beta2,
                loss_log_interval=args.loss_log_interval,
                eval_log_interval=args.eval_log_interval,
                image_log_interval=args.image_log_interval,
                sample_log_interval=args.sample_log_interval,
                sample_log=args.sample_log,
                sample_dir=args.sample_dir,
                checkpoint_dir=args.checkpoint_dir,
                collate_fn=label_squeezing_collate_fn,
                cuda=cuda
            )

            local_weights.append(copy.deepcopy(w))
            # local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

    # print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
