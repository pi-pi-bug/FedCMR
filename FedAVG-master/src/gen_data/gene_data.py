import numpy as np
import os
import torchvision
import torchvision.datasets.mnist as mnist
import torchvision.transforms as transforms

from update import LocalUpdate, test_inference, DatasetSplit
from utils import average_weights, exp_details, get_dataset, DATASET_CONFIGS, gaussian_intiailize, \
    label_squeezing_collate_fn, fill_dataset
from sampling import mnist_iid


class DataGenerator:
    """
    Training and testing data generator

    Created by Hang Bai
    """
    def __init__(self, args):
        self.args = args
        #self.generate_data()

    def generate_data(self):
        capacity = self.args.batch_size * max(self.args.generator_iterations, self.args.solver_iterations)
        local_dataset_train, local_dataset_test = {}, {}  # user_id -> List[dataset1, ... dataset2] 客户端本地数据集

        if self.args.dataset == 'permutated-mnist':
            # generate permutations for the mnist classification tasks.
            np.random.seed(self.args.mnist_permutation_seed)
            permutations = [
                np.random.permutation(DATASET_CONFIGS['mnist']['size'] ** 2) for
                _ in range(self.args.mnist_permutation_number)
            ]

            train_datasets, test_datasets, users_dicts = [], [], []  # dict_users: user_id -> list[image_id...]
            for p in permutations:
                train_dataset, test_dataset, users_dict = get_dataset(self.args, 'permutated-mnist', permutation=p)
                train_datasets.append(train_dataset)
                test_datasets.append(test_dataset)
                users_dicts.append(users_dict)

            for user in range(self.args.num_users):
                user_train_datasets, user_test_datasets = [], []
                for i in range(self.args.mnist_permutation_number):
                    user_idxs = list(users_dicts[i][user])
                    train_idxs = user_idxs[:int(0.9 * len(user_idxs))]
                    test_idxs = user_idxs[int(0.9 * len(user_idxs)):]

                    local_data_train = DatasetSplit(train_datasets[i], train_idxs)
                    local_data_test = DatasetSplit(train_datasets[i], test_idxs)

                    user_train_datasets.append(local_data_train)
                    user_test_datasets.append(local_data_test)

                local_dataset_train[user] = user_train_datasets
                local_dataset_test[user] = user_test_datasets

            # decide what configuration to use.
            dataset_config = DATASET_CONFIGS['mnist']
            return local_dataset_train, local_dataset_test, dataset_config

        elif self.args.dataset == 'svhn-mnist':
            svhn_dataset_train, svhn_dataset_test, svhn_dict_users = get_dataset(self.args, 'svhn')  # 全局训练测试集
            mnist_dataset_train, mnist_dataset_test, mnist_dict_users = get_dataset(self.args, 'color-mnist')

            local_svhn_dataset_train, local_svhn_dataset_test = {}, {}
            local_mnist_dataset_train, local_mnist_dataset_test = {}, {}

            # decide what configuration to use.
            dataset_config = DATASET_CONFIGS['mnist-color']

            for user in range(self.args.num_users):
                # split local indexes for train and test (90, 10)
                svhn_idxs = list(svhn_dict_users[user])
                mnist_idxs = list(mnist_dict_users[user])

                svhn_idxs_train = svhn_idxs[:int(0.9 * len(svhn_idxs))]
                svhn_idxs_test = svhn_idxs[int(0.9 * len(svhn_idxs)):]

                mnist_idxs_train = mnist_idxs[:int(0.9 * len(mnist_idxs))]
                mnist_idxs_test = mnist_idxs[int(0.9 * len(mnist_idxs)):]

                local_svhn_dataset_train[user] = DatasetSplit(svhn_dataset_train, svhn_idxs_train)
                local_mnist_dataset_train[user] = DatasetSplit(mnist_dataset_train, mnist_idxs_train)
                # # fill the dataset 防止数据集不够
                # local_svhn_dataset_train[user] = fill_dataset(capacity, local_svhn_dataset_train[user])
                # local_mnist_dataset_train[user] = fill_dataset(capacity, local_mnist_dataset_train[user])
                local_dataset_train[user] = ([local_svhn_dataset_train[user], local_mnist_dataset_train[user]])  # 本地训练集

                local_svhn_dataset_test[user] = DatasetSplit(svhn_dataset_train,
                                                             svhn_idxs_test)  # svhn_dataset_test -> svhn_dataset_train
                local_mnist_dataset_test[user] = DatasetSplit(mnist_dataset_train, mnist_idxs_test)  # 同上
                # # fill the dataset 防止数据集不够
                # local_svhn_dataset_test[user] = fill_dataset(capacity, local_svhn_dataset_test[user])
                # local_mnist_dataset_test[user] = fill_dataset(capacity, local_mnist_dataset_test[user])
                local_dataset_test[user] = ([local_svhn_dataset_test[user], local_mnist_dataset_test[user]])  # 本地测试集
            return local_dataset_train, local_dataset_test, dataset_config

        elif self.args.dataset == 'five-mnist':
            # MNIST dataset
            data_dir = '../data/five-mnist/'
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Pad(2),
                transforms.ToTensor(),
            ])
            train_dataset = torchvision.datasets.MNIST(data_dir, train=True, transform=apply_transform,
                                                       download=True)
            test_dataset = torchvision.datasets.MNIST(data_dir, train=False, transform=apply_transform,
                                                      download=True)
            root = '../data/five-mnist//MNIST/raw/'
            train_set = (mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
                         mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
                         )
            test_set = (
                mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
                mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
            )

            train_idxs_01, test_idxs_01 = [], []
            train_idxs_23, test_idxs_23 = [], []
            train_idxs_45, test_idxs_45 = [], []
            train_idxs_67, test_idxs_67 = [], []
            train_idxs_89, test_idxs_89 = [], []

            for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
                if label.item() in (0, 1):
                    train_idxs_01.append(i)
                elif label.item() in (2, 3):
                    train_idxs_23.append(i)
                elif label.item() in (4, 5):
                    train_idxs_45.append(i)
                elif label.item() in (6, 7):
                    train_idxs_67.append(i)
                elif label.item() in (8, 9):
                    train_idxs_89.append(i)

            for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
                if label.item() in (0, 1):
                    test_idxs_01.append(i)
                elif label.item() in (2, 3):
                    test_idxs_23.append(i)
                elif label.item() in (4, 5):
                    test_idxs_45.append(i)
                elif label.item() in (6, 7):
                    test_idxs_67.append(i)
                elif label.item() in (8, 9):
                    test_idxs_89.append(i)

            train_dataset_01, test_dataset_01 = DatasetSplit(train_dataset, train_idxs_01), DatasetSplit(test_dataset,
                                                                                                         test_idxs_01)
            # iid sampling
            users_dict_01 = mnist_iid(train_dataset_01, self.args.num_users)
            train_dataset_23, test_dataset_23 = DatasetSplit(train_dataset, train_idxs_23), DatasetSplit(test_dataset,
                                                                                                         test_idxs_23)
            users_dict_23 = mnist_iid(train_dataset_23, self.args.num_users)
            train_dataset_45, test_dataset_45 = DatasetSplit(train_dataset, train_idxs_45), DatasetSplit(test_dataset,
                                                                                                         test_idxs_45)
            users_dict_45 = mnist_iid(train_dataset_45, self.args.num_users)
            train_dataset_67, test_dataset_67 = DatasetSplit(train_dataset, train_idxs_67), DatasetSplit(test_dataset,
                                                                                                         test_idxs_67)
            users_dict_67 = mnist_iid(train_dataset_67, self.args.num_users)
            train_dataset_89, test_dataset_89 = DatasetSplit(train_dataset, train_idxs_89), DatasetSplit(test_dataset,
                                                                                                         test_idxs_89)
            users_dict_89 = mnist_iid(train_dataset_89, self.args.num_users)

            train_datasets = [train_dataset_01, train_dataset_23, train_dataset_45, train_dataset_67, train_dataset_89]
            test_datasets = [test_dataset_01, test_dataset_23, test_dataset_45, test_dataset_67, test_dataset_89]
            users_dicts = [users_dict_01, users_dict_23, users_dict_45, users_dict_67,
                           users_dict_89]  # dict_users: user_id -> list[image_id...]

            for user in range(self.args.num_users):
                user_train_datasets, user_test_datasets = [], []
                for i in range(5):
                    user_idxs = list(users_dicts[i][user])
                    train_idxs = user_idxs[:int(0.9 * len(user_idxs))]
                    test_idxs = user_idxs[int(0.9 * len(user_idxs)):]

                    local_data_train = DatasetSplit(train_datasets[i], train_idxs)
                    local_data_train = fill_dataset(capacity, local_data_train)  # 填充数据集
                    local_data_test = DatasetSplit(train_datasets[i], test_idxs)
                    local_data_test = fill_dataset(capacity, local_data_test)  # 填充数据集
                    user_train_datasets.append(local_data_train)
                    user_test_datasets.append(local_data_test)

                local_dataset_train[user] = user_train_datasets
                local_dataset_test[user] = user_test_datasets

            # decide what configuration to use.
            dataset_config = DATASET_CONFIGS['mnist']
            return local_dataset_train, local_dataset_test, dataset_config

        else:
            raise RuntimeError('Given undefined experiment: {}'.format(self.args.dataset))
