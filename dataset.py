# -*- coding: utf-8 -*

import torchvision as tv 
import torch as t 
import torchvision.transforms as transforms

class DataSet():
    '''
    封装待使用的数据集
    '''
    def __init__(self, train_loader, test_loader, classes):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.classes = classes

class DataBuilder():
    '''
    构造训练集、测试集
    '''
    def __init__(self, args) -> None:
        self.args = args
    
    # 训练集数据预处理
    def train_transform(self):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        return transform
    
    # 测试集数据预处理
    def test_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换成Tensor
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        return transform

    
    # 构造训练集
    def train_builder(self):
        train_set = tv.datasets.CIFAR10(
            root=self.args.data_path,
            train=True,
            download=self.args.is_download,
            transform=self.train_transform()
        )

        train_loader = t.utils.data.DataLoader(
            train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )
        return train_loader
    
    # 构造测试集
    def test_builder(self):
        test_set = tv.datasets.CIFAR10(
            root=self.args.data_path,
            train=False,
            download=self.args.is_download,
            transform=self.test_transform()
        )

        test_loader = t.utils.data.DataLoader(
            test_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers
        )
        return test_loader