# -*- coding: utf-8 -*

from predict import Predictor
from train import Trainer
from test import Tester 
from model import LeNet, Vgg16_Net
from dataset import DataSet, DataBuilder 
from util import check_path, show_model

import torch as t 
import torch.nn as nn 
from torch import optim
import argparse
import os
from PIL import Image


def main(args):

    check_path(args)

    # CIFAR-10的全部类别，一共10类
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 数据集
    data_builder = DataBuilder(args)
    dataSet = DataSet(data_builder.train_builder(), data_builder.test_builder(), classes)
    
    # 选择模型
    if args.lenet:
        net = LeNet()
        model_name = args.name_le
    elif args.vgg:
        net = Vgg16_Net()
        model_name = args.name_vgg
    elif args.resnet:
        pass
    
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    
    # SGD优化器
    optimizer = optim.SGD(
        net.parameters(), 
        lr=args.learning_rate, 
        momentum=args.sgd_momentum, 
        weight_decay=args.weight_decay
    )

    # 余弦退火调整学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    
    # 模型的参数保存路径，默认为 "./model/state_dict"
    model_path = os.path.join(args.model_path, model_name)

    # 启动训练
    if args.do_train:
        print("Training...")
        trainer = Trainer(net, criterion, optimizer, scheduler, dataSet.train_loader, args)
        trainer.train(epochs=args.epoch)
        t.save(net.state_dict(), model_path)
    
    # 启动测试
    if args.do_eval:
        if not os.path.exists(model_path):
            print("Sorry, there's no saved model yet, you need to train first.")
            return
        print("Testing...")
        net.load_state_dict(t.load(model_path))
        # net.eval()
        tester = Tester(dataSet.test_loader, net, args)
        tester.test()
    
    if args.show_model:
        if not os.path.exists(model_path):
            print("Sorry, there's no saved model yet, you need to train first.")
            return
        show_model(args)
    
    if args.do_predict:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        net.load_state_dict(t.load(model_path, map_location=device))
        predictor = Predictor(net, classes)
        # img_path = 'test'
        # img_name = [os.path.join(img_path, x) for x in os.listdir(img_path)]
        # for img in img_name:
        #     predictor.predict(img)
        img_path = 'test/cat0.jpg'
        predictor.predict(img_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
   
    # 数据集
    parser.add_argument("--num_workers", default=0, type=int, help="Thread number for training.")
    parser.add_argument("--is_download", default=True, type=bool, help="Download the datasets if there is no data.")

    # 根目录
    parser.add_argument("--data_path", default="data", type=str, help="The directory of the CIFAR-10 data.")
    parser.add_argument("--model_path", default="model", type=str, help="The directory of the saved model.")
    
    # 模型参数文件名字
    parser.add_argument("--name_le", default="state_dict_le", type=str, help="The name of the saved model's parameters.")
    parser.add_argument("--name_vgg", default="state_dict_vgg", type=str, help="The name of the saved model's parameters.")
    parser.add_argument("--name_res", default="state_dict_res", type=str, help="The name of the saved model's parameters.")

    # 训练相关
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--epoch", default=10, type=int, help="The number of training epochs.")
    parser.add_argument("--seed", default=42, type=int, help="The random seed used for initialization.")
    
    # 超参数
    parser.add_argument("--learning_rate", default=0.01, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=5e-4, type=int, help="Weight decay of SGD optimzer.")
    parser.add_argument("--sgd_momentum", default=0.9, type=float, help="The momentum of the SGD optimizer.")
    
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="The Epsilon of Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # 命令
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_predict", action="store_true", help="Predict single image with the saved model.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available.")
    parser.add_argument("--show_model", action="store_true", help="Display the state dict of the model.")
    parser.add_argument("--lenet", action="store_true", help="Use LeNet-5 as the model.")
    parser.add_argument("--vgg", action="store_true", help="Use VGG-16 as the model.")
    parser.add_argument("--resnet", action="store_true", help="Use ResNet as the model.")


    args = parser.parse_args()
    main(args)