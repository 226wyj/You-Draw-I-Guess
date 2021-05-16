# -*- coding: utf-8 -*

from predict import Predictor
from train import Trainer
from test import Tester 
from models.lenet import LeNet
from models.vgg16 import Vgg16_Net
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from dataset import DataSet, DataBuilder 
from util import check_path, show_model

import torch as t 
import torch.nn as nn 
from torch import optim
import argparse
import os


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
    elif args.resnet18:
        net = ResNet18()
        model_name = args.name_res18
    elif args.resnet34:
        net = ResNet34()
        model_name = args.name_res34
    elif args.resnet50:
        net = ResNet50()
        model_name = args.name_res50  
    elif args.resnet101:
        net = ResNet101()
        model_name = args.name_res101
    elif args.resnet152:
        net = ResNet152()
        model_name = args.name_res152

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

    
    # 模型的参数保存路径
    model_path = os.path.join(args.model_path, model_name)


    # 启动训练
    if args.do_train:
        print("Training...")

        trainer = Trainer(net, criterion, optimizer, scheduler, 
            dataSet.train_loader, dataSet.test_loader, model_path, args)

        trainer.train(epochs=args.epoch)
        # t.save(net.state_dict(), model_path)
    
    # 启动测试，如果--do_train也出现，则用刚刚训练的模型进行测试
    # 否则就使用已保存的模型进行测试
    if args.do_eval:
        if not args.do_train and not os.path.exists(model_path):
            print("Sorry, there's no saved model yet, you need to train first.")
            return
        # --do_eval
        if not args.do_train:
            checkpoint = t.load(model_path)
            net.load_state_dict(checkpoint['net'])
            accuracy = checkpoint['acc']
            epoch = checkpoint['epoch']
            print("Using saved model, accuracy : %f  epoch: %d" % (accuracy, epoch))
        tester = Tester(dataSet.test_loader, net, args)
        tester.test()

    if args.show_model:
        if not os.path.exists(model_path):
            print("Sorry, there's no saved model yet, you need to train first.")
            return
        show_model(args)
    
    if args.do_predict:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        checkpoint = t.load(model_path, map_location=device)
        net.load_state_dict(checkpoint['net'])
        predictor = Predictor(net, classes)
        img_path = 'test'
        img_name = [os.path.join(img_path, x) for x in os.listdir(img_path)]
        for img in img_name:
            predictor.predict(img)
        # img_path = 'test/cat0.jpg'
        # predictor.predict(img_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
   
    # Data
    parser.add_argument("--num_workers", default=0, type=int, help="Thread number for training.")
    parser.add_argument("--is_download", default=True, type=bool, help="Download the datasets if there is no data.")

    # Dir
    parser.add_argument("--data_path", default="data", type=str, help="The directory of the CIFAR-10 data.")
    parser.add_argument("--model_path", default="saved", type=str, help="The directory of the saved model.")
    
    # File Name
    parser.add_argument("--name_le", default="lenet.pth", type=str, help="The name of the saved model's parameters.")
    parser.add_argument("--name_vgg", default="vgg16.pth", type=str, help="The name of the saved model's parameters.")
    parser.add_argument("--name_res18", default="resnet18.pth", type=str, help="The name of the saved model's parameters.")
    parser.add_argument("--name_res34", default="resnet34.pth", type=str, help="The name of the saved model's parameters.")
    parser.add_argument("--name_res50", default="resnet50.pth", type=str, help="The name of the saved model's parameters.")
    parser.add_argument("--name_res101", default="resnet101.pth", type=str, help="The name of the saved model's parameters.")
    parser.add_argument("--name_res152", default="resnet152.pth", type=str, help="The name of the saved model's parameters.")

    # Train
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--epoch", default=10, type=int, help="The number of training epochs.")
    parser.add_argument("--seed", default=42, type=int, help="The random seed used for initialization.")
    
    # Hyper Parameters
    parser.add_argument("--learning_rate", default=0.01, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=5e-4, type=int, help="Weight decay of SGD optimzer.")
    parser.add_argument("--sgd_momentum", default=0.9, type=float, help="The momentum of the SGD optimizer.")
    
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="The Epsilon of Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # Commands
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_predict", action="store_true", help="Predict single image with the saved model.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available.")
    parser.add_argument("--show_model", action="store_true", help="Display the state dict of the model.")
    parser.add_argument("--lenet", action="store_true", help="Use LeNet-5 as the model.")
    parser.add_argument("--vgg", action="store_true", help="Use VGG-16 as the model.")
    parser.add_argument("--resnet18", action="store_true", help="Use ResNet as the model.")
    parser.add_argument("--resnet34", action="store_true", help="Use ResNet as the model.")
    parser.add_argument("--resnet50", action="store_true", help="Use ResNet as the model.")
    parser.add_argument("--resnet101", action="store_true", help="Use ResNet as the model.")
    parser.add_argument("--resnet152", action="store_true", help="Use ResNet as the model.")


    args = parser.parse_args()
    main(args)