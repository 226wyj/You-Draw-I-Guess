# -*- coding: utf-8 -*

import torch as t
from torch.autograd import Variable
from tqdm import tqdm

class Trainer():
    def __init__(self, net, criterion, optimizer, 
                    scheduler, train_loader, test_loader, model_path, args):
        
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_path = model_path
        self.args = args
        self.best_acc = 0.0
        self.device = t.device("cuda:1" if t.cuda.is_available() and not args.no_cuda else "cpu")
        self.net.to(self.device)

    # 在训练集上进行评估以此来确定是否保存模型
    def _evaluate(self):
        correct = 0     # 预测正确的图片数
        total = 0       # 总共的图片数
        self.net.eval() # 将net设置成eval模式
        print('Evaluating...')
        for data in tqdm(self.test_loader, desc="Eval Iteration", ncols=70):
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.net(Variable(images))
            _, predicted = t.max(outputs.data, 1)   # torch.max返回值为(values, indices)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        accuracy = 100. * correct / total
        return accuracy

    def save_model(self, epoch):
        accuracy = self._evaluate()
        if accuracy > self.best_acc:
            print('Accuracy: %f' % accuracy)
            print('Saving model...')
            state = {
                'net': self.net.state_dict(),
                'acc': accuracy,
                'epoch': epoch
            }
            t.save(state, self.model_path)
            self.best_acc = accuracy


    def train(self, epochs):
        self.net.train()    # 将net设置成训练模式
        for epoch in range(epochs):
            print("\n******** Epoch %d / %d ********\n" % (epoch + 1, epochs))
            running_loss = 0.0
            epoch_iterator = tqdm(self.train_loader, desc="Train Iteration", ncols=70)
            for i, data in enumerate(epoch_iterator):

                # 输入数据
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                inputs, labels = inputs.to(self.device), labels.to(self.device)


                # 梯度清零
                self.optimizer.zero_grad()

                # forward + backward
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels).to(self.device)
                loss.backward()

                # 更新参数
                self.optimizer.step()

                # 更新学习率
                self.scheduler.step()

                # 打印训练信息
                running_loss += loss.item()
                # if i % 2000 == 1999:
                #     print('[%d, %5d] loss: %3f' % (epoch + 1, i + 1, running_loss / 2000))
                #     running_loss = 0.0
            print('\nEpoch {} finish, loss: {}\n'.format(epoch + 1, running_loss / (i + 1)))
            self.save_model(epoch)
        print('\nFinish training\n')               
