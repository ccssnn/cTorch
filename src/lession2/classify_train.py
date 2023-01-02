import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_cifar10 import *

import torch.optim as optim

from torch.profiler import profile, record_function, ProfilerActivity

#定义一个简单的网络类
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #定义第一层卷积神经网络，输入通道维度为3,输出通道为6，卷积核大小5x5(注：两个维度一致，仅需写一个即可)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,kernel_size=5)

        #定义三层全连接网络
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # 在（2,2）的池化窗口下执行最大池化操作
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) #If the size is a square you can only specify a single number
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def Train(net, train_dataloader, eporch, save_per_eporch=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9) #NOTE: update_step = (1 / (1 - momentum)) * lr * gradient
    net.train()

    print_per_batch = 2000
    total_iter_num = eporch * len(train_dataloader)
    for idx in range(eporch):
        running_loss = 0.0
        for batch_index, (imgs, targets) in enumerate(train_dataloader): #NOTE: max(batch_index + 1) == len(train_dataloader)
            iter_num = idx * len(train_dataloader) + batch_index

            outs = net(imgs)
            loss = criterion(outs, targets)

            optimizer.zero_grad() #NOTE: 清空Tensor的gradient内存
            loss.backward()
            optimizer.step() #NOTE: 更新所有待训练的Tensor

            running_loss += loss.item()
            if (batch_index + 1) % print_per_batch == 0: #NOTE: 打印训练信息
                print(f'eporch: {idx}/{eporch}, iteration: {iter_num}/{total_iter_num}, loss: {running_loss/print_per_batch}')
                running_loss = 0
        if (idx + 1) % save_per_eporch == 0:
            model_path = 'checkpoints/classify_%d.pth' % idx
            torch.save(net.state_dict(), model_path)
            print(f'save checkpoints: {model_path} done')
    print('Train Finish!!!!!!!!!!!!!!!!!!')

def Test(test_dataloader, classes, model_path):
    print(f'load pth model from {model_path}')
    net = Net()
    net.load_state_dict(torch.load(model_path))

    diff = 0
    num = 0

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            for batch_index, (imgs, labels) in enumerate(test_dataloader):
                outs = net(imgs)
                _, preds = torch.max(outs, 1) #NOTE: argmax with channel direction
                #  print(f'outs= {outs}')
                #  print(f'preds = {preds}')
                for j in range(len(preds)):
                    #  print(f'[{batch_index, j}]: gt = {classes[labels[j]]} vs ex = {classes[preds[j]]}')
                    if (labels[j] != preds[j]):
                        diff += 1
                num += len(preds)
    #  print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="cpu_time_total"))

    print(f'top1: diff = {diff}/{num} = {diff/num}')

if __name__ == '__main__':
    print('Hello lession2: classify train')
    train_dataloader, test_dataloader, classes = PreprocessImage()

    net = Net()
    print(f'{net}')
    model_path='checkpoints/classify_29.pth'
    net.load_state_dict(torch.load(model_path)) #NOTE: fine tune
    Train(net, train_dataloader, eporch=30, save_per_eporch=5)
    Test(test_dataloader, classes, model_path)
