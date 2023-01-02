import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

def ImgShow(img, target=''):
    print(f'image shape: {img.shape}')
    img = img / 2 + 0.5 #[-1, 1]->[0,1]
    np_img = img.numpy()
    img = np.transpose(np_img, (1,2,0))

    plt.cla()
    plt.text(0, 0, target)
    plt.imshow(img) #NOTE: [C,H,W]->[H, W, C]
    #  plt.show()
    plt.pause(1.5) #NOTE: unit: s

def PreprocessImage(view_imgs=False):
    #output[channel] = (input[channel] - mean[channel]) / std[channel]`
    transform = transforms.Compose([transforms.ToTensor(), #NOTE: PILImage to torch Tensor
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) #NOTE: 原始图片为3通道[0,1]的PILImage格式，norm后为3通道[-1,1]
        ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print(f'len train_dataset: {len(train_dataset)}, test_dataset: {len(test_dataset)}')

    if view_imgs:
        plt.figure(dpi=300, figsize=(0.5, 0.5))
        for idx, (img, target) in enumerate(train_dataset):
            print(f'target:{classes[target]}, value: {target}')
            if (idx % 100) == 0:
                ImgShow(img, classes[target])
        for idx, (img, target) in enumerate(test_dataset):
            print(f'target:{classes[target]}, value: {target}')
            if (idx % 100) == 0:
                ImgShow(img, classes[target])

    return (train_dataloader, test_dataloader, classes)
