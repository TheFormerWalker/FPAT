# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 19:50:20 2022

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms





from models.resnet import ResNet18


import os
import torch.nn.functional as F

import numpy as np 

from torch.autograd import Variable

import csv
import torch.optim as optim


from pytorch_ares.attack_torch import PGD,DeepFool,CW,FGSM,MIM,Nattack
from autoattack import AutoAttack


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)
    



device = 'cuda'
 
normalize = NormalizeByChannelMeanStd(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))


file_name = 'FPAT-CIFAR100-10B.pth'

transform_test = transforms.Compose([
    transforms.ToTensor(),

])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),

])

train_dir = "./train"  # 训练集路径
train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
val_dir = "./test"
test_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform_test)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)


cudnn.benchmark = False


 
def test(epoch):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    MIM_adv_correct = 0
    PGD7_adv_correct = 0
    PGD100_adv_correct = 0
    CW_adv_correct = 0
    AA_adv_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):

            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = net(inputs)

            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()
 
            
            #生成pgd对抗样本   
            with torch.enable_grad():
               MIM_adv = MIM_adversary.forward(inputs, targets,None)
               PGD7_adv = PGD7_adversary.forward(inputs, targets,None)
               PGD100_adv = PGD100_adversary.forward(inputs, targets,None)
               CW_adv = CW_adversary.forward(inputs, targets,None)
               AA_adv = AA_adversary.run_standard_evaluation(inputs, targets,bs=len(targets))
               #adv = adversary.run_standard_evaluation(inputs, targets,bs=len(targets))

            MIM_adv_outputs = net(MIM_adv)
            _, predicted = MIM_adv_outputs.max(1)
            MIM_adv_correct += predicted.eq(targets).sum().item()
            
            PGD7_adv_outputs = net(PGD7_adv)
            _, predicted = PGD7_adv_outputs.max(1)
            PGD7_adv_correct += predicted.eq(targets).sum().item()
            
            PGD100_adv_outputs = net(PGD100_adv)
            _, predicted = PGD100_adv_outputs.max(1)
            PGD100_adv_correct += predicted.eq(targets).sum().item()
            
            CW_adv_outputs = net(CW_adv)
            _, predicted = CW_adv_outputs.max(1)
            CW_adv_correct += predicted.eq(targets).sum().item()
            
            AA_adv_outputs = net(AA_adv)
            _, predicted = AA_adv_outputs.max(1)
            AA_adv_correct += predicted.eq(targets).sum().item()

                
    print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
    print('Total MIM test Accuarcy:', 100. * MIM_adv_correct / total)
    print('Total PGD7 test Accuarcy:', 100. * PGD7_adv_correct / total)
    print('Total PGD100 test Accuarcy:', 100. * PGD100_adv_correct / total)
    print('Total CW test Accuarcy:', 100. * CW_adv_correct / total)
    print('Total AA test Accuarcy:', 100. * AA_adv_correct / total)




checkpoint = torch.load('./checkpoint/' + file_name)


net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
net.load_state_dict(checkpoint['net'])
net = torch.nn.Sequential(normalize, net)

net = net.to(device)


MIM_adversary=MIM(net, 0.0314, float('inf'), 0.00784, 7, 1, 'cifar10', None, "ce", "cuda")
PGD7_adversary=PGD(net,8/255,float('inf'), 2/255,7, "cifar10",False,"ce", "cuda")
PGD100_adversary=PGD(net,8/255,float('inf'), 2/255,100, "cifar10",False,"ce", "cuda")
CW_adversary = PGD(net,8/255,float('inf'), 2/255,7, "cifar10",False,"cw", "cuda")
AA_adversary=AutoAttack(net,steps=7.,query=0,eps=8/255,version='rand')

 
criterion = nn.CrossEntropyLoss()





for epoch in range(0, 1):
    
    test(epoch)

