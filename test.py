import time
import random
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torchvision
import torchvision.utils
import torch.nn as nn
import torch.optim as optim

import numpy as np

import help_attack

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def draw_one(lbl, pic):
    npimg = pic.numpy()
    fig = plt.figure(figsize = (1, 3))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(lbl)
    plt.show()

def visualize(net):
    ind_arr = [7,8,28,4,26,1,3,5,2,0]
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Load the NR-dataset
    adv_vec = torch.from_numpy(np.load("./NR_dataset/adv.npy"))
    lbl_vec = torch.from_numpy(np.load("./NR_dataset/lbl.npy"))
    net.eval()
    with torch.no_grad():
        for i in ind_arr:
            curr_label = lbl_vec[0][i]
            curr_pic = adv_vec[0][i]
            draw_one(classes[curr_label], torchvision.utils.make_grid(curr_pic, normalize=True))
            curr_pic = curr_pic.to(device)
            outputs = net(curr_pic.unsqueeze(0))
            _, predicted = outputs.max(1)
            print('  ', classes[predicted])
        

def std_test(net):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(testloader)
    val_acc = correct / total
    print("\nValidation:\nTest Loss=%.4f, Test accuracy=%.4f\n" % (test_loss / (num_val_steps), val_acc))
    
def robust_test(net, EPS, ITS):
    
    # Set attack
    atk = help_attack.PGDL2(net, eps=EPS, alpha=20*(EPS/ITS), steps=ITS)
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        adv_input = atk(inputs, targets)
        adv_input = adv_input.to(device)
        with torch.no_grad():
            outputs = net(adv_input)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(testloader)
    val_acc = correct / total
    print("\nRobustness:\nTest Loss=%.4f, Robust Test accuracy=%.4f\n" % (test_loss / (num_val_steps), val_acc))
