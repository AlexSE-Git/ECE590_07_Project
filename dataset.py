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

def gen_NR_dataset(net, batch_size, EPS, ALP, ITS, log_every_n=50):
    print("Generating non-robust dataset...")
    # Set attack
    atk = help_attack.PGDL2(net, eps=EPS, alpha=ALP, steps=ITS)
    atk.set_mode_targeted(target_map_function=None)
    adv_vec = []
    lbl_vec = []
    global_steps = 0
    start = time.time()

    print('==> Preparing data..')
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)

        
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        targets = targets.to(device)

        # Random target label
        n_labels = torch.randint(10, (len(targets),))
        lbl_vec.append(n_labels)

        n_labels = n_labels.to(device)

        adv_input = atk(inputs, n_labels)
        adv_vec.append(adv_input)
        global_steps += 1

        if global_steps % log_every_n == 0:
                end = time.time()
                num_examples_per_second = log_every_n * batch_size / (end - start)
                print("[Step=%d]\t%.1f examples/second"
                      % (global_steps, num_examples_per_second))
                start = time.time()

    adv_vec = torch.stack(adv_vec)
    lbl_vec = torch.stack(lbl_vec) 
    np.save("./NR_dataset/adv.npy", adv_vec.cpu())
    np.save("./NR_dataset/lbl.npy", lbl_vec.cpu())
    print("Done!")

