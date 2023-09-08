import os
import copy
import sys
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
def rescale(x):
    #x = np.asarray(x)
    x = np.asarray(x)
    x = x.reshape(81, 1, 28, 28)
    #x = x.reshape(16, 1, 32, 32)
    return x

def show(x):
    figs, axes = plt.subplots(9, 9, figsize =(2*9, 2*9))
    for i in range(9):
        for j in range(9):
            axes[i][j].imshow(x[i*9+j][0])
    plt.show()

def remove_gen_sudoku_puzzle(Q, A, n_clue):
    res = np.copy(Q)
    pool = []
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            if (Q[i][j] != 0):
                pool.append((i,j))
    assert (len(pool) == 17)
    #print(n_clue)
    np.random.shuffle(pool)
    for (i, j) in pool[:17 - n_clue]:
        res[i][j] = 0
    return res


def gen_sudoku_puzzle(Q, A, n_clue):
    res = np.copy(Q)
    pool = []
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            if (Q[i][j] == 0):
                pool.append((i,j))
            else:
                n_clue -= 1
    #print(n_clue)
    np.random.shuffle(pool)
    for (i, j) in pool[:n_clue]:
        res[i][j] = A[i][j]
    return res

sss = str(sys.argv[3])
print (sss)

def convert(n, sudokus, digit_map, name, rate):
    ret = []
    ret_labels = []
    ret_Q = []
    n_iter = n // len(sudokus) + 1
    offset = 0
    for i in range(10):
        break
        interval = int(len(digit_map[i]) // 4)
        np.random.seed(123)
        idx = np.arange(len(digit_map[i]))
        np.random.shuffle(idx)
        digit_map[i] = np.array(digit_map[i])
        digit_map[i] = digit_map[i][idx]
        digit_map[i] = digit_map[i][interval * rate:min(interval * (rate + 1), len(digit_map[i]))]

    for i in range(n_iter):
        for sudoku in sudokus:
            Q = sudoku[0]
            A = sudoku[1]
            global sss
            if (int(sss) < 17):
                Q = remove_gen_sudoku_puzzle(Q, A, int(sss))
            else:
                Q = gen_sudoku_puzzle(Q, A, int(sss))
            flatten = 0#[number - offset for sublist in sudoku for number in sublist]
            ret_labels.append(A.reshape(-1))
            ret_Q.append(Q.reshape(-1))
            emnist_sudoku = []
            for number in list(Q.reshape(-1)):
                rnd = np.random.randint(len(digit_map[number]))
                if (number == 0):
                    emnist_sudoku.append(digit_map[number][rnd]) 
                else:
                    emnist_sudoku.append(digit_map[number][rnd])
            tmp = rescale(emnist_sudoku)
            #print(Q)
            #print(A)
            #show(tmp)
            ret.append(tmp)
    return ret, ret_labels, ret_Q


seed = int(sys.argv[2])
np.random.seed(seed)

n = int(sys.argv[1])
img_sz = 32

sudokus = np.load("minimum.npy")[-9000:]
#[:, 1, :, :]
print(sudokus.shape)
#print(sudokus[0][0])
#print(sudokus[0][1])

label2digits = {}
label2upper = {}
#label2digits = np.load("/atlaslocaldisk/shared/9by9_dichen/selected_digits.npy").item()
#label2upper = np.load("/atlaslocaldisk/shared/9by9_dichen/selected_uppers.npy").item()
#label2digits = np.load("selected_digits_offset2.npy", allow_pickle=True).item()
#label2upper = np.load("/atlaslocaldisk/shared/9by9_dichen/selected_uppers_offset2.npy", allow_pickle=True).item()
data_root='/tmp/public_dataset/pytorch'
test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=data_root, train=False, download=True,
                           transform=transforms.Compose([
                                transforms.ToTensor(),
                                #transforms.Normalize((0.1307,), (0.3081,))
                                transforms.Normalize((0.5,), (0.5,))
                            ])),
            batch_size=1, shuffle=True)
for i in range(10):
    label2digits[i] = []
for img, label in test_loader:
    img = img[0].cpu().data.numpy() 
    assert (img.shape == (1, 28, 28))
    label = label[0].item()
    label2digits[label].append(img)

name = "NAME"
suffix = "fortest%s" % (sss)
rtn, rtn_labels, rtn_Q = convert(n, sudokus, label2digits, name, 2)
rtn, rtn_labels, rtn_Q = rtn[:1000], rtn_labels[:1000], rtn_Q[:1000]
rtn, rtn_labels, rtn_Q = np.array(rtn), np.array(rtn_labels), np.asarray(rtn_Q)

print (rtn.shape)
print (rtn_labels.shape)
print (rtn_Q.shape)

s_rtn = []
s_rtn_labels = []
s_rtn_Q = []
idx = np.arange(len(rtn_Q))
np.random.shuffle(idx)
for i in idx:
    s_rtn.append(rtn[i])
    s_rtn_labels.append(rtn_labels[i])
    s_rtn_Q.append(rtn_Q[i])

np.save("YOUR PATH/9by9_%s_%s.npy" % (name, suffix), s_rtn)
np.save("YOUR PATH/9by9_%s_labels_%s.npy" % (name, suffix), s_rtn_labels)
np.save("YOUR PATH/9by9_%s_Q_%s.npy" % (name, suffix), s_rtn_Q)






