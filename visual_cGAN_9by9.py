#!/usr/bin/env python
# coding: utf-8
import argparse
import os, sys 
import numpy as np
import math
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch

from resnet_visual_sudoku import *
from utils import Generator

np.random.seed(19950420)
torch.manual_seed(19950420)
torch.cuda.manual_seed_all(19941216)

batch_size = 64
latent_dim = 100
n_classes = 10
img_shape = (1, 32, 32)
img_size = 32

failll = 0
nums = 9
emb_dim = 512
#input_dim = emb_dim 
input_dim = 2 * emb_dim #+ nums ** 2

keep_p = 0.8

keep_ps = [0.8, 0.5, 0.3]
mix_ratios = [0, 0.5, 0.8, 0.3]
use_cells = [0, 1]
use_cell = 1
model_pools = ["visual-all-trained", "all-trained1", "trained-17.pt", "trained-27.pt"]
model_infos = []




# In[2]:
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    GREY = '\033[90m'

def save_sudoku_image(imgs, path):
    f, axes = plt.subplots(9, 9, figsize = (9 * 2, 9 * 2))
    for i in range(9):
        for j in range(9):
            axes[i][j].imshow(imgs[i*9+j], cmap = "gray")
            axes[i][j].axis('off')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    plt.close('all')

       
class predictor(nn.Module):
    def __init__(self):
        super(predictor, self).__init__()
        self.encoding_dim = emb_dim 

        self.emb = nn.Embedding(10, emb_dim)
        self.fc0 = torch.nn.Linear(emb_dim, nums)
        self.fc1 = torch.nn.Linear(nums + 1, emb_dim)

        #self.fc1 = torch.nn.Linear(nums **2 * (nums), 2048)
        #self.fc2 = torch.nn.Linear(2048, 2048)
        #self.fc3 = torch.nn.Linear(2048, 2048)
        #self.fc4 = torch.nn.Linear(2048,  nums **3)
        
        #self.lstm = torch.nn.LSTMCell(input_dim, emb_dim)
        self.lstm = torch.nn.LSTMCell(emb_dim, emb_dim)

        self.fc5 = torch.nn.Linear(input_dim, 512)
        self.fc6 = torch.nn.Linear(512, 512)
        self.fc7 = torch.nn.Linear(512, 512)
        self.fc8 = torch.nn.Linear(self.encoding_dim, nums)

        self.softmax = torch.nn.Softmax(dim = 1)
        self.softmax2 = torch.nn.Softmax(dim = 2)

        self.bn1 = torch.nn.BatchNorm1d(1024) 
        self.bn2 = torch.nn.BatchNorm1d(1024) 
        self.bn3 = torch.nn.BatchNorm1d(self.encoding_dim)

    def graph_layer(self, x, hx, cx, graph, loc_emb, nums, emb_dim, batch_size): 
        
        embs = x.view(batch_size, nums**2, emb_dim)
        #embs = torch.cat([embs, loc_emb], dim = 2)
        embs = embs.permute(0, 2, 1) # bs, emb_dim, 81

        graph_emb = torch.matmul(embs, graph)# bs, emb_dim, 81
        graph_emb = graph_emb.permute(0, 2, 1) # bs, 81, emb_dim
        graph_emb = graph_emb.contiguous().view(-1, emb_dim) #bs * 81, emb_dim
        #input_emb = torch.cat([ori_embs, graph_emb], dim = 1) #bs * 81, emb_dim * 2 
        input_emb = graph_emb
        
        #x = F.relu(self.fc5(input_emb))
        #x = self.bn1(x)
        #x = F.relu(self.fc6(x)) 
        #x = self.bn2(x)
        #x = self.fc7(x) 
        #x = self.bn3(x)
        x = input_emb
        hx, cx = self.lstm(x, (hx, cx))

        #print(hx.shape, cx.shape)
        return hx, hx, cx

    def forward(self, ori_embs, soft_argmax, graph, loc_emb, nums, emb_dim, batch_size, mix_ratio):
        L = torch.gt(ori_embs, 0).long().reshape(batch_size, nums ** 2)
        L[L == 0] = mix_ratio
        L[L == 1] = 1.0 - mix_ratio
        #tmp = torch.tensor(np.array([[xx for xx in range(10)] for _ in range(batch_size * (nums ** 2))])).long().cuda()
        #raw_embs = self.emb(tmp).permute(2, 0, 1) # emb_dim, bs * 81, 10
        #print (raw_embs.size())
        #x = torch.sum(raw_embs * soft_argmax, dim=2)
        x = self.emb(ori_embs.long()) # bs, 81, emb_dim
        
        hx = torch.zeros(batch_size * (nums ** 2) , emb_dim)
        cx = torch.zeros(batch_size * (nums ** 2) , emb_dim)
        if (use_cuda):
            hx = hx.cuda()
            cx = cx.cuda()
        for i in range(32):
            x, hx, cx = self.graph_layer(x, hx, cx, graph, loc_emb, nums, emb_dim, batch_size) #bs * 81, emb_dim

        raw_P = self.softmax(self.fc8(x)).reshape(batch_size, nums ** 2, nums) # bs, 81, 9
        Q = torch.clamp(ori_embs.long() - 1, 0, 8)
        one_hot_Q = nn.functional.one_hot(Q, num_classes=9).reshape(batch_size, nums ** 2, nums)
        #P1 = P1.permute(2, 0, 1)# 9, bs, 81
 
        #P = P1 * (1 - P0) + P2 * P0 
        P1 = raw_P.permute(2, 0, 1) * (1.0 - L)
        P2 = one_hot_Q.permute(2, 0, 1) * L
        P  = (P1 + P2).permute(1, 2, 0).reshape(batch_size * (nums ** 2), nums)

        return P

generator_digit = Generator()
if torch.cuda.is_available():
    print ("use cuda")
    generator_digit = generator_digit.cuda()
generator_digit.load_state_dict(torch.load("./models/G28-180.model"))
generator_digit.eval()


from skimage import data
from skimage.transform import resize



def entropy(x):
    return -torch.sum(x * torch.log(x + 1e-9), dim = 1)

def show_sudoku(x1, x2, x_mix, name="1"):
    n_col = 9 
    n_row = 9
    fig, axes = plt.subplots(n_row, n_col * 3, figsize = (3 *n_col, n_row))
    for j in range(n_row):
        for k in range(n_col):
            axes[j][k].imshow(x1[j][k], cmap = "gray")
            axes[j][9 + k].imshow(x2[j][k], cmap="gray")
            axes[j][18 + k].imshow(x_mix[j][k], cmap="gray")
    plt.show()

def gen_alldiff_constraints(nums, batch_size):
    
    sqr_nums = int(np.sqrt(nums))
    idx = np.arange(nums**2).reshape(nums, nums)
    all_diffs = []
    for bs in range(batch_size):
        all_diff = []
        for i in range(nums):
            all_diff.append(idx[:,i])

        for i in range(nums):
            all_diff.append(idx[i,:])

        for i in range(sqr_nums):
            for j in range(sqr_nums):
                all_diff.append(idx[i*sqr_nums:(i+1)*sqr_nums, j*sqr_nums:(j+1)*sqr_nums].reshape(-1))

        all_diff = np.asarray(all_diff, dtype="int32")
        all_diff += bs * (nums**2)
        all_diffs.append(all_diff)

    all_diffs = np.concatenate(all_diffs, axis = 0)
    return all_diffs

def gen_constraint_graph(nums):
    sqrt_nums = int(np.sqrt(nums))
    A = np.zeros((nums * nums, nums * nums))
    idx = np.arange(nums**2).reshape(nums, nums)
    for i in range(nums * nums):
        row = i // nums
        col = i % nums
        a = row // sqrt_nums
        b = col //sqrt_nums
        neighbors = list(idx[row, :]) + list(idx[:, col]) + list(idx[a*sqrt_nums:(a+1)*sqrt_nums, b*sqrt_nums:(b+1)*sqrt_nums].reshape(-1))
        neighbors = list(set(neighbors))
        for j in neighbors:
            if (j!=i):
                A[i][j] = 1

    D = np.eye(nums * nums) * (nums*2 - 2 + (sqrt_nums - 1)**2)
    L = D - A
    sqrt_inv_D = np.eye(nums * nums) * np.sqrt(1.0/(nums*2 - 2 + (sqrt_nums - 1)**2))
    graph = np.dot(np.dot(sqrt_inv_D, L), sqrt_inv_D)

    return graph

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, is_eval=True):
    model.load_state_dict(torch.load(path))
    if (is_eval):
        model.eval()

def save_data(x, name):
    print(name, x.shape)
    np.save(name, x)

parser = argparse.ArgumentParser()
parser.add_argument("--st1", default=1, type=int)
parser.add_argument("--ed1", default=4, type=int)
parser.add_argument("--st2", default=5, type=int)
parser.add_argument("--ed2", default=8, type=int)
parser.add_argument("--ori", default=0, type=int)
parser.add_argument("--lr", default=0.00001, type=float)
parser.add_argument("--ocase", default=0, type=int)
parser.add_argument("--load", default=0, type=int)
parser.add_argument("--save_name", default="Test", type=str)
parser.add_argument("--load_name", default="Test", type=str)
parser.add_argument("--save", default=0, type=int)
parser.add_argument("--save_unsolve", default=0, type=int)
parser.add_argument("--train_st", type=int)
parser.add_argument("--train_ed", type=int)
parser.add_argument("--load_from", default="20", type=str)
parser.add_argument("--sss", type=str)
parser.add_argument("--batch_size", default=40, type=int)
parser.add_argument("--clip_step", default=10000, type=int)
parser.add_argument("--epoch", default=1000, type=int)
parser.add_argument("--all_diff", default=1.0, type=float)
parser.add_argument("--entropy_cell", default=0.01, type=float)
parser.add_argument("--scale_recon", default=0.005, type=float)
parser.add_argument("--mode", default="train", type=str)
parser.add_argument("--model_idx", default="visual4", type=str)
parser.add_argument("--restart", default=0, type=int)
parser.add_argument("--aug", default=0, type=int)
parser.add_argument("--drnet", default=0, type=int)
parser.add_argument("--mix_ratio", default=0, type=float)
parser.add_argument("--random_model", default=1, type=int)
parser.add_argument("--random_dropout", default=1, type=int)
parser.add_argument("--random_mix", default=1, type=int)
parser.add_argument("--update_resnet", default=1, type=int)
parser.add_argument("--push_away", default=1, type=int)
parser.add_argument("--model_number", default=20, type=int)
parser.add_argument("--no_random", default=0, type=int)
parser.add_argument("--satnet", default=0, type=int)
parser.add_argument("--resnet_gap", default=1, type=int)
parser.add_argument("--no_restart", default=0, type=int)

args = parser.parse_args()

print(args)

pred_path = "./models/pred.model-%s" % args.model_idx
sep_path = "./models/sep.model-%s" % args.model_idx
ext_path = "./models/ext.model-%s" % args.model_idx 

sss = args.sss

ocase = args.ocase
print ("ocase: ", ocase)

if args.satnet:
    sudoku = np.load("YOUR_PATH_FOR_SUDOKU_")
    ori_label = np.load("[YOUR_PATH]")[-1000:]
    Qs = np.load("[YOUR_PATH]")[-1000:]
else:
# I introduce the meaning of this three variables here
# sudoku: visual sudoku instances
# ori_label: the label of the sudoku
# Qs: the normal sudoku (represented in string instead of image)
    if args.mode == "solve":
        sudoku = np.load("[YOUR_PATH]") #10000, 81, 1, 32, 32
        ori_label = np.load("[YOUR_PATH]") #10000, 81 
        Qs = np.load("[YOUR_PATH]")
    else:
        i = args.train_st
        sudoku = np.load("[YOUR_PATH]") #10000, 81, 1, 32, 32
        ori_label = np.load("[YOUR_PATH]") #10000, 81 
        Qs = np.load("[YOUR_PATH]")

    for i in range(args.train_st + 1, args.train_ed + 1):
        t_sudoku = np.load("[YOUR_PATH]") #10000, 81, 1, 32, 32
        t_label = np.load("[YOUR_PATH]") #10000, 81 
        t_Q = np.load("[YOUR_PATH]")

        sudoku = np.concatenate((sudoku, t_sudoku), axis=0)
        ori_label = np.concatenate((ori_label, t_label), axis=0)
        Qs = np.concatenate((Qs, t_Q), axis=0)

print (Qs.shape)

sudoku = sudoku[:args.clip_step]
ori_label = ori_label[:args.clip_step]
Qs = Qs[:args.clip_step]

print(sudoku.shape)

base = 1
best_sudoku_acc = 0.
best_epoch = 0
best_epoch2 = 0

n_data = sudoku.shape[0]

use_cuda = torch.cuda.is_available()

n_col = 9
n_row = 9

#lr = 0.00001
lr = args.lr

N_epoch = args.epoch
cur_model_num = 1
if args.no_random and not args.drnet:
    args.load_from = cur_model_num
    assert (10 * args.model_number <= args.epoch)

sep = resnet18(predictor=False)
ext = resnet18(predictor=True, emb_dim=emb_dim)
pred = predictor()
pretrain_resnet18 = resnet18(predictor=True, emb_dim=10, num_classes=10)
#pretrain_resnet18.load_state_dict(torch.load("/home/fs01/yb263/Deep-Reasoning-Network-Nature/Games/Visual_Sudoku/pytorch-playground/mnist/log/fullmnist/best-25.pth"))
#pretrain_resnet18.load_state_dict(torch.load("/home/fs01/yb263/drnet_restart/mixed_sudoku/pytorch-playground/mnist/log/fullmnist/best-25.pth"))
pretrain_resnet18.load_state_dict(torch.load("./best-25.pth"))

if args.load:
    assert (not args.drnet)
    print ("load from: ", args.load_name)
    load_model(pred, "./models/%s%s" % (args.load_name, args.load_from))
    load_model(sep, "./digits_se_best.pt")
    #load_model(ext, "./models/%s_ext%s" % (args.load_name, args.load_from))

if use_cuda:
    sep = sep.cuda()
    ext = ext.cuda()
    pred = pred.cuda()
    pretrain_resnet18 = pretrain_resnet18.cuda()
    print("use_cuda")
pretrain_resnet18.eval()
print ("n_data: ", n_data)
print ("base: ", base)

batch_size = args.batch_size
check_freq = 1 #int(args.clip_step // batch_size)

###all diff#####
alldiff_constraints = gen_alldiff_constraints(nums, batch_size) #bs * 27, 9
print("all_diff", alldiff_constraints.shape)


###constraint graph ####
graph = gen_constraint_graph(nums)
print("graph", graph.shape)
graph =  Variable(torch.tensor(graph).float(), requires_grad=False)
if (use_cuda):
    graph = graph.cuda()

###optional location embedding###
loc_emb = np.zeros((batch_size, nums**2, nums**2))
for i in range(batch_size):
    for j in range(nums**2):
        loc_emb[i][j][j]=1

loc_emb = Variable(torch.tensor(loc_emb).float(), requires_grad=False)
if (use_cuda):
    loc_emb = loc_emb.cuda()


if args.ori:
    pass
else:
    labels = []
    for i in range(nums ** 2 * batch_size):
        for j in range(10):
            labels.append(j)
    gen_labels = torch.LongTensor(labels)

if use_cuda:
    gen_labels =  gen_labels.cuda()
    
print("Training Starts")
best_acc = -1
entropy_factor = 1

# construct group map for further local searh
# group map (x, y) to the small block idx
group = {}
bases = [(0, 0), (3, 0), (6, 0), (0, 3), (0, 6), (3, 3), (3, 6), (6, 3), (6, 6)]
g = 0
for bx, by in bases:
    for i in range(3):
        for j in range(3):
            group[(i + bx, j + by)] = g
    g += 1
"""
for i in range(9):
    for j in range(9):
        print (group[(i, j)], end=" ")
    print ()
exit(-1)
"""


indices = np.arange(n_data, dtype = "int32")
if (args.restart):
    assert (args.train_st == args.train_ed)
    indices = np.load("unsolved_visual_%d.npy" % (args.train_st)).astype("int")
    indices = list(indices)
    for i in range(100 - len(indices)):
        indices.append(np.random.randint(n_data))
    print("Restart mode")
    print("current unsolved:", len(indices))
    batch_size = min(batch_size, len(indices))

obs = args.batch_size
imgsz = 28

st_time = time.time()
m_idx = int(args.load_from)
img_saved = False
for _epoch_ in range(N_epoch):
    all_sudoku_acc = 0
    all_label_acc = 0
    all_recon_loss = 0
    cnt = 0
    
    pred_probs = []
    gt_labels = []
    
    s_lst = []
    l_lst = []
    q_lst = []
    known_cell_lst = []
    
    np.random.shuffle(indices)

    batches = []
    batches_Q = []
    batches_L = []
    batches_known_cell = []
    cur_kc_acc = 0
    avg_acc = []
    tot = 0

    batch_size = obs
    alldiff_constraints = gen_alldiff_constraints(nums, batch_size)
    llabels = []
    for i in range(nums ** 2 * batch_size):
        for j in range(10):
            llabels.append(j)
    gen_labels = torch.LongTensor(llabels)
    if use_cuda:
        gen_labels = gen_labels.cuda()

    print ("current batch size: ", batch_size)
    print ("current learning cases: ", len(indices) + failll, " ", len(np.unique(indices)) + failll)
    for curnum, idx in enumerate(indices):
        s = sudoku[idx] #81, 1, 32, 32
        Q = Qs[idx]
        known_cell = (Q > 0).astype("int")

        l = ori_label[idx]
        
        s = s.reshape(nums, nums, imgsz, imgsz)
        s = np.reshape(s, (nums ** 2, 1, imgsz, imgsz))
        
        #loading samples
        s_lst.append(s)
        l_lst.append(l)
        q_lst.append(Q)
        known_cell_lst.append(known_cell)

        if (len(s_lst) == batch_size or curnum == len(indices) - 1):
            if len(s_lst) != batch_size:
                batch_size = len(s_lst)
                alldiff_constraints = gen_alldiff_constraints(nums, batch_size)
                llabels = []
                for i in range(nums ** 2 * batch_size):
                    for j in range(10):
                        llabels.append(j)
                gen_labels = torch.LongTensor(llabels)
                if use_cuda:
                    gen_labels = gen_labels.cuda()

            s = np.concatenate(s_lst, axis = 0) # bs * 81, 1, 32, 32
            s = Variable(torch.tensor(s).float(), requires_grad=False)
            cps = copy.deepcopy(s)
            tensor_kcl = Variable(torch.tensor(known_cell_lst).float(), requires_grad=False) #bs, 81
            #print (len(s_lst))
            #print (s.size())

            if use_cuda:
                s = s.cuda()
                cps = cps.cuda()
                tensor_kcl = tensor_kcl.cuda()
            
            epochs = 1
            for ii in range(epochs):
                tot += 1
                batches.append(s.cpu())
                batches_Q.append(q_lst)
                batches_L.append(l_lst)
                batches_known_cell.append(np.copy(known_cell_lst))

                #s: bs * 81, 1, 32, 32
                """
                ori_embs = ext(s, emb = True) # bs * 81, emb_dim
                #print("ori_embs", ori_embs.shape) 
                raw_P, labels_distribution = pred(ori_embs, graph, loc_emb, nums, emb_dim, batch_size) #bs * 81, 9 

                #print("raw_P", raw_P.shape) 
                #print("labels_distribution", labels_distribution.shape) 

                if (use_cuda):
                    z = sep(torch.tensor(s.reshape(-1, 1, 32, 32)).float()).cuda() #bs * 81, 1, 10, 100
                else:
                    z = sep(torch.tensor(s.reshape(-1, 1, 32, 32)).float())
                #print("z", z.shape)
                optimizer = torch.optim.Adam(list(pred.parameters()) + list(sep.parameters()) + list(ext.parameters()), lr=lr)
                """
                z = sep(s) #bs * 81, 100
                pretrain_predres = pretrain_resnet18(s, emb=True) # bs * 81, 10
                assert (pretrain_predres.size() == (batch_size * 81, 10))
                s = torch.argmax(pretrain_predres, axis=1) 
                soft_argmax = F.softmax(pretrain_predres, dim=1)
                snp = s.cpu().data.numpy().reshape(batch_size, 81)
                for kkk in range(batch_size):
                    ssss = snp[kkk]
                    avg_acc.append(1 if np.mean(np.equal(q_lst[kkk], snp[kkk]).astype("int")) == 1 else 0)
                    #if len(ssss[ssss > 0]) == args.train_st:
                        #avg_acc += 1
                labels_distribution = pred(s, soft_argmax, graph, loc_emb, nums, emb_dim, int(s.size(0) / 81), args.mix_ratio)
                if args.update_resnet:
                    #optimizer = torch.optim.Adam(list(pred.parameters()) + list(pretrain_resnet18.parameters()), lr=lr)
                    optimizer = torch.optim.Adam([{"params": pred.parameters()}, {"params": pretrain_resnet18.parameters(), "lr": 0.01 * lr}, {"params": sep.parameters(), "lr": 0.01 * lr}], lr=lr)
                else:
                    optimizer = torch.optim.Adam(list(pred.parameters()), lr=lr)
                
                # compute accs
                labels = labels_distribution.cpu().data.numpy()
                

                labels_argmax = np.argmax(labels, axis=1)

                l = np.concatenate(l_lst, axis = 0)
                gt_labels.append(l)

                
                eqn = np.equal(labels_argmax + base, l).astype("int").reshape(batch_size, nums**2)
                
                known_cell_acc = np.sum(eqn * np.asarray(known_cell_lst)) / np.sum(known_cell_lst)
                label_acc = np.mean(eqn)
                sudoku_acc = np.mean((np.sum(eqn, axis = (1)) == (nums ** 2)).astype("float32"))
                """
                # Compute the results after local search
                new_labels_argmax = copy.deepcopy(labels_argmax).reshape(batch_size, nums ** 2)
                for cb in range(batch_size):
                    cur_argmax = new_labels_argmax[cb]
                    cur_pl = labels.reshape(batch_size, nums ** 2, 9)[cb]
                    assert (cur_pl.shape == (nums ** 2, 9))
                    #local_search(cur_argmax, cur_pl, q_lst[cb], l_lst[cb])
                new_labels_argmax = new_labels_argmax.reshape(-1)
                new_eqn = np.equal(new_labels_argmax.reshape(-1) + base, l).astype("int").reshape(batch_size, nums ** 2) 
                new_known_cell_acc = np.sum(new_eqn * np.asarray(known_cell_lst)) / np.sum(known_cell_lst)
                new_label_acc = np.mean(new_eqn)
                new_sudoku_acc = np.mean((np.sum(new_eqn, axis = (1)) == (nums ** 2)).astype("float32"))
                """

                """
                # generative decoder

                gen_img = generator_digit(z.view(-1, latent_dim), gen_labels) #bs*81*10, 1, 32, 32
                gen_img = gen_img.view((nums ** 2) * batch_size, nums + 1, 1, 32, 32) #bs * 81, 10, 1, 32, 32

                #labels_distribution bs * 81, 9 
                gen_mix = gen_img.permute(2, 3, 4, 0, 1) * raw_P # 1, 32, 32, bs * 81, 10 

                gen_mix = torch.sum(gen_mix, dim = 4) # avg by distribution 1, 32, 32, bs*81

                gen_mix = gen_mix.permute(3, 0, 1, 2).view(-1, nums ** 2, 32, 32) #bs, 81, 32, 32
                
                ori_mix = s.view(-1, nums **2, 32, 32)
                #if (sudoku_acc == 1 and img_saved == False):
                """
                gen_img = generator_digit(z.view(-1, latent_dim), gen_labels)
                gen_img = gen_img.view(81 * batch_size, 10, 1, 28, 28)
                softmax = nn.Softmax()
                digits_p = softmax(pretrain_predres)
                gen_mix = gen_img.permute(2, 3, 4, 0, 1) * digits_p
                gen_mix = torch.sum(gen_mix, dim=4).permute(3, 0, 1, 2).view(81 * batch_size, 1, 28, 28)
                loss_recon = torch.mean(torch.abs(gen_mix - cps))
                """
                loss_recon = torch.abs(ori_mix - gen_mix) #bs, 81, 32, 32
                loss_recon = torch.mean(torch.sum(loss_recon, dim = (2, 3)))
                """
                
                """
                entropy_raw_P = entropy(raw_P)
                entropy_zero = torch.sum(entropy_raw_P * (1 - tensor_kcl).view(-1))/torch.sum(1-tensor_kcl)
                """

                entropy_cell = entropy(labels_distribution) #bs * 81
                
                all_diff_loss = entropy(torch.mean(labels_distribution[torch.LongTensor(alldiff_constraints)], dim = 1)) #bs * 27 
                
                entropy_alldiff = all_diff_loss

                scale_recon = args.scale_recon


                drop_out_recon = torch.nn.Dropout(p = 1.0 - 1.0)
                drop_out_cell = torch.nn.Dropout(p = 1.0 - 1.0)
                drop_out_alldiff = torch.nn.Dropout(p = 1.0 - keep_p)


                #loss_recon_drop = loss_recon
                entropy_cell_drop = torch.mean(drop_out_cell(entropy_cell))
                #entropy_raw_P_drop = torch.mean(drop_out_cell(entropy_raw_P))
                entropy_alldiff_drop = torch.mean(drop_out_alldiff(entropy_alldiff))
                
                if (_epoch_ < 0 and args.mode == "train"):
                    loss = args.scale_recon * loss_recon_drop
                else:
                    #if cur_kc_acc > 0.98:
                    #    loss = args.entropy_cell * entropy_factor * (entropy_cell_drop) \
                    #    - args.all_diff * (entropy_alldiff_drop)
                    #loss = args.scale_recon * loss_recon_drop +  args.entropy_cell * entropy_factor * (entropy_cell_drop + entropy_raw_P_drop) \
                    #    - args.all_diff * (entropy_alldiff_drop)
                    if use_cell:
                        loss = args.entropy_cell * entropy_factor * (entropy_cell_drop ) \
                            - args.all_diff * (entropy_alldiff_drop) + args.scale_recon * loss_recon
                    else:
                        loss = -args.all_diff * (entropy_alldiff_drop) + args.scale_recon * loss_recon
                
                if (args.mode != "test"):
                    optimizer.zero_grad()
                    loss.backward()
                    #for name, param in pred.named_parameters():
                    #    if param.requires_grad:
                    #        print (name, " ", param.grad)
                    #for name, param in pretrain_resnet18.named_parameters():
                    #    if param.requires_grad:
                    #        print (name, " ", param.grad)
                    #if _epoch_ > 5:
                    #    exit(-1)
                    optimizer.step()

                #if (ii % batch_size == 0 and (cnt + 1) % check_freq == 0):
                if ((cnt + 1) % 10 == 0): 
                    #print ("Initial Image")
                    #show_sudoku(s1_lst[0], s2_lst[0], np.reshape(s_mix_lst[0], (4, 4, 32, 32)), "s1")
                    print ("Iteration %d: total_loss: %f, entropy_cell: %f, entropy_alldiff: %f"\
                    % (cnt, loss.item(),  entropy_cell_drop.item(),  entropy_alldiff_drop.item()))
                    print ("epoch = %d, iter-%d, old_label_acc = %f, old_sudoku_acc= %f, old_known_cell_acc = %f" % \
                    (_epoch_, ii, label_acc, sudoku_acc, known_cell_acc))
                    #print ("epoch = %d, iter-%d, new_label_acc = %f, new_sudoku_acc= %f, new_known_cell_acc = %f" % \
                    #(_epoch_, ii, new_label_acc, new_sudoku_acc, new_known_cell_acc))
                    
                    #print(labels_argmax.shape)
                    #print(l.shape)
                    for i in range(nums):
                        for j in range(nums):
                            x = labels_argmax[i*nums + j] + base
                            gt_x = l[i*nums + j]
                            if (x == gt_x):
                                print(x, end = ",")
                            else:
                                if (known_cell_lst[0][i*nums+j]):
                                    print(color.PURPLE + "%d"%x + color.END, end = ",")
                                else:
                                    print(color.RED + "%d"%x + color.END, end = ",")
                        print(" ", end = "")
                        """
                        for j in range(nums):
                            x = new_labels_argmax[i*nums + j] + base
                            gt_x = l[i*nums + j]
                            if (x == gt_x):
                                print(x, end = ",")
                            else:
                                if (known_cell_lst[0][i*nums+j]):
                                    print(color.PURPLE + "%d"%x + color.END, end = ",")
                                else:
                                    print(color.RED + "%d"%x + color.END, end = ",")
                        print(" ", end = "")
                        """
                        for j in range(nums):
                            if (known_cell_lst[0][i*nums+j] == 1):
                                print(color.GREEN + "%d"%l[i*nums + j] + color.END, end = ",")
                            else:
                                print("%d"%l[i*nums + j], end = ",")
                        print("")
            
            s_lst = []
            l_lst = []
            q_lst = []
            known_cell_lst = []
                    
                 
            #all_recon_loss += loss_recon.item()
            all_recon_loss = 0
            all_sudoku_acc += sudoku_acc
            all_sudoku_acc += sudoku_acc
            cnt += 1
    print ("avg acc: ", np.sum(avg_acc), "total: ", len(indices))
        
    if (_epoch_ % check_freq == 0):
        print("validating performance")
        pred.eval()
        pretrain_resnet18.eval()
        all_sudoku_acc = []
        all_label_acc = []
        all_kc_acc = []
        new_all_sudoku_acc = []
        new_all_label_acc = []
        new_all_kc_acc = []

        unsolved = []
        try_to_solved = 0
        for i in range(len(batches)):
            s = batches[i]
            kc = batches_known_cell[i]
            q_lst = batches_Q[i]
            l_lst = batches_L[i]
            #tensor_kcl = Variable(torch.tensor(kc).float(), requires_grad=False) #bs, 81

            if use_cuda:
                s = s.cuda()
                #tensor_kcl = tensor_kcl.cuda()

            batch_size = int(s.size(0) / 81)
            try_to_solved += batch_size

            #s: bs, 81, 1, 32, 32
            #ori_embs = ext(s, emb = True) # bs * 81, emb_dim

            pretrain_predres = pretrain_resnet18(s, emb=True)
            s = torch.argmax(pretrain_predres, axis=1)
            soft_argmax = F.softmax(pretrain_predres, dim=1)
            labels_distribution = pred(s, soft_argmax, graph, loc_emb, nums, emb_dim, batch_size, args.mix_ratio) #bs * 81, 9 

            labels = labels_distribution.cpu().data.numpy()
             
            pred_probs.append(labels)

            l = gt_labels[i]
            labels_argmax = np.argmax(labels, axis=1)

            eqn = np.equal(labels_argmax + base, l).astype("int").reshape(batch_size, nums**2)
                
            label_acc = np.mean(eqn)
            sudoku_acc = (np.sum(eqn, axis = (1)) == (nums ** 2)).astype("float32")
            known_cell_acc = np.sum(eqn * np.asarray(kc)) / np.sum(kc)
            """
            new_labels_argmax = copy.deepcopy(labels_argmax).reshape(batch_size, nums ** 2)
            for cb in range(new_labels_argmax.shape[0]):
                cur_argmax = new_labels_argmax[cb]
                cur_pl = labels.reshape(batch_size, nums ** 2, 9)[cb]
                assert (cur_pl.shape == (nums ** 2, 9))
                local_search(cur_argmax, cur_pl, q_lst[cb], l_lst[cb])
            new_labels_argmax = new_labels_argmax.reshape(-1)
            new_eqn = np.equal(new_labels_argmax.reshape(-1) + base, l).astype("int").reshape(batch_size, nums ** 2) 
            new_known_cell_acc = np.sum(new_eqn * np.asarray(kc)) / np.sum(kc)
            new_label_acc = np.mean(new_eqn)
            new_sudoku_acc = np.mean((np.sum(new_eqn, axis = (1)) == (nums ** 2)).astype("float32"))
            """
            def _check_valid_sudoku(x):
                # each row
                for i in range(nums):
                    app = []
                    for j in range(nums):
                        if x[i][j] in app:
                            return 0
                        app.append(x[i][j])
                for i in range(nums):
                    app = []
                    for j in range(nums):
                        if x[j][i] in app:
                            return 0
                        app.append(x[j][i])
                for i in range(0, nums, 3):
                    for j in range(0, nums, 3):
                        app = []
                        for ii in range(i, i + 3):
                            for jj in range(j, j + 3):
                                if x[ii][jj] in app:
                                    return 0
                                app.append(x[ii][jj])
                return 1


            for j in range(len(sudoku_acc)):
                if (sudoku_acc[j] == 0):
                    if _check_valid_sudoku(labels_argmax.reshape(batch_size, nums, nums)[j]):
                        failll += 1
                    else:
                        unsolved.append(indices[try_to_solved - batch_size + j])
                    #print (i, " ", j, " ", batch_size, " ", i * batch_size + j)
            print ("failll: ", failll)
            
            sudoku_acc = np.mean(sudoku_acc)
            all_sudoku_acc.append(sudoku_acc)
            all_label_acc.append(label_acc)
            all_kc_acc.append(known_cell_acc)
            #new_all_sudoku_acc.append(new_sudoku_acc)
            #new_all_label_acc.append(new_label_acc)
            #new_all_kc_acc.append(new_known_cell_acc)
        

        
        all_sudoku_acc = np.mean(all_sudoku_acc)
        all_label_acc = np.mean(all_label_acc)
        all_kc_acc = np.mean(all_kc_acc)
        #new_all_sudoku_acc = np.mean(new_all_sudoku_acc)
        #new_all_label_acc = np.mean(new_all_label_acc)
        #new_all_kc_acc = np.mean(new_all_kc_acc)

        if args.mode == "solve":
            indices = np.array(unsolved)
            #batch_size = min(batch_size, len(unsolved))
            #alldiff_constraints = gen_alldiff_constraints(nums, batch_size)
            print ("current solve mode unsolved case: ", len(unsolved) + failll, " ", len(np.unique(unsolved)) + failll)

        if (args.mode == "solve" and args.save_unsolve): 
            np.save("unsolved_visual_%d" % (args.train_st), unsolved)
        print("#puzzle = %d, old_sudoku_acc = %f, old_label_acc = %f, old_known_cell_acc = %f recon_loss = %f (best_acc = %f)"%\
        (cnt * batch_size, all_sudoku_acc, all_label_acc, all_kc_acc, all_recon_loss/cnt, best_acc))
        """
        print("#puzzle = %d, new_sudoku_acc = %f, new_label_acc = %f, new_known_cell_acc = %f recon_loss = %f (best_acc = %f)"%\
        (cnt * batch_size, new_all_sudoku_acc, new_all_label_acc, new_all_kc_acc, all_recon_loss/cnt, best_acc))
        """
        cur_kc_acc = all_kc_acc

        if args.save:
            assert (args.mode == "train")
            if len(model_infos) < 20:
                model_infos.append(all_sudoku_acc)
                model_idx = len(model_infos)
                save_model(pred, "./models/%s_pred%d" % (args.save_name, model_idx))
                save_model(ext, "./models/%s_ext%d" % (args.save_name, model_idx))
                save_model(sep, "./models/%s_sep%d" % (args.save_name, model_idx))
            else:
                min_acc = np.min(np.array(model_infos))
                if min_acc <= all_sudoku_acc:
                    min_idx = np.argmin(np.array(model_infos))
                    model_infos[min_idx] = all_sudoku_acc
                    save_model(pred, "./models/%s_pred%d" % (args.save_name, min_idx + 1))
                    save_model(ext, "./models/%s_ext%d" % (args.save_name, min_idx + 1))
                    save_model(sep, "./models/%s_sep%d" % (args.save_name, min_idx + 1))
        """    
        if (args.mode != "solve" and (all_sudoku_acc > best_acc)):
            best_acc = all_sudoku_acc
            best_epoch = _epoch_
            if (best_acc > 0.97):
                entropy_factor = 1.0
        """
        if False and (best_acc > all_sudoku_acc):
        #if (try_to_solved > len(indices)):
            best_epoch = _epoch_
        elif args.drnet:
            pass
        else:
            if args.mode == "solve" and args.no_random and _epoch_ - best_epoch >= 10: 
                if not args.no_restart:
                    print ("move to the next model model num: %d" % (cur_model_num))
                cur_model_num += 1
                if _epoch_ > 1100 and (args.no_restart or cur_model_num > args.model_number):
                    exit(-1)
                if cur_model_num > args.model_number:
                    cur_model_num = 1
                m_idx = cur_model_num
                if not args.no_restart:
                    load_model(pred, "./models/%s%d" % (args.load_name, m_idx))
                use_cell = np.random.randint(2)
                keep_p = keep_ps[np.random.randint(len(keep_ps))]
                #args.mix_ratio = mix_ratios[np.random.randint(len(mix_ratios))]
                args.mix_ratio = 0
                best_epoch = _epoch_
            if args.mode == "solve" and not args.no_random and _epoch_ - best_epoch >= 10:
                print ("tuine for restart")
                use_cell = np.random.randint(2)
                if args.random_dropout:
                    keep_p = keep_ps[np.random.randint(len(keep_ps))]
                if args.random_model:
                    #m_idx = np.random.randint(args.model_number) + 1
                    m_idx = (m_idx + 1) % args.model_number + 1
                    load_model(pred, "./models/%s%d" % (args.load_name, m_idx))
                if args.random_mix:
                    args.mix_ratio = mix_ratios[np.random.randint(len(mix_ratios))]
                print ("now setting: keep_p: %f, use_cell; %d, model: %s" % (keep_p, use_cell, str(m_idx)))
                #pretrain_resnet18.load_state_dict(torch.load("/home/fs01/yb263/Deep-Reasoning-Network-Nature/Games/Visual_Sudoku/pytorch-playground/mnist/log/fullmnist/best-25.pth"))
                pretrain_resnet18.load_state_dict(torch.load("/home/fs01/yb263/drnet_restart/mixed_sudoku/pytorch-playground/mnist/log/fullmnist/best-25.pth"))
                best_epoch = _epoch_
            if args.mode == "solve" and _epoch_ - best_epoch2 >= args.resnet_gap:
                print ("restart the pretrain resnet18")
                #pretrain_resnet18.load_state_dict(torch.load("/home/fs01/yb263/Deep-Reasoning-Network-Nature/Games/Visual_Sudoku/pytorch-playground/mnist/log/fullmnist/best-25.pth"))
                #pretrain_resnet18.load_state_dict(torch.load("/home/fs01/yb263/drnet_restart/mixed_sudoku/pytorch-playground/mnist/log/fullmnist/best-25.pth"))
                pretrain_resnet18.load_state_dict(torch.load("./best-25.pth"))
                best_epoch2 = _epoch_


                    
        cnt = 0
        all_label_acc = 0
        all_sudoku_acc = 0
        all_recon_loss = 0
        sys.stdout.flush()
        pred.train()
    print ("current training time: ", time.time() - st_time)

pred_probs = np.concatenate(pred_probs, axis = 0)
gt_labels = np.concatenate(gt_labels, axis = 0)

#save_data(pred_probs, "visual_sudoku-res/pred_probs")
#save_data(gt_labels, "visual_sudoku-res/gt_labels")
