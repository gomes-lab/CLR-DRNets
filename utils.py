import numpy as np
import torch
import copy
import sys
import torch.nn as nn
from torch.utils.data import Dataset

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

def augment_X(Q, debug=False, at=0):
    Q = Q.reshape(9, 9)
    init_flag = np.random.rand()
    if init_flag < 0.25:
        return Q.reshape(-1)
    aug_type = np.random.randint(6)
    if debug:
        aug_type = at

    if debug:
        print ("original Q: ")
        print (Q)

    if aug_type == 0:
        # Random permutation
        x = np.random.permutation(np.arange(9))
        x += 1
        assert (np.min(x) >= 1)
        assert (np.max(x) <= 9)
        tQ = copy.deepcopy(Q)
        for i in range(1, 10):
            Q[tQ == i] = x[i - 1]
    elif aug_type == 1:
        # Transposing the matrix
        Q = np.transpose(Q)
    elif aug_type == 2:
        # permutation rows within a single block
        for i in range(3):
            y = np.arange(3)
            x = np.random.permutation(np.arange(3))
            x += i * 3
            y += i * 3
            Q[y, :] = Q[x, :]
    elif aug_type == 3:
        # permutation cols within a single block
        for i in range(3):
            y = np.arange(3)
            x = np.random.permutation(np.arange(3))
            x += i * 3
            y += i * 3
            Q[:, y] = Q[:, x]
    elif aug_type == 4:
        # permutation the blocks row-wise
        tQ = copy.deepcopy(Q)
        x = np.random.permutation(np.arange(3))
        for i in range(3):
            for j in range(3):
                Q[i * 3: i * 3 + 3, j * 3: j * 3 + 3] = tQ[i * 3: i * 3 + 3, x[j] * 3: x[j] * 3 + 3]
    elif aug_type == 5:
        # permutation the blocks col-wise
        tQ = copy.deepcopy(Q)
        x = np.random.permutation(np.arange(3))
        for i in range(3):
            for j in range(3):
                Q[j * 3: j * 3 + 3, i * 3: i * 3 + 3] = tQ[x[j] * 3: x[j] * 3 + 3, i * 3: i * 3 + 3]
    if debug:
        print ("after augment Q: ")
        print (Q)

    return Q.reshape(-1)


def augment(Q, l, debug=False, at=0):
    Q = Q.reshape(9, 9)
    l = l.reshape(9, 9)
    init_flag = np.random.rand()
    if init_flag < 0.25:
        return Q.reshape(-1), l.reshape(-1)
    aug_type = np.random.randint(6)
    if debug:
        aug_type = at

    if debug:
        print ("original Q: ")
        print (Q)

    if aug_type == 0:
        # Random permutation
        x = np.random.permutation(np.arange(9))
        x += 1
        assert (np.min(x) >= 1)
        assert (np.max(x) <= 9)
        tQ = copy.deepcopy(Q)
        tl = copy.deepcopy(l)
        for i in range(1, 10):
            Q[tQ == i] = x[i - 1]
            l[tl == i] = x[i - 1]
    elif aug_type == 1:
        # Transposing the matrix
        Q = np.transpose(Q)
        l = np.transpose(l)
    elif aug_type == 2:
        # permutation rows within a single block
        for i in range(3):
            y = np.arange(3)
            x = np.random.permutation(np.arange(3))
            x += i * 3
            y += i * 3
            Q[y, :] = Q[x, :]
            l[y, :] = l[x, :]
    elif aug_type == 3:
        # permutation cols within a single block
        for i in range(3):
            y = np.arange(3)
            x = np.random.permutation(np.arange(3))
            x += i * 3
            y += i * 3
            Q[:, y] = Q[:, x]
            l[:, y] = l[:, x]
    elif aug_type == 4:
        # permutation the blocks row-wise
        tQ = copy.deepcopy(Q)
        tl = copy.deepcopy(l)
        x = np.random.permutation(np.arange(3))
        for i in range(3):
            for j in range(3):
                Q[i * 3: i * 3 + 3, j * 3: j * 3 + 3] = tQ[i * 3: i * 3 + 3, x[j] * 3: x[j] * 3 + 3]
                l[i * 3: i * 3 + 3, j * 3: j * 3 + 3] = tl[i * 3: i * 3 + 3, x[j] * 3: x[j] * 3 + 3]
    elif aug_type == 5:
        # permutation the blocks col-wise
        tQ = copy.deepcopy(Q)
        tl = copy.deepcopy(l)
        x = np.random.permutation(np.arange(3))
        for i in range(3):
            for j in range(3):
                Q[j * 3: j * 3 + 3, i * 3: i * 3 + 3] = tQ[x[j] * 3: x[j] * 3 + 3, i * 3: i * 3 + 3]
                l[j * 3: j * 3 + 3, i * 3: i * 3 + 3] = tl[x[j] * 3: x[j] * 3 + 3, i * 3: i * 3 + 3]
    if debug:
        print ("after augment Q: ")
        print (Q)

    return Q.reshape(-1), l.reshape(-1)

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

n_classes = 10
latent_dim = 100
img_shape = (1, 28, 28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        #print ("label: " labels)
        #print ("!!!!!!!!!!!!!!!1 ", self.label_emb(labels).size(), " ", noise.size())
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img



def gen_local_embs(batch_size, nums, use_cuda):
    loc_emb = np.zeros((batch_size, nums**2, nums**2))
    for i in range(batch_size):
        for j in range(nums**2):
            loc_emb[i][j][j]=1
    loc_emb = Variable(torch.tensor(loc_emb).float(), requires_grad=False)
    return loc_emb

def compute_gen_labels(batch_size, nums, use_cuda):
    labels = []
    for i in range(nums ** 2 * batch_size):
        for j in range(10):
            labels.append(j)
    gen_labels = torch.LongTensor(labels)
    if use_cuda:
        gen_labels = gen_labels.cuda()
    return gen_labels
# Note paramB should be a lost of the numpy array
def model_similarity_loss(paramA, paramB):
    loss = 0
    cnt = 0
    for a, b in zip(paramA, iter(paramB)):
        c  = torch.from_numpy(b).cuda()
        loss -= torch.sum(torch.abs(a - c) ** 2)
        cnt += 1
    return loss / cnt

class SudokuDataset(Dataset):

    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.y is None:
            return self.X[idx, ...]
        else:
            return self.X[idx, ...], self.y[idx, ...]

class MetaSudokuDataset(Dataset):

    def __init__(self, X, model_perf, tt, sss, model_ls, alpha=0.2, transform=1):
        self.X = X
        self.alpha = alpha
        self.model_perf = model_perf
        self.tt = tt
        self.sss = sss
        self.model_ls = model_ls
        self.transform = transform
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ret = [0 for _ in range(len(self.model_ls))] 
        #ret = [x - 1 for x in self.model_ls]
        for pos, model_idx in enumerate(self.model_ls):
            if self.model_perf[(self.tt, self.sss, model_idx)][idx] == 1.0:
                ret[pos] = 1.0 - self.alpha + float(self.alpha) / float(len(self.model_ls))
            else:
                ret[pos] = float(self.alpha) / float(len(self.model_ls))
        if self.transform:
            ret_X = augment_X(self.X[idx, ...])
        else:
            ret_X = self.X[idx, ...]
        #return self.X[idx, ...], np.array(ret)
        return ret_X, np.array(ret)

class MixMetaSudokuDataset(Dataset):

    def __init__(self, X, model_perf, hints_map, tt, sss, model_ls, alpha=0.2, transform=1):
        self.X = X
        self.alpha = alpha
        self.hints_map = hints_map
        self.model_perf = model_perf
        self.tt = tt
        self.sss = sss
        self.model_ls = model_ls
        self.transform = transform
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ret = [0 for _ in range(len(self.model_ls))] 
        #ret = [x - 1 for x in self.model_ls]
        for pos, model_idx in enumerate(self.model_ls):
            #print (idx, " ", self.hints_map[(self.tt, idx)])
            if self.model_perf[(self.tt, self.hints_map[(self.tt, idx)][0], model_idx)][self.hints_map[(self.tt, idx)][1]] == 1.0:
                ret[pos] = 1.0 - self.alpha + float(self.alpha) / float(len(self.model_ls))
            else:
                ret[pos] = float(self.alpha) / float(len(self.model_ls))
        if self.transform:
            ret_X = augment_X(self.X[idx, ...])
        else:
            ret_X = self.X[idx, ...]
        #return self.X[idx, ...], np.array(ret)
        return ret_X, np.array(ret)

class PureSudokuDataset(Dataset):

    def __init__(self, Q, ori_label):
        self.Q = Q
        self.ori_label = ori_label

    def __len__(self):
        return len(self.Q)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.Q[idx, ...], self.ori_label[idx, ...]


if __name__ == "__main__":
    a = np.zeros((9, 9))
    a[0][0] = 1
    a[1][2] = 2
    a[2][3] = 5
    a[3][4] = 4
    a[4][2] = 5
    a[5][7] = 9
    a[6][0] = 2
    a[7][8] = 1
    a[8][8] = 9
    for i in range(1):
        ss = int(sys.argv[1])
        print ("type: ", ss)
        augment(copy.deepcopy(a), a, debug=True, at=ss)
        sys.stdout.flush()
