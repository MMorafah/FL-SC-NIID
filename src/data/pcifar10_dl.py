import os, sys
from sklearn.utils import shuffle
from PIL import Image
import numpy as np
from scipy import ndimage

import torch
from torchvision.datasets import MNIST, CIFAR10, SVHN, FashionMNIST, CIFAR100, ImageFolder, DatasetFolder, USPS
from torchvision.datasets.vision import VisionDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.utils.data as data
import torchvision.datasets.utils as utils
from scipy import ndimage

class CIFAR10_permutated(data.Dataset):
    def __init__(self, data, dataidxs=None):
        self.data = data
        self.dataidxs = dataidxs
        self.x, self.y = self.__build_truncated_dataset__()
    
    def __build_truncated_dataset__(self):
        x = self.data['x']
        y = self.data['y']
        
        if self.dataidxs is not None:
            x = x[self.dataidxs]
            y = y[self.dataidxs]

        return x, y

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.x[index], self.y[index]

#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.x)

    
def gen_pcifar10(datadir):
    perm_data = {}
    taskcla = []
    size = [3, 32, 32]

    nperm = 5  # 5 tasks
    seeds = np.array(list(range(nperm)), dtype=int)

    pmnist_dir = datadir + 'pcifar10/'
    if not os.path.isdir(pmnist_dir):
        print('Generating 5 random permuations CIFAR10 for the first time')
        os.makedirs(pmnist_dir)
        # Pre-load
        # MNIST
        mean = (0.1307,)
        std = (0.3081,)
        dat = {}
        dat['train'] = CIFAR10(datadir, train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        dat['test'] = CIFAR10(datadir, train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        for i, r in enumerate(seeds):
            print(i, end=',')
            sys.stdout.flush()
            perm_data[i] = {}
            perm_data[i]['name'] = 'pcifar10-{:d}'.format(i)
            perm_data[i]['ncla'] = 10
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                perm_data[i][s] = {'x': [], 'y': []}
                for image, target in loader:
                    aux = image.view(-1).numpy()
                    aux = shuffle(aux, random_state=r * 100 + i)
                    image = torch.FloatTensor(aux).view(size)
                    perm_data[i][s]['x'].append(image)
                    perm_data[i][s]['y'].append(target.numpy()[0])

            # "Unify" and save
            for s in ['train', 'test']:
                perm_data[i][s]['x'] = torch.stack(perm_data[i][s]['x']).view(-1, size[0], size[1], size[2])
                perm_data[i][s]['y'] = torch.LongTensor(np.array(perm_data[i][s]['y'], dtype=int)).view(-1)
                torch.save(perm_data[i][s]['x'],os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'x.bin'))
                torch.save(perm_data[i][s]['y'],os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'y.bin'))
        print()
        
    else:
        print('Loading 5 random permutations')
        # Load binary files
        for i, r in enumerate(seeds):
            perm_data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            perm_data[i]['ncla'] = 10
            perm_data[i]['name'] = 'pmnist-{:d}'.format(i)

            # Load
            for s in ['train', 'test']:
                perm_data[i][s] = {'x': [], 'y': []}
                perm_data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'x.bin'))
                perm_data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'y.bin'))
    
    net_dataidx_map = {}
    net_dataidx_map_test = {}
    traindata_cls_counts = {}
    clients_permdict = {}
    cnt = 0
    for i in range(nperm):
        n_train = perm_data[i]['train']['x'].shape[0]
        x_train = perm_data[i]['train']['x']
        y_train = perm_data[i]['train']['y']
        n_test = perm_data[i]['test']['x'].shape[0]
        
        n_parties=20
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties*5)
        for j in range(n_parties):
            dataidx = batch_idxs[j]
            net_dataidx_map[cnt] = dataidx
            net_dataidx_map_test[cnt] = np.arange(n_test)
            clients_permdict[cnt] = i
            
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            traindata_cls_counts[cnt] = tmp            
            cnt+=1 
    
    return net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, clients_permdict, perm_data

    
def gen_dataloader(net_dataidx_map, net_dataidx_map_test, clients_permdict, perm_data, args): 
    clients_trainds = {}
    clients_testds = {}
    clients_traindl = {}
    clients_testdl = {}
    
    n_users = len(net_dataidx_map.keys())
    train_bs = args.batch_size
    test_bs = 128
    nperm = 5
    
    for i in range(n_users): 
        p = clients_permdict[i]
        train_ds = CIFAR10_permutated(perm_data[p]['train'], dataidxs=net_dataidx_map[i])
        test_ds = CIFAR10_permutated(perm_data[p]['test'], dataidxs=net_dataidx_map_test[i])
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)
        
        clients_trainds[i] = train_ds
        clients_testds[i] = test_ds
        clients_traindl[i] = train_dl
        clients_testdl[i] = test_dl
        
    glob_testds = {}
    glob_testdl = {}
    for j in range(nperm):
        test_ds = CIFAR10_permutated(perm_data[j]['test'], dataidxs=None)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)
        glob_testds[j] = test_ds
        glob_testdl[j] = test_dl
        
    return clients_trainds, clients_testds, clients_traindl, clients_testdl, glob_testds, glob_testdl
