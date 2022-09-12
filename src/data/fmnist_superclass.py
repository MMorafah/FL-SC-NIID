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

def FMNIST_SuperClass(train_ds, test_ds, args):
    
    idxs_train = np.arange(len(train_ds))
    labels_train = np.array(train_ds.target)
    y_train = np.array(train_ds.target)
    # Sort Labels Train 
    idxs_labels_train = np.vstack((idxs_train, labels_train))
    idxs_labels_train = idxs_labels_train[:, idxs_labels_train[1, :].argsort()]
    idxs_train = idxs_labels_train[0, :]
    labels_train = idxs_labels_train[1, :]
        
    idxs_test = np.arange(len(test_ds))
    labels_test = np.array(test_ds.target)
    y_test = np.array(test_ds.target)
    # Sort Labels Test 
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    labels_test = idxs_labels_test[1, :]
    
    superclass = [[0, 3, 1], [2, 4, 6, 8], [5, 9, 7]]
    nclass=10
    idxs_superclass = {}
    net_dataidx_map = {}
    net_dataidx_map_test = {}
    traindata_cls_counts = {}
    testdata_cls_counts = {}
    cnt=0
    for r, clust in enumerate(superclass):
        temp_idxs = np.array([], dtype='int')
        for c in clust:
            temp_idxs = np.hstack([temp_idxs, idxs_train[c*6000:(c+1)*6000]])

        idxs_superclass[r] = temp_idxs

        n_parties=int(len(clust)/nclass*args.num_users)
        np.random.shuffle(idxs_superclass[r])

        batch_idxs = np.array_split(idxs_superclass[r], n_parties)
        for j in range(n_parties):
            dataidx = batch_idxs[j]
            net_dataidx_map[cnt] = dataidx
            net_dataidx_map_test[cnt] = np.array([], dtype='int')
            for cc in clust:
                net_dataidx_map_test[cnt] = np.hstack([net_dataidx_map_test[cnt],
                                                       idxs_test[cc*1000:(cc+1)*1000]])

            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            traindata_cls_counts[cnt] = tmp    

            unq, unq_cnt = np.unique(y_test[net_dataidx_map_test[cnt]], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            testdata_cls_counts[cnt] = tmp    
            cnt+=1 
            
    return net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts

def FMNIST_SuperClass_NIID(train_ds, test_ds, args):
    
    idxs_train = np.arange(len(train_ds))
    labels_train = np.array(train_ds.target)
    y_train = np.array(train_ds.target)
    
    idxs_test = np.arange(len(test_ds))
    labels_test = np.array(test_ds.target)
    y_test = np.array(test_ds.target)
    
    superclass = [[0, 3, 1], [2, 4, 6, 8], [5, 9, 7]]
    nclass=10
    idxs_superclass = {}
    net_dataidx_map = {i:np.array([],dtype='int') for i in range(args.num_users)}
    net_dataidx_map_test = {i:np.array([],dtype='int') for i in range(args.num_users)}
    traindata_cls_counts = {}
    testdata_cls_counts = {}
    cnt=0
    for r, clust in enumerate(superclass):        
        ##### Forming the labels for each clients
        n_parties=int(len(clust)/nclass*args.num_users)
        
        times={el:0 for el in clust}
        contain={j:None for j in range(cnt, cnt+n_parties)}
        num=3
        for j in range(cnt, cnt+n_parties):
            rand_labels = np.random.choice(clust, size=num, replace=False)
            contain[j] = rand_labels
            for el in rand_labels:
                times[el]+=1
        
        #### Assigning samples to each client 
        for el in clust:
            idx_k = np.where(y_train==el)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k,times[el])
            
            idxs_test = np.where(y_test==el)[0]
            ids=0
            for j in range(cnt, cnt+n_parties):
                if el in contain[j]:
                    net_dataidx_map[j]=np.hstack([net_dataidx_map[j], split[ids]])
                    net_dataidx_map_test[j] = np.hstack([net_dataidx_map_test[j], idxs_test])
                    ids+=1
                    
        for j in range(cnt, cnt+n_parties):
            dataidx = net_dataidx_map[j]           
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            traindata_cls_counts[j] = tmp    
            
            dataidx = net_dataidx_map_test[j]
            unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            testdata_cls_counts[j] = tmp    
        
        cnt+=n_parties
            
    return net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts
