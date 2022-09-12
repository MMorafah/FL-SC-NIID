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


def CIFAR100_SuperClass(train_ds, test_ds, args):
    
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
    
    superclass = [[0, 83, 53, 82], [1, 54, 43, 51, 70, 92, 62], 
                [2, 97, 27, 65, 64, 36, 28, 61, 99, 18, 77, 79, 80, 34, 88, 42, 38, 44, 
                63, 50, 78, 66, 84, 8, 39, 55, 72, 93, 91, 3, 4, 29, 31, 7, 24, 20, 
                26, 45, 74, 5, 25, 15, 19, 32, 9, 16, 10, 22, 40, 11, 35, 98, 46, 6, 
                14, 57, 94, 56, 13, 58, 37, 81, 90, 89, 85, 21, 48, 86, 87, 41, 75, 
                12, 71, 49, 17, 60, 76, 33, 68], 
                [23, 69, 30, 95, 67, 73], [47, 96, 59, 52]]

    nclass=100
    idxs_superclass = {}
    net_dataidx_map = {}
    net_dataidx_map_test = {}
    traindata_cls_counts = {}
    testdata_cls_counts = {}
    cnt=0
    
    n_parties_ratio = np.array([1/len(superclass) for i in range(len(superclass))])
    for i in range(len(superclass)):
        n_parties_ratio[i]+= n_parties_ratio[i]*(len(superclass[i])/nclass)
        
    n_parties_ratio = n_parties_ratio/sum(n_parties_ratio)
    n_parties_ratio = [int(np.ceil(el*args.num_users)) for el in n_parties_ratio]
    s = sum(n_parties_ratio)
    if s>args.num_users:
        inds = np.random.choice(len(n_parties_ratio), size=s-args.num_users, replace=True)
        for _i in inds:
            n_parties_ratio[_i]-=1
    elif s<args.num_users:
        inds = np.random.choice(len(n_parties_ratio), size=args.num_users-s, replace=True)
        for _i in inds:
            n_parties_ratio[_i]+=1

    assert sum(n_parties_ratio)==args.num_users
    
    for r, clust in enumerate(superclass):
        temp_idxs = np.array([], dtype='int')
        for c in clust:
            temp_idxs = np.hstack([temp_idxs, idxs_train[c*500:(c+1)*500]])

        idxs_superclass[r] = temp_idxs
        
        #n_parties=int(len(clust)/nclass*args.num_users)
        n_parties = n_parties_ratio[r]
        np.random.shuffle(idxs_superclass[r])

        batch_idxs = np.array_split(idxs_superclass[r], n_parties)
        for j in range(n_parties):
            dataidx = batch_idxs[j]
            net_dataidx_map[cnt] = dataidx
            net_dataidx_map_test[cnt] = np.array([], dtype='int')
            for cc in clust:
                net_dataidx_map_test[cnt] = np.hstack([net_dataidx_map_test[cnt],
                                                       idxs_test[cc*100:(cc+1)*100]])

            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            traindata_cls_counts[cnt] = tmp    

            unq, unq_cnt = np.unique(y_test[net_dataidx_map_test[cnt]], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            testdata_cls_counts[cnt] = tmp    
            cnt+=1 
            
    return net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts

def CIFAR100_SuperClass_NIID(train_ds, test_ds, args):
    
    idxs_train = np.arange(len(train_ds))
    labels_train = np.array(train_ds.target)
    y_train = np.array(train_ds.target)
    
    idxs_test = np.arange(len(test_ds))
    labels_test = np.array(test_ds.target)
    y_test = np.array(test_ds.target)
    
    superclass = [[0, 83, 53, 82], [1, 54, 43, 51, 70, 92, 62], 
                [2, 97, 27, 65, 64, 36, 28, 61, 99, 18, 77, 79, 80, 34, 88, 42, 38, 44, 
                63, 50, 78, 66, 84, 8, 39, 55, 72, 93, 91, 3, 4, 29, 31, 7, 24, 20, 
                26, 45, 74, 5, 25, 15, 19, 32, 9, 16, 10, 22, 40, 11, 35, 98, 46, 6, 
                14, 57, 94, 56, 13, 58, 37, 81, 90, 89, 85, 21, 48, 86, 87, 41, 75, 
                12, 71, 49, 17, 60, 76, 33, 68], 
                [23, 69, 30, 95, 67, 73], [47, 96, 59, 52]]
        
#     superclass = [[0, 83, 53, 82, 1, 54, 43, 51, 70, 92, 62], 
#                 [2, 97, 27, 65, 64, 36, 28, 61, 99, 18, 77, 79, 80, 34, 88, 42, 38, 44, 
#                 63, 50, 78, 66, 84, 8, 39, 55, 72, 93, 91, 3, 4, 29, 31, 7, 24, 20, 
#                 26, 45, 74, 5, 25, 15, 19, 32, 9, 16, 10, 22, 40, 11, 35, 98, 46, 6, 
#                 14, 57, 94, 56, 13, 58, 37, 81, 90, 89, 85, 21, 48, 86, 87, 41, 75, 
#                 12, 71, 49, 17, 60, 76, 33, 68, 23, 69, 30, 95, 67, 73, 47, 96, 59, 52]] ## 2
    
    nclass=100
    idxs_superclass = {}
    net_dataidx_map = {i:np.array([],dtype='int') for i in range(args.num_users)}
    net_dataidx_map_test = {i:np.array([],dtype='int') for i in range(args.num_users)}
    traindata_cls_counts = {}
    testdata_cls_counts = {}
    cnt=0
    
    n_parties_ratio = np.array([1/len(superclass) for i in range(len(superclass))])
    for i in range(len(superclass)):
        n_parties_ratio[i]+= n_parties_ratio[i]*(len(superclass[i])/nclass)
        
    n_parties_ratio = n_parties_ratio/sum(n_parties_ratio)
    n_parties_ratio = [int(np.ceil(el*args.num_users)) for el in n_parties_ratio]
    s = sum(n_parties_ratio)
    if s>args.num_users:
        inds = np.random.choice(len(n_parties_ratio), size=s-args.num_users, replace=True)
        for _i in inds:
            n_parties_ratio[_i]-=1
    elif s<args.num_users:
        inds = np.random.choice(len(n_parties_ratio), size=args.num_users-s, replace=True)
        for _i in inds:
            n_parties_ratio[_i]+=1

    assert sum(n_parties_ratio)==args.num_users
    
    for r, clust in enumerate(superclass):        
        ##### Forming the labels for each clients
        #n_parties=int(len(clust)/nclass*args.num_users)
        n_parties = n_parties_ratio[r]
        
        num = int(np.ceil(0.2*len(clust)))
        if num<2:
            num=2
        
        check=True
        while check:
            times={el:0 for el in clust}
            contain={j:None for j in range(cnt, cnt+n_parties)}
            for j in range(cnt, cnt+n_parties):
                ind_a = j%len(clust)
                rand_labels = [clust[ind_a]]
                kk=1
                while (kk<num):
                    ind=np.random.choice(clust, size=1, replace=False).tolist()[0]
                    if (ind not in rand_labels):
                        kk+=1
                        rand_labels.append(ind)
                #rand_labels = np.random.choice(clust, size=num, replace=False)
                contain[j] = rand_labels
                for el in rand_labels:
                    times[el]+=1

            check = not np.all([el>0 for el in times.values()])
        #print(times)
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

def CIFAR100_SuperClass_Old_NIID(train_ds, test_ds, args):
    
    idxs_train = np.arange(len(train_ds))
    labels_train = np.array(train_ds.target)
    y_train = np.array(train_ds.target)
    
    idxs_test = np.arange(len(test_ds))
    labels_test = np.array(test_ds.target)
    y_test = np.array(test_ds.target)
   
    nclass=100
    classes=np.arange(nclass)
    sc1 = np.random.choice(classes, size=4, replace=False).tolist()
    classes = np.delete(classes, np.where(classes==sc1))
    sc2 = np.random.choice(classes, size=7, replace=False).tolist()
    classes = np.delete(classes, np.where(classes==sc2))
    sc3 = np.random.choice(classes, size=79, replace=False).tolist()
    classes = np.delete(classes, np.where(classes==sc3))
    sc4 = np.random.choice(classes, size=6, replace=False).tolist()
    classes = np.delete(classes, np.where(classes==sc4))
    sc5 = np.random.choice(classes, size=4, replace=False).tolist()    
    superclass = [sc1, sc2, sc3, sc4, sc5]
    
    idxs_superclass = {}
    net_dataidx_map = {i:np.array([],dtype='int') for i in range(args.num_users)}
    net_dataidx_map_test = {i:np.array([],dtype='int') for i in range(args.num_users)}
    traindata_cls_counts = {}
    testdata_cls_counts = {}
    cnt=0
    
    n_parties_ratio = np.array([1/len(superclass) for i in range(len(superclass))])
    for i in range(len(superclass)):
        n_parties_ratio[i]+= n_parties_ratio[i]*(len(superclass[i])/nclass)
        
    n_parties_ratio = n_parties_ratio/sum(n_parties_ratio)
    n_parties_ratio = [int(np.ceil(el*args.num_users)) for el in n_parties_ratio]
    s = sum(n_parties_ratio)
    if s>args.num_users:
        inds = np.random.choice(len(n_parties_ratio), size=s-args.num_users, replace=True)
        for _i in inds:
            n_parties_ratio[_i]-=1
    elif s<args.num_users:
        inds = np.random.choice(len(n_parties_ratio), size=args.num_users-s, replace=True)
        for _i in inds:
            n_parties_ratio[_i]+=1

    assert sum(n_parties_ratio)==args.num_users
    
    for r, clust in enumerate(superclass):        
        ##### Forming the labels for each clients
        #n_parties=int(len(clust)/nclass*args.num_users)
        n_parties = n_parties_ratio[r]
        
        num = int(np.ceil(0.2*len(clust)))
        if num<2:
            num=2
        
        check=True
        while check:
            times={el:0 for el in clust}
            contain={j:None for j in range(cnt, cnt+n_parties)}
            for j in range(cnt, cnt+n_parties):
                ind_a = j%len(clust)
                rand_labels = [clust[ind_a]]
                kk=1
                while (kk<num):
                    ind=np.random.choice(clust, size=1, replace=False).tolist()[0]
                    if (ind not in rand_labels):
                        kk+=1
                        rand_labels.append(ind)
                #rand_labels = np.random.choice(clust, size=num, replace=False)
                contain[j] = rand_labels
                for el in rand_labels:
                    times[el]+=1

            check = not np.all([el>0 for el in times.values()])
        #print(times)
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

def CIFAR100_SuperClass_NIID_DIR(train_ds, test_ds, args):
    
    labels_train = np.array(train_ds.target)
    y_train = np.array(train_ds.target)
    
    labels_test = np.array(test_ds.target)
    y_test = np.array(test_ds.target)
    
    superclass = [[0, 83, 53, 82], [1, 54, 43, 51, 70, 92, 62], 
                [2, 97, 27, 65, 64, 36, 28, 61, 99, 18, 77, 79, 80, 34, 88, 42, 38, 44, 
                63, 50, 78, 66, 84, 8, 39, 55, 72, 93, 91, 3, 4, 29, 31, 7, 24, 20, 
                26, 45, 74, 5, 25, 15, 19, 32, 9, 16, 10, 22, 40, 11, 35, 98, 46, 6, 
                14, 57, 94, 56, 13, 58, 37, 81, 90, 89, 85, 21, 48, 86, 87, 41, 75, 
                12, 71, 49, 17, 60, 76, 33, 68], 
                [23, 69, 30, 95, 67, 73], [47, 96, 59, 52]]
    
    nclass=100
    idxs_superclass = {}
    net_dataidx_map = {i:np.array([],dtype='int') for i in range(args.num_users)}
    net_dataidx_map_test = {i:np.array([],dtype='int') for i in range(args.num_users)}
    traindata_cls_counts = {}
    testdata_cls_counts = {}
    cnt=0
    
    n_parties_ratio = np.array([1/len(superclass) for i in range(len(superclass))])
    for i in range(len(superclass)):
        n_parties_ratio[i]+= n_parties_ratio[i]*(len(superclass[i])/nclass)
        
    n_parties_ratio = n_parties_ratio/sum(n_parties_ratio)
    n_parties_ratio = [int(np.ceil(el*args.num_users)) for el in n_parties_ratio]
    s = sum(n_parties_ratio)
    if s>args.num_users:
        inds = np.random.choice(len(n_parties_ratio), size=s-args.num_users, replace=True)
        for _i in inds:
            n_parties_ratio[_i]-=1
    elif s<args.num_users:
        inds = np.random.choice(len(n_parties_ratio), size=args.num_users-s, replace=True)
        for _i in inds:
            n_parties_ratio[_i]+=1

    assert sum(n_parties_ratio)==args.num_users
    
    for r, clust in enumerate(superclass):        
        ##### Forming the labels for each clients
        #n_parties=int(len(clust)/nclass*args.num_users)
        n_parties = n_parties_ratio[r]
        N=int(len(clust)*500)
        
        min_size = 0
        min_require_size = 15
        beta=args.beta
        #beta = 0.5
        #np.random.seed(2021)
        #print(clust)
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in clust:
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)

                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            #print(sum([len(idx_j) for idx_j in idx_batch]))
            min_size = min([len(idx_j) for idx_j in idx_batch])
            
        #### Assigning samples to each client         
        for j in range(cnt, cnt+n_parties):
            np.random.shuffle(idx_batch[j-cnt])
            net_dataidx_map[j] = np.hstack([net_dataidx_map[j], idx_batch[j-cnt]])
    
            dataidx = net_dataidx_map[j]           
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            traindata_cls_counts[j] = tmp   
            
            for key in tmp.keys():
                idxs_test = np.where(y_test==key)[0]
                net_dataidx_map_test[j] = np.hstack([net_dataidx_map_test[j], idxs_test])
                
            dataidx = net_dataidx_map_test[j]
            unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            testdata_cls_counts[j] = tmp    
        
        cnt+=n_parties
            
    return net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts

def CIFAR100_SuperClass_Old_NIID_DIR(train_ds, test_ds, args):
    
    labels_train = np.array(train_ds.target)
    y_train = np.array(train_ds.target)
    
    labels_test = np.array(test_ds.target)
    y_test = np.array(test_ds.target)
    
    nclass=100
    classes=np.arange(nclass)
    sc1 = np.random.choice(classes, size=4, replace=False).tolist()
    classes = np.delete(classes, np.where(classes==sc1))
    sc2 = np.random.choice(classes, size=7, replace=False).tolist()
    classes = np.delete(classes, np.where(classes==sc2))
    sc3 = np.random.choice(classes, size=79, replace=False).tolist()
    classes = np.delete(classes, np.where(classes==sc3))
    sc4 = np.random.choice(classes, size=6, replace=False).tolist()
    classes = np.delete(classes, np.where(classes==sc4))
    sc5 = np.random.choice(classes, size=4, replace=False).tolist()    
    superclass = [sc1, sc2, sc3, sc4, sc5]
    print(superclass)
    
    idxs_superclass = {}
    net_dataidx_map = {i:np.array([],dtype='int') for i in range(args.num_users)}
    net_dataidx_map_test = {i:np.array([],dtype='int') for i in range(args.num_users)}
    traindata_cls_counts = {}
    testdata_cls_counts = {}
    cnt=0
    
    n_parties_ratio = np.array([1/len(superclass) for i in range(len(superclass))])
    for i in range(len(superclass)):
        n_parties_ratio[i]+= n_parties_ratio[i]*(len(superclass[i])/nclass)
        
    n_parties_ratio = n_parties_ratio/sum(n_parties_ratio)
    n_parties_ratio = [int(np.ceil(el*args.num_users)) for el in n_parties_ratio]
    s = sum(n_parties_ratio)
    if s>args.num_users:
        inds = np.random.choice(len(n_parties_ratio), size=s-args.num_users, replace=True)
        for _i in inds:
            n_parties_ratio[_i]-=1
    elif s<args.num_users:
        inds = np.random.choice(len(n_parties_ratio), size=args.num_users-s, replace=True)
        for _i in inds:
            n_parties_ratio[_i]+=1

    assert sum(n_parties_ratio)==args.num_users
    
    for r, clust in enumerate(superclass):        
        ##### Forming the labels for each clients
        #n_parties=int(len(clust)/nclass*args.num_users)
        n_parties = n_parties_ratio[r]
        N=int(len(clust)*500)
        
        min_size = 0
        min_require_size = 15
        beta=args.beta
        #beta = 0.5
        #np.random.seed(2021)
        #print(clust)
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in clust:
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)

                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            #print(sum([len(idx_j) for idx_j in idx_batch]))
            min_size = min([len(idx_j) for idx_j in idx_batch])
            
        #### Assigning samples to each client         
        for j in range(cnt, cnt+n_parties):
            np.random.shuffle(idx_batch[j-cnt])
            net_dataidx_map[j] = np.hstack([net_dataidx_map[j], idx_batch[j-cnt]])
    
            dataidx = net_dataidx_map[j]           
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            traindata_cls_counts[j] = tmp   
            
            for key in tmp.keys():
                idxs_test = np.where(y_test==key)[0]
                net_dataidx_map_test[j] = np.hstack([net_dataidx_map_test[j], idxs_test])
                
            dataidx = net_dataidx_map_test[j]
            unq, unq_cnt = np.unique(y_test[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            testdata_cls_counts[j] = tmp    
        
        cnt+=n_parties
            
    return net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts