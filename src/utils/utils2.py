import sys
import os
#sys.path.append("..")
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from src.data import *
from src.models import *
from src.utils import * 

import numpy as np

import copy
import gc 

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class Logger(object):
    def __init__(self, fname):
        self.terminal = sys.stdout
        self.log = open(fname, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass 

def FedAvg(w, weight_avg=None):
    """
    Federated averaging
    :param w: list of client model parameters
    :return: updated server model parameters
    """
    if weight_avg == None:
        weight_avg = [1/len(w) for i in range(len(w))]
        
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k].cuda() * weight_avg[0]
        
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k].cuda() + w[i][k].cuda() * weight_avg[i]
        #w_avg[k] = torch.div(w_avg[k].cuda(), len(w)) 
    return w_avg

def init_nets(args, dropout_p=0.5):

    users_model = []

    for net_i in range(-1, args.num_users):
        if args.dataset == "generated":
            net = PerceptronModel().to(args.device)
        elif args.model == "mlp":
            if args.dataset == 'covtype':
                input_size = 54
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'a9a':
                input_size = 123
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'rcv1':
                input_size = 47236
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'SUSY':
                input_size = 18
                output_size = 2
                hidden_sizes = [16,8]
            net = FcNet(input_size, hidden_sizes, output_size, dropout_p).to(args.device)
        elif args.model == "vgg":
            net = vgg11().to(args.device)
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(args.device)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).to(args.device)
            elif args.dataset == 'celeba':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2).to(args.device)
        elif args.model =="simple-cnn-3":
            if args.dataset == 'cifar100': 
                net = SimpleCNN_3(input_dim=(16 * 3 * 5 * 5), hidden_dims=[120*3, 84*3], output_dim=100).to(args.device)
            if args.dataset == 'tinyimagenet':
                net = SimpleCNNTinyImagenet_3(input_dim=(16 * 3 * 13 * 13), hidden_dims=[120*3, 84*3], 
                                              output_dim=200).to(args.device)
        elif args.model == "vgg-9":
            if args.dataset in ("mnist", 'femnist'):
                net = ModerateCNNMNIST().to(args.device)
            elif args.dataset in ("cifar10", "cinic10", "svhn"):
                # print("in moderate cnn")
                net = ModerateCNN().to(args.device)
            elif args.dataset == 'celeba':
                net = ModerateCNN(output_dim=2).to(args.device)
        elif args.model == 'resnet9': 
            if args.dataset in ['cifar100']: 
                net = ResNet9(in_channels=3, num_classes=100)
            elif args.dataset == 'stl10':
                net = ResNet9(in_channels=3, num_classes=100, dim=4608)
            elif args.dataset == 'tinyimagenet': 
                net = ResNet9(in_channels=3, num_classes=200, dim=512*2*2)
        elif args.model == "resnet":
            net = ResNet50_cifar10().to(args.device)
        elif args.model == "vgg16":
            net = vgg16().to(args.device)
        else:
            print("not supported yet")
            exit(1)
        if net_i == -1: 
            net_glob = copy.deepcopy(net)
            initial_state_dict = copy.deepcopy(net_glob.state_dict())
            server_state_dict = copy.deepcopy(net_glob.state_dict())
            if args.load_initial:
                initial_state_dict = torch.load(args.load_initial)
                server_state_dict = torch.load(args.load_initial)
                net_glob.load_state_dict(initial_state_dict)
        else:
            users_model.append(copy.deepcopy(net))
            users_model[net_i].load_state_dict(initial_state_dict)

    return users_model, net_glob, initial_state_dict, server_state_dict


def get_clients_data(args):
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                       args.datadir,
                                                                                       args.batch_size,
                                                                                       32)
    if args.partition[0:2] == 'sc':
        if args.dataset == 'cifar10':        
            if args.partition == 'sc_niid_dir':
                print('Loading CIFAR10 SuperClass NIID Dir for all clients')

                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = CIFAR10_SuperClass_NIID_DIR(train_ds_global, test_ds_global, args)

            elif args.partition[0:7] == 'sc_niid':
                print('Loading CIFAR10 SuperClass NIID for all clients')

                num = eval(args.partition[7:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = CIFAR10_SuperClass_NIID(train_ds_global, test_ds_global, num, args)
                
            elif args.partition == 'sc_old_niid_dir':
                print('Loading CIFAR10 SuperClass OLD NIID Dir for all clients')

                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = CIFAR10_SuperClass_Old_NIID_DIR(train_ds_global, test_ds_global, args)
            
            elif args.partition[0:11] == 'sc_old_niid':
                print('Loading CIFAR10 SuperClass OLD NIID for all clients')

                num = eval(args.partition[11:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = CIFAR10_SuperClass_Old_NIID(train_ds_global, test_ds_global, num, args)

        elif args.dataset == 'cifar100':
            if args.partition == 'sc_niid_dir':
                print('Loading CIFAR100 SuperClass NIID Dir for all clients')

                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = CIFAR100_SuperClass_NIID_DIR(train_ds_global, test_ds_global, args)
                
            elif args.partition[0:7] == 'sc_niid':
                print('Loading CIFAR100 SuperClass NIID for all clients')

                num = eval(args.partition[7:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = CIFAR100_SuperClass_NIID(train_ds_global, test_ds_global, args)
                
            elif args.partition == 'sc_old_niid_dir':
                print('Loading CIFAR100 SuperClass OLD NIID Dir for all clients')

                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = CIFAR100_SuperClass_Old_NIID_DIR(train_ds_global, test_ds_global, args)
                
            elif args.partition[0:11] == 'sc_old_niid':
                print('Loading CIFAR100 SuperClass OLD NIID for all clients')

                num = eval(args.partition[11:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = CIFAR100_SuperClass_Old_NIID(train_ds_global, test_ds_global, args)
        
        elif args.dataset == 'stl10':
            if args.partition == 'sc_niid_dir':
                print('Loading STL10 SuperClass NIID Dir for all clients')

                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = STL10_SuperClass_NIID_DIR(train_ds_global, test_ds_global, args)
                
            elif args.partition[0:7] == 'sc_niid':
                print('Loading STL10 SuperClass NIID for all clients')

                num = eval(args.partition[7:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = STL10_SuperClass_NIID(train_ds_global, test_ds_global, num, args)
                
            elif args.partition == 'sc_old_niid_dir':
                print('Loading STL10 SuperClass OLD NIID Dir for all clients')

                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = STL10_SuperClass_Old_NIID_DIR(train_ds_global, test_ds_global, args)
                
            elif args.partition[0:11] == 'sc_old_niid':
                print('Loading STL10 SuperClass OLD NIID for all clients')

                num = eval(args.partition[11:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = STL10_SuperClass_Old_NIID(train_ds_global, test_ds_global, num, args)
                
        elif args.dataset == 'fmnist':
            if args.partition == 'sc_niid_dir':
                pass
            elif args.partition[0:7] == 'sc_niid':
                print('Loading FMNIST SuperClass NIID for all clients')

                num = eval(args.partition[7:])
                net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
                = FMNIST_SuperClass_NIID(train_ds_global, test_ds_global, args)
    else:
        print(f'Loading {args.dataset}, {args.partition} for all clients')
        args.local_view = True
        X_train, y_train, X_test, y_test, net_dataidx_map, net_dataidx_map_test, \
        traindata_cls_counts, testdata_cls_counts = partition_data(args.dataset, 
        args.datadir, args.logdir, args.partition, args.num_users, beta=args.beta, local_view=args.local_view)

    return train_dl_global, test_dl_global, train_ds_global, test_ds_global, net_dataidx_map, net_dataidx_map_test, \
            traindata_cls_counts, testdata_cls_counts

