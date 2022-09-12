import numpy as np

import copy
import os 
import gc 
import time 
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.data import *
from src.models import *
from src.fedavg import *
from src.client import * 
from src.clustering import *
from src.utils import * 

print('-'*40)
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.gpu) ## Setting cuda on GPU 

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass
    
path = args.savedir + args.alg + '/' + args.dataset + '/'
mkdirs(path)
template = "Algorithm {}, Clients {}, Dataset {}, Model {}, Non-IID {}, Threshold {}, K {}, Linkage {}, LR {}, Ep {}, Rounds {}, bs {}, frac {}"

s = template.format(args.alg, args.num_users, args.dataset, args.model, args.partition, args.cluster_alpha, args.n_basis, args.linkage, args.lr, args.local_ep, args.rounds, args.local_ep, args.frac)
print(s)
print(str(args))
##################################### Data partitioning section 
print('-'*40)
if args.dataset == 'pmnist':
    print('Loading PMNIST for all clients')
    net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, \
    clients_permdict, perm_data = gen_pmnist(args.datadir)
    
    clients_trainds, clients_testds, clients_traindl, clients_testdl, glob_testds, \
    glob_testdl = gen_dataloader(net_dataidx_map, net_dataidx_map_test, clients_permdict, perm_data, args)
elif args.dataset == 'pcifar10':
    print('Loading PCIFAR10 for all clients')
    net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, \
    clients_permdict, perm_data = gen_pcifar10(args.datadir)
    
    clients_trainds, clients_testds, clients_traindl, clients_testdl, glob_testds, \
    glob_testdl = gen_dataloader(net_dataidx_map, net_dataidx_map_test, clients_permdict, perm_data, args)
elif args.dataset == 'cifar10_superclass':
    print('Loading CIFAR10 SuperClass for all clients')
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader('cifar10',
                                                                                   args.datadir,
                                                                                   args.batch_size,
                                                                                   32)
    
    net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
    = CIFAR10_SuperClass(train_ds_global, test_ds_global, args)
elif args.dataset == 'cifar10_superclass_niid':
    print('Loading CIFAR10 SuperClass NIID for all clients')
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader('cifar10',
                                                                                   args.datadir,
                                                                                   args.batch_size,
                                                                                   32)
    
    net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
    = CIFAR10_SuperClass_NIID(train_ds_global, test_ds_global, args) 
elif args.dataset == 'cifar10_superclass_niid_dir':
    print('Loading CIFAR10 SuperClass NIID DIR for all clients')
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader('cifar10',
                                                                                   args.datadir,
                                                                                   args.batch_size,
                                                                                   32)
    
    net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
    = CIFAR10_SuperClass_NIID_DIR(train_ds_global, test_ds_global, args)  
elif args.dataset == 'cifar100_superclass':
    print('Loading CIFAR100 SuperClass for all clients')
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader('cifar100',
                                                                                   args.datadir,
                                                                                   args.batch_size,
                                                                                   32)
    
    net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
    = CIFAR100_SuperClass(train_ds_global, test_ds_global, args)
elif args.dataset == 'cifar100_superclass_niid':
    print('Loading CIFAR100 SuperClass NIID for all clients')
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader('cifar100',
                                                                                   args.datadir,
                                                                                   args.batch_size,
                                                                                   32)
    
    net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
    = CIFAR100_SuperClass_NIID(train_ds_global, test_ds_global, args)
elif args.dataset == 'fmnist_superclass':
    print('Loading FMNIST SuperClass for all clients')
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader('fmnist',
                                                                                   args.datadir,
                                                                                   args.batch_size,
                                                                                   32)
    
    net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
    = FMNIST_SuperClass(train_ds_global, test_ds_global, args)
elif args.dataset == 'fmnist_superclass_niid':
    print('Loading FMNIST SuperClass NIID for all clients')
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader('fmnist',
                                                                                   args.datadir,
                                                                                   args.batch_size,
                                                                                   32)
    
    net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts \
    = FMNIST_SuperClass_NIID(train_ds_global, test_ds_global, args)
print('-'*40)
################################### build model
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
            if args.dataset in ("cifar10", "cinic10", "svhn", 'pcifar10', 'cifar10_superclass',
                               'cifar10_superclass_niid', 'cifar10_superclass_niid_dir'):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(args.device)
            elif args.dataset in ("mnist", 'femnist', 'fmnist', 'pmnist', 'fmnist_superclass',
                                 'fmnist_superclass_niid'):
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
            if args.dataset in ['cifar100', 'cifar100_superclass', 'cifar100_superclass_niid']: 
                net = ResNet9(in_channels=3, num_classes=100)
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

#     model_meta_data = []
#     layer_type = []
#     for (k, v) in nets[0].state_dict().items():
#         model_meta_data.append(v.shape)
#         layer_type.append(k)

    return users_model, net_glob, initial_state_dict, server_state_dict

print('-'*40)
print('Building models for clients')
print(f'MODEL: {args.model}, Dataset: {args.dataset}')
users_model, net_glob, initial_state_dict, server_state_dict = init_nets(args, dropout_p=0.5)
print('-'*40)
print(net_glob)
print('')
total = 0 
for name, param in net_glob.named_parameters():
    print(name, param.size())
    total += np.prod(param.size())
    #print(np.array(param.data.cpu().numpy().reshape([-1])))
    #print(isinstance(param.data.cpu().numpy(), np.array))
print(f'total params {total}')
print('-'*40)
################################# Fixing all to the same Init and data partitioning and random users 
#print(os.getcwd())

# tt = '../initialization/' + 'traindata_'+args.dataset+'_'+args.partition+'.pkl'
# with open(tt, 'rb') as f:
#     net_dataidx_map = pickle.load(f)
    
# tt = '../initialization/' + 'testdata_'+args.dataset+'_'+args.partition+'.pkl'
# with open(tt, 'rb') as f:
#     net_dataidx_map_test = pickle.load(f)
    
# tt = '../initialization/' + 'traindata_cls_counts_'+args.dataset+'_'+args.partition+'.pkl'
# with open(tt, 'rb') as f:
#     traindata_cls_counts = pickle.load(f)
    
# tt = '../initialization/' + 'testdata_cls_counts_'+args.dataset+'_'+args.partition+'.pkl'
# with open(tt, 'rb') as f:
#     testdata_cls_counts = pickle.load(f)

# tt = '../initialization/' + 'init_'+args.model+'_'+args.dataset+'.pth'
# initial_state_dict = torch.load(tt, map_location=args.device)

# server_state_dict = copy.deepcopy(initial_state_dict)
# for idx in range(args.num_users):
#     users_model[idx].load_state_dict(initial_state_dict)
    
# net_glob.load_state_dict(initial_state_dict)

# tt = '../initialization/' + 'comm_users.pkl'
# with open(tt, 'rb') as f:
#     comm_users = pickle.load(f)
    
################################# Initializing Clients 
print('-'*40)
print('Initializing Clients')
clients = []
for idx in range(args.num_users):
    sys.stdout.flush()
    print(f'-- Client {idx}, Labels Stat {traindata_cls_counts[idx]}')
    if args.dataset in ['pmnist', 'pcifar10']:
        train_dl_local = clients_traindl[idx]
        test_dl_local = clients_testdl[idx]
    elif args.dataset in ['cifar10_superclass', 'cifar10_superclass_niid', 'cifar10_superclass_niid_dir']: 
        noise_level=0
        dataidxs = net_dataidx_map[idx]
        dataidxs_test = net_dataidx_map_test[idx]
        
        train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader('cifar10', 
                                                                       args.datadir, args.local_bs, 32, 
                                                                       dataidxs, noise_level, 
                                                                       dataidxs_test=dataidxs_test)
    elif args.dataset in ['cifar100_superclass', 'cifar100_superclass_niid']: 
        noise_level=0
        dataidxs = net_dataidx_map[idx]
        dataidxs_test = net_dataidx_map_test[idx]
        
        train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader('cifar100', 
                                                                       args.datadir, args.local_bs, 32, 
                                                                       dataidxs, noise_level, 
                                                                       dataidxs_test=dataidxs_test)
    elif args.dataset in ['fmnist_superclass', 'fmnist_superclass_niid']: 
        noise_level=0
        dataidxs = net_dataidx_map[idx]
        dataidxs_test = net_dataidx_map_test[idx]
        
        train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader('fmnist', 
                                                                       args.datadir, args.local_bs, 32, 
                                                                       dataidxs, noise_level, 
                                                                       dataidxs_test=dataidxs_test)
        
    clients.append(Client_FedAvg(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep, 
               args.lr, args.momentum, args.device, train_dl_local, test_dl_local))

print('-'*40)
###################################### Federation 
print('Starting SOLO')
print('-'*40)
start = time.time()

clients_local_acc = {i:[] for i in range(args.num_users)}

for idx in range(args.num_users):
    print(f'Client {idx} is training...')
    sys.stdout.flush()
    for epoch in range(args.rounds):
        loss = clients[idx].train(is_print=False)
        
        if epoch in [int(0.5*args.rounds), int(0.8*args.rounds)]:
            _, acc = clients[idx].eval_test()
            clients_local_acc[idx].append(acc)
    
    _, acc = clients[idx].eval_test()
    clients_local_acc[idx].append(acc)
                
    template = ("Client {:3d}, labels {}, final_acc {:3.3f}, best_acc {:3.3f} \n")
    print(template.format(idx, traindata_cls_counts[idx], clients_local_acc[idx][-1], np.max(clients_local_acc[idx])))

end = time.time()
duration = end-start
print('-'*40)
############################### Printing Final Test and Train ACC / LOSS
print('-'*40)
final_acc = []
best_acc = []
for idx in range(args.num_users):
    final_acc.append(clients_local_acc[idx][-1])
    best_acc.append(np.max(clients_local_acc[idx]))
    
avg_final_acc = np.mean(final_acc)
avg_best_acc = np.mean(best_acc)
print(f'Avg Final Acc: {avg_final_acc:.2f}, Avg Best Acc: {avg_best_acc:.2f}')
print(f'SOLO Time: {duration:.2f}')
############################# Saving Print Results 
with open(path+str(args.trial)+'_final_results.txt', 'a') as text_file:
    print(f'Avg Final Acc: {avg_final_acc:.2f}, Avg Best Acc: {avg_best_acc:.2f}', file=text_file)
    print(f'SOLO Time: {duration:.2f}', file=text_file)

print('-'*40)