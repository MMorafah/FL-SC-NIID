import numpy as np

import copy
import os 
import gc 
import pickle 

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import sys
sys.path.append("..")

from src.data import *
from src.models import *
from src.fedavg import *
from src.client import * 
from src.clustering import *
from src.utils import * 

args = args_parser()

args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

torch.cuda.set_device(args.gpu) ## Setting cuda on GPU 

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass
    
path = args.savedir + args.alg + '/' + args.partition + '/' + args.dataset + '/'
mkdirs(path)

template = "Algorithm {}, Clients {}, Dataset {}, Model {}, Non-IID {}, Threshold {}, K {}, Linkage {}, LR {}, Ep {}, Rounds {}, bs {}, frac {}"

s = template.format(args.alg, args.num_users, args.dataset, args.model, args.partition, args.cluster_alpha, args.n_basis, args.linkage, args.lr, args.local_ep, args.rounds, args.local_ep, args.frac)

args.dataset = "mix4" 
args.partition = "homo" 
args.model = "simple-cnn" 

print(s)
print(str(args))

##################################### Loading All the 4 Datasets 

train_dl_cifar10, test_dl_cifar10, train_ds_cifar10, test_ds_cifar10 = get_dataloader('cifar10',
                                                                                   args.datadir,
                                                                                   train_bs=args.batch_size,
                                                                                   test_bs=32, same_size=True, 
                                                                                   target_transform=lambda x: int(x+10))

train_dl_svhn, test_dl_svhn, train_ds_svhn, test_ds_svhn = get_dataloader('svhn',
                                                                           args.datadir,
                                                                           train_bs=args.batch_size,
                                                                           test_bs=32, same_size=True,
                                                                         target_transform=lambda x: int(x+20))

train_dl_fmnist, test_dl_fmnist, train_ds_fmnist, test_ds_fmnist = get_dataloader('fmnist',
                                                                                   args.datadir,
                                                                                   train_bs=args.batch_size,
                                                                                   test_bs=32, same_size=True,
                                                                                 target_transform=lambda x: int(x+30))

train_dl_usps, test_dl_usps, train_ds_usps, test_ds_usps = get_dataloader('usps',
                                                                           args.datadir,
                                                                           train_bs=args.batch_size,
                                                                           test_bs=32, same_size=True,
                                                                         target_transform=lambda x: int(x))
# train_ds_usps = torchvision.datasets.USPS(
#     root='../data', train=True, download=True, transform=None)

print(f'CIFAR-10 Train Data {len(train_ds_cifar10)}, Each client 500, shards {int(len(train_ds_cifar10)/500)}')
print(f'SVHN Train Data {len(train_ds_svhn)}, Each client 500, shards {int(len(train_ds_svhn)/500)}')
print(f'FMNIST Train Data {len(train_ds_fmnist)}, Each client 500, shards {int(len(train_ds_fmnist)/500)}')
print(f'USPS Train Data {len(train_ds_usps)}, Each client 500, shards {int(len(train_ds_usps)/500)}')

print(f'CIFAR-10 Test Data {len(test_ds_cifar10)}')
print(f'SVHN Test Data {len(test_ds_svhn)}')
print(f'FMNIST Test Data {len(test_ds_fmnist)}')
print(f'USPS Test Data {len(test_ds_usps)}')
    
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
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10).to(args.device)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10).to(args.device)
            elif args.dataset == 'celeba':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2).to(args.device)
            elif args.dataset == "mix4": 
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=40).to(args.device)
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
            if args.dataset == 'cifar100': 
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

print(f'MODEL: {args.model}, Dataset: {args.dataset}')

users_model, net_glob, initial_state_dict, server_state_dict = init_nets(args, dropout_p=0.5)

print(net_glob)

total = 0 
for name, param in net_glob.named_parameters():
    print(name, param.size())
    total += np.prod(param.size())
    #print(np.array(param.data.cpu().numpy().reshape([-1])))
    #print(isinstance(param.data.cpu().numpy(), np.array))
print(total)

################################# Fixing all to the same Init and data partitioning and random users 
#print(os.getcwd())

tt = '../initialization/' + 'traindata_'+args.dataset+'_'+args.partition+'.pkl'
with open(tt, 'rb') as f:
    net_dataidx_map = pickle.load(f)
    
# tt = '../initialization/' + 'testdata_'+args.dataset+'_'+args.partition+'.pkl'
# with open(tt, 'rb') as f:
#     net_dataidx_map_test = pickle.load(f)
    
tt = '../initialization/' + 'traindata_cls_counts_'+args.dataset+'_'+args.partition+'.pkl'
with open(tt, 'rb') as f:
    traindata_cls_counts = pickle.load(f)
    
# tt = '../initialization/' + 'testdata_cls_counts_'+args.dataset+'_'+args.partition+'.pkl'
# with open(tt, 'rb') as f:
#     testdata_cls_counts = pickle.load(f)

tt = '../initialization/' + 'init_'+args.model+'_'+args.dataset+'.pth'
initial_state_dict = torch.load(tt, map_location=args.device)

server_state_dict = copy.deepcopy(initial_state_dict)
for idx in range(args.num_users):
    users_model[idx].load_state_dict(initial_state_dict)
    
net_glob.load_state_dict(initial_state_dict)

tt = '../initialization/' + 'comm_users.pkl'
with open(tt, 'rb') as f:
    comm_users = pickle.load(f)
    
################################# Initializing Clients 
clients_dataset = {}
for i in range(args.num_users): 
    if  0<=i <=13: 
        clients_dataset[i] = 'usps'
    elif 14<=i<=45: 
        clients_dataset[i] = 'cifar10'
    elif 46<=i<=71: 
        clients_dataset[i] = 'svhn'
    elif 72<=i<=99: 
        clients_dataset[i] = 'fmnist'

datasets_order = {0:'USPS', 1:'CIFAR-10', 2:'SVHN', 3:'FMNIST'}

clients = []
    
for idx in range(args.num_users):
    
    dataset_name = clients_dataset[idx]
    dataidxs_test = None
    
    dataidxs = net_dataidx_map[idx]
        
    if dataset_name == 'usps':
        target_transform=lambda x: int(x) 
    elif dataset_name == 'cifar10':
        target_transform=lambda x: int(x+10) 
    elif dataset_name == 'svhn': 
        target_transform=lambda x: int(x+20) 
    elif dataset_name == 'fmnist': 
        target_transform=lambda x: int(x+30) 
          
    print(f'Initializing Client {idx}, {dataset_name}')

    noise_level = args.noise
    if idx == args.num_users - 1:
        noise_level = 0

    if args.noise_type == 'space':
        train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(dataset_name, 
                                                                       args.datadir, args.local_bs, 32, 
                                                                       dataidxs, noise_level, idx, 
                                                                       args.num_users-1, 
                                                                       dataidxs_test=dataidxs_test,
                                                                       same_size=True,
                                                                       target_transform=target_transform)
    else:
        noise_level = args.noise / (args.num_users - 1) * idx
        train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(dataset_name, 
                                                                       args.datadir, args.local_bs, 32, 
                                                                       dataidxs, noise_level, 
                                                                       dataidxs_test=dataidxs_test, 
                                                                       same_size=True, 
                                                                       target_transform=target_transform)
    
    clients.append(Client_ClusterFL(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep, 
               args.lr, args.momentum, args.device, train_dl_local, test_dl_local))
    
###################################### Federation 

float_formatter = "{:.4f}".format
#np.set_printoptions(formatter={float: float_formatting_function})
np.set_printoptions(formatter={'float_kind':float_formatter})

clients_best_local_acc = [0 for _ in range(args.num_users)]
clients_best_glob_acc = [0 for _ in range(args.num_users)]

for idx in range(args.num_users):
    print(f'Client {idx} is training...')
           
    for epoch in range(args.rounds):

        loss = clients[idx].train(is_print=False)

        loss, acc = clients[idx].eval_test()

        if acc > clients_best_local_acc[idx]:
            clients_best_local_acc[idx] = copy.deepcopy(acc)
            
    loss, acc = clients[idx].eval_test()

    if acc > clients_best_glob_acc[idx]:
        clients_best_glob_acc[idx] = copy.deepcopy(acc)


    template = ("Client {:3d}, labels {}, best_local_acc {:3.3f}, best_glob_acc {:3.3f} \n")
    print(template.format(idx, traindata_cls_counts[idx], clients_best_local_acc[idx], clients_best_glob_acc[idx]))

        
############################### Printing Final Test and Train ACC / LOSS
test_loss = []
test_acc = []
train_loss = []
train_acc = []

test_loss_glob = []
test_acc_glob = []
for idx in range(args.num_users):        
    loss, acc = clients[idx].eval_test()
        
    test_loss.append(loss)
    test_acc.append(acc)
    
    #loss, acc = clients[idx].eval_test_glob(test_dl_global)
        
    test_loss_glob.append(loss)
    test_acc_glob.append(acc)
    
    loss, acc = clients[idx].eval_train()
    
    train_loss.append(loss)
    train_acc.append(acc)

test_loss = sum(test_loss) / len(test_loss)
test_acc = sum(test_acc) / len(test_acc)

test_loss_glob = sum(test_loss_glob) / len(test_loss_glob)
test_acc_glob = sum(test_acc_glob) / len(test_acc_glob)

train_loss = sum(train_loss) / len(train_loss)
train_acc = sum(train_acc) / len(train_acc)

print(f'Train Loss: {train_loss}, Test_loss: {test_loss}')
print(f'Train Acc: {train_acc}, Test Acc: {test_acc}')
print(f'Test Loss GLob: {test_loss_glob}, Test Acc Glob: {test_acc_glob}')


print(f'Best Clients AVG Acc Local: {np.mean(clients_best_local_acc)}')
print(f'Best Clients AVG Acc Glob: {np.mean(clients_best_glob_acc)}')

############################# Saving Print Results 
with open(path+str(args.trial)+'_final_results.txt', 'a') as text_file:
    print(f'Train Loss: {train_loss}, Test_loss: {test_loss}', file=text_file)
    print(f'Train Acc: {train_acc}, Test Acc: {test_acc}', file=text_file)
    print(f'Test Loss GLob: {test_loss_glob}, Test Acc Glob: {test_acc_glob}', file=text_file)

    print(f'Best Clients AVG Acc Local: {np.mean(clients_best_local_acc)}', file=text_file)
    print(f'Best Clients AVG Acc Glob: {np.mean(clients_best_glob_acc)}', file=text_file)
    