import numpy as np

import copy
import os 
import gc 
import pickle
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
    
if args.partition != '':
    path = args.savedir + args.alg +'/' + args.dataset + '/' + args.partition + '/'
else: 
    path = args.savedir + args.alg +'/' + args.dataset + '/'
    
path = args.savedir + args.alg +'/' + 'cifar10_niid2' + '/'
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
    print('Loading CIFAR10 SuperClass NIID Dir for all clients')
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
else:
    print(f'Loading {args.dataset}, {args.partition} for all clients')
    args.local_view = True
    X_train, y_train, X_test, y_test, net_dataidx_map, net_dataidx_map_test, \
    traindata_cls_counts, testdata_cls_counts = partition_data(args.dataset, 
    args.datadir, args.logdir, args.partition, args.num_users, beta=args.beta, local_view=args.local_view)

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                       args.datadir,
                                                                                       args.batch_size,
                                                                                       32)
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

#tt = '../initialization/' + 'init_'+args.model+'_'+args.dataset+'.pth'
tt = '../initialization/init_simple-cnn_cifar10_finetune.pth'
initial_state_dict = torch.load(tt, map_location=args.device)
net_glob.load_state_dict(initial_state_dict)

server_state_dict = copy.deepcopy(initial_state_dict)
for idx in range(args.num_users):
    users_model[idx].load_state_dict(initial_state_dict)
    
#net_glob.load_state_dict(initial_state_dict)

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
    else:
        noise_level=0
        dataidxs = net_dataidx_map[idx]
        dataidxs_test = net_dataidx_map_test[idx]
        
        train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, 
                                                                       args.datadir, args.local_bs, 32, 
                                                                       dataidxs, noise_level, 
                                                                       dataidxs_test=dataidxs_test)
        
    clients.append(Client_FedAvg(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep, 
               args.lr, args.momentum, args.device, train_dl_local, test_dl_local))

print('-'*40)
###################################### Federation 
print('Starting FL')
print('-'*40)
start = time.time()

loss_train = []
clients_local_acc = {i:[] for i in range(args.num_users)}
w_locals, loss_locals = [], []
glob_acc = []
w_div = []
w_div_layer = []
w_div_glob=[]
wg_wg_angle=[]

w_glob = copy.deepcopy(initial_state_dict)
for iteration in range(args.rounds):
        
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    #idxs_users = comm_users[iteration]
    
    print(f'----- ROUND {iteration+1} -----')   
    sys.stdout.flush()
    for idx in idxs_users:
        clients[idx].set_state_dict(copy.deepcopy(w_glob)) 
            
        loss = clients[idx].train(is_print=False)
        loss_locals.append(copy.deepcopy(loss))
          
    # print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    template = '-- Average Train loss {:.3f}'
    print(template.format(loss_avg))
    
    ####### Weight Div ####### 
    w_glob_old = copy.deepcopy(w_glob)
    
    wg_vec=[]
    for key, val in w_glob.items():
        wg_vec.append(val.view(-1).detach().cpu().numpy())
    wg_vec = np.hstack(wg_vec)
    
    w_diff_layer={}
    for key, val in w_glob.items():
        w_diff_layer[key] = []
        
    w_diff = []
    for idx in idxs_users:
        wc = copy.deepcopy(clients[idx].get_state_dict())
        wc_vec=[]
        for key, val in wc.items():
            wg_tmp = w_glob[key].view(-1).detach().cpu().numpy()
            wc_tmp = val.view(-1).detach().cpu().numpy()
            
            tt = np.linalg.norm(wg_tmp-wc_tmp, ord=2)
            w_diff_layer[key].append(tt)
            
            wc_vec.append(wc_tmp)
        wc_vec = np.hstack(wc_vec)
        w_diff.append(np.linalg.norm(wg_vec-wc_vec, ord=2))
    
    w_div.append(np.mean(w_diff))
    template = "-- Weight Divergence: {:.3f}"
    print(template.format(w_div[-1]))
    
    w_div_layer_tmp=[]
    for key, val in w_glob.items():
        w_diff_layer[key] = np.mean(w_diff_layer[key])
        w_div_layer_tmp.append(w_diff_layer[key])
        template = "-- Weight Divergence Layer {}: {:.3f}"
        print(template.format(key, w_div_layer_tmp[-1]))
        
    w_div_layer.append(w_div_layer_tmp)
    ####### FedAvg ####### START
    total_data_points = sum([len(net_dataidx_map[r]) for r in idxs_users])
    fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in idxs_users]
    w_locals = []
    for idx in idxs_users:
        w_locals.append(copy.deepcopy(clients[idx].get_state_dict()))

    ww = FedAvg(w_locals, weight_avg=fed_avg_freqs)
    w_glob = copy.deepcopy(ww)
    net_glob.load_state_dict(copy.deepcopy(ww))
    
    wg_vec_new=[]
    wg_wg_tmp_angle=[]
    for key, val in w_glob.items():
        wg_old_tmp = w_glob_old[key].view(-1).detach().cpu().numpy()
        wg_new_tmp = val.view(-1).detach().cpu().numpy()
        
        wg_wg=(wg_new_tmp.T@wg_old_tmp)/(np.linalg.norm(wg_new_tmp, ord=2)*np.linalg.norm(wg_old_tmp, ord=2))
        tt = 180/np.pi*np.arccos(wg_wg)
        wg_wg_tmp_angle.append(tt)
        template = "-- Angle Layer {}: {:.3f}"
        print(template.format(key, wg_wg_tmp_angle[-1]))
        
        wg_vec_new.append(wg_new_tmp)
    wg_vec_new = np.hstack(wg_vec_new)
   
    wg_wg_angle.append(wg_wg_tmp_angle)
    w_div_glob.append(np.linalg.norm(wg_vec_new-wg_vec, ord=2))
    template = "-- AVG Angle Glob: {:.3f}"
    print(template.format(np.mean(wg_wg_angle[-1])))
    
    template = "-- Weight Divergence Glob: {:.3f}"
    print(template.format(w_div_glob[-1]))
    ####### FedAvg ####### END
    _, acc = eval_test(net_glob, args, test_dl_global)
    
    glob_acc.append(acc)
    template = "-- Global Acc: {:.3f}, Global Best Acc: {:.3f}"
    print(template.format(glob_acc[-1], np.max(glob_acc)))
    
    print_flag = False
    if iteration+1 in [int(0.10*args.rounds), int(0.25*args.rounds), int(0.5*args.rounds), int(0.8*args.rounds)]: 
        print_flag = True
    
    if print_flag:
        print('*'*25)
        print(f'Check Point @ Round {iteration+1} --------- {int((iteration+1)/args.rounds*100)}% Completed')
        temp_acc = []
        temp_best_acc = []
        for k in range(args.num_users):
            sys.stdout.flush()
            loss, acc = clients[k].eval_test() 
            clients_local_acc[k].append(acc)
            temp_acc.append(clients_local_acc[k][-1])
            temp_best_acc.append(np.max(clients_local_acc[k]))
                
            template = ("Client {:3d}, current_acc {:3.2f}, best_acc {:3.2f}")
            print(template.format(k, clients_local_acc[k][-1], np.max(clients_local_acc[k])))
        
        #print('*'*25)
        template = ("-- Avg Local Acc: {:3.2f}")
        print(template.format(np.mean(temp_acc)))
        template = ("-- Avg Best Local Acc: {:3.2f}")
        print(template.format(np.mean(temp_best_acc)))
        print('*'*25)
           
    loss_train.append(loss_avg)

    ## clear the placeholders for the next round 
    loss_locals.clear()
    
    ## calling garbage collector 
    gc.collect()
    
end = time.time()
duration = end-start
print('-'*40)
############################### Testing Local Results 
print('*'*25)
print('---- Testing Final Local Results ----')
temp_acc = []
temp_best_acc = []
for k in range(args.num_users):
    sys.stdout.flush()
    loss, acc = clients[k].eval_test() 
    clients_local_acc[k].append(acc)
    temp_acc.append(clients_local_acc[k][-1])
    temp_best_acc.append(np.max(clients_local_acc[k]))

    template = ("Client {:3d}, Final_acc {:3.2f}, best_acc {:3.2f} \n")
    print(template.format(k, clients_local_acc[k][-1], np.max(clients_local_acc[k])))

template = ("-- Avg Local Acc: {:3.2f}")
print(template.format(np.mean(temp_acc)))
template = ("-- Avg Best Local Acc: {:3.2f}")
print(template.format(np.mean(temp_best_acc)))
print('*'*25)
############################### FedAvg Final Results
print('-'*40)
print('FINAL RESULTS')
template = "-- Global Acc: {:.3f}" 
print(template.format(glob_acc[-1]))

template = "-- Global Best Acc: {:.3f}"
print(template.format(np.max(glob_acc)))

template = ("-- Avg Local Acc: {:3.2f}")
print(template.format(np.mean(temp_acc)))

template = ("-- Avg Best Local Acc: {:3.2f}")
print(template.format(np.mean(temp_best_acc)))

print(f'-- FL Time: {duration:.2f}')
print('-'*40)
############################### FedAvg+ (FedAvg + FineTuning)
print('-'*40)
print('FedAvg+ ::: FedAvg + Local FineTuning')
sys.stdout.flush()

local_acc = []
for idx in range(args.num_users): 
    clients[idx].set_state_dict(copy.deepcopy(w_glob)) 
    loss = clients[idx].train(is_print=False)
    _, acc = clients[idx].eval_test()
    local_acc.append(acc)
        
print(f'-- FedAvg+ :: AVG Local Acc: {np.mean(local_acc):.2f}')
############################# Saving Print Results 
with open(path+str(args.trial)+'_final_results.txt', 'a') as text_file:
    template = "-- Global Acc: {:.3f}" 
    print(template.format(glob_acc[-1], file=text_file))

    template = "-- Global Best Acc: {:.3f}"
    print(template.format(np.max(glob_acc)), file=text_file)
          
    template = ("-- Avg Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_acc)), file=text_file)

    template = ("-- Avg Best Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_best_acc)), file=text_file)
    
    print(f'-- FedAvg+ :: Clients AVG Local Acc: {np.mean(local_acc):.2f}', file=text_file)
    print(f'FL Time: {duration:.2f}', file=text_file)

with open(path+str(args.trial)+'_w_div.npy', 'wb') as f:
    w_div = np.array(w_div)
    np.save(f, w_div)
    w_div_layer = np.array(w_div_layer)
    np.save(f, w_div_layer)
    w_div_glob = np.array(w_div_glob)
    np.save(f, w_div_glob)
    wg_wg_angle = np.array(wg_wg_angle)
    np.save(f, wg_wg_angle)

print('-'*40)