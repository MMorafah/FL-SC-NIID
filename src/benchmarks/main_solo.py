import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.data import *
from src.models import *
from src.client import * 
from src.clustering import *
from src.utils import * 

def main_solo(args):
    
    path = args.logdir + args.alg +'/' + args.dataset + '/' + args.partition + '/'
    
    mkdirs(path)
    print(' ')
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    #print(str(args))
    ##################################### Data partitioning section 
    print('-'*40)
    print('Getting Clients Data')

    train_dl_global, test_dl_global, train_ds_global, test_ds_global, net_dataidx_map, net_dataidx_map_test, \
    traindata_cls_counts, testdata_cls_counts = get_clients_data(args)

    print('-'*40)
    ################################### build model
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
    #initial_state_dict = torch.load(tt, map_location=args.device)
    #net_glob.load_state_dict(initial_state_dict)

    #server_state_dict = copy.deepcopy(initial_state_dict)
    #for idx in range(args.num_users):
    #    users_model[idx].load_state_dict(initial_state_dict)

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
    print(f'SOLO Time: {duration/60:.2f} minutes')
    print('-'*40)
    
    return avg_final_acc, avg_best_acc, duration

def run_solo(args, fname):
    alg_name = 'SOLO'
    
    exp_avg_final_local=[]
    exp_avg_best_local=[]
    exp_fl_time=[]
    
    for trial in range(args.ntrials):
        print('*'*40)
        print(' '*20, alg_name)
        print(' '*20, 'Trial %d'%(trial+1))
        
        avg_final_local, avg_best_local, duration = main_solo(args)
        
        exp_avg_final_local.append(avg_final_local)
        exp_avg_best_local.append(avg_best_local)
        exp_fl_time.append(duration/60)
        
        print('*'*40)
        print(' '*20, 'End of Trial %d'%(trial+1))
        print(' '*20, 'Final Results')

        template = ("-- Avg Final Local Acc: {:3.2f}")
        print(template.format(exp_avg_final_local[-1]))

        template = ("-- Avg Best Local Acc: {:3.2f}")
        print(template.format(exp_avg_best_local[-1]))

        print(f'-- FL Time: {exp_fl_time[-1]:.2f} minutes')
        
    print('*'*40)
    print(' '*20, alg_name)
    print(' '*20, 'Avg %d Trial Results'%args.ntrials)
    
    template = ("-- Avg Final Local Acc: {:3.2f} +- {:.3f}")
    print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)))

    template = ("-- Avg Best Local Acc: {:3.2f} +- {:.3f}")
    print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)))

    print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes')
    
    with open(fname+'_results_summary.txt', 'a') as text_file:
        print('*'*40, file=text_file)
        print(' '*20, alg_name, file=text_file)
        print(' '*20, 'Avg %d Trial Results'%args.ntrials, file=text_file)
    
        template = ("-- Avg Final Local Acc: {:3.2f} +- {:.3f}")
        print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)), file=text_file)

        template = ("-- Avg Best Local Acc: {:3.2f} +- {:.3f}")
        print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)), file=text_file)

        print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes', file=text_file)
        print('*'*40)
        
    return