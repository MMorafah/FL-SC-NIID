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

def main_cfl(args):
    
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

        clients.append(Client_CFL(idx, copy.deepcopy(users_model[idx]), args.local_bs, args.local_ep, 
                   args.lr, args.momentum, args.device, train_dl_local, test_dl_local))

    print('-'*40)
    ###################################### Federation 
    print('Starting FL')
    print('-'*40)
    start = time.time()
    
    EPS_1 = 0.4
    EPS_2 = 1.6

    loss_train = []
    clients_local_acc = {i:[] for i in range(args.num_users)}
    w_locals, loss_locals = [], []
    glob_acc = []
    
    cluster_indices = [np.arange(args.num_users).astype("int")]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    w_glob = copy.deepcopy(initial_state_dict)
    for iteration in range(args.rounds):
        print(f'----- ROUND {iteration+1} -----')   
        sys.stdout.flush()
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #idxs_users = comm_users[iteration]
        
        for idx in idxs_users:
            clients[idx].set_state_dict(copy.deepcopy(w_glob)) 

            loss = clients[idx].train(is_print=False)
            loss_locals.append(copy.deepcopy(loss))

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        template = '-- Average Train loss {:.3f}'
        print(template.format(loss_avg))

        ####### CFL ###### START
        similarities = compute_pairwise_similarities(clients)
        cluster_indices_new = []
        for idc in cluster_indices:
            max_norm = compute_max_update_norm([clients[i] for i in idc])
            mean_norm = compute_mean_update_norm([clients[i] for i in idc])

            if mean_norm<EPS_1 and max_norm>EPS_2 and len(idc)>2 and iteration>20:
                c1, c2 = cluster_clients(similarities[idc][:,idc]) 
                cluster_indices_new += [c1, c2]
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        aggregate_clusterwise(client_clusters)
        ####### CFL ###### END
        
        print_flag = False
        if iteration+1 in [int(0.10*args.rounds), int(0.25*args.rounds),
                           int(0.5*args.rounds), int(0.8*args.rounds)]: 
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
    
    template = ("-- Avg Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_acc)))

    template = ("-- Avg Best Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_best_acc)))

    print(f'-- FL Time: {duration/60:.2f} minutes')
    print('-'*40)
   
    avg_final_local = np.mean(temp_acc)
    avg_best_local = np.mean(temp_best_acc)
    
    return avg_final_local, avg_best_local, duration

def run_cfl(args, fname):
    alg_name = 'CFL'
    
    exp_avg_final_local=[]
    exp_avg_best_local=[]
    exp_fl_time=[]
    
    for trial in range(args.ntrials):
        print('*'*40)
        print(' '*20, alg_name)
        print(' '*20, 'Trial %d'%(trial+1))
        
        avg_final_local, avg_best_local, duration = main_cfl(args)
        
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