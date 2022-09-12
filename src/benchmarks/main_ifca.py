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

def main_ifca(args):
    
    path = args.logdir + args.alg +'/' + args.dataset + '/' + args.partition + '/'
    mkdirs(path)
    
    NUM_CLUSTER = args.nclusters

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
    
    w_glob_per_cluster = []
    for _ in range(NUM_CLUSTER):
        net_glob.apply(weight_init)
        w_glob_per_cluster.append(copy.deepcopy(net_glob.state_dict()))

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
    print('Starting FL')
    print('-'*40)
    start = time.time()

    loss_train = []
    clients_local_acc = {i:[] for i in range(args.num_users)}
    w_locals, loss_locals = [], []
    glob_acc = {i:[] for i in range(NUM_CLUSTER)}
    
    w_glob = copy.deepcopy(initial_state_dict)
    for iteration in range(args.rounds):
        print(f'----- ROUND {iteration+1} -----')   
        sys.stdout.flush()
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #idxs_users = comm_users[iteration]

        selected_clusters = [[] for _ in range(NUM_CLUSTER)]
        w_locals_clusters = [[] for _ in range(NUM_CLUSTER)]
        assert (NUM_CLUSTER == len(w_glob_per_cluster))

        for idx in idxs_users:
            acc_select = []
            for i in range(NUM_CLUSTER):
                clients[idx].set_state_dict(copy.deepcopy(w_glob_per_cluster[i])) 

                loss, acc = clients[idx].eval_test() 
                acc_select.append(acc)

            idx_cluster = np.argmax(acc_select)
            selected_clusters[idx_cluster].append(idx)
            #print(f'Client {idx}, Select Cluster: {idx_cluster}')
            #print(f'acc clusters: {acc_select}')
            
            clients[idx].set_state_dict(copy.deepcopy(w_glob_per_cluster[idx_cluster])) 

            loss = clients[idx].train(is_print=False)
            loss_locals.append(copy.deepcopy(loss))
            w_locals_clusters[idx_cluster].append(copy.deepcopy(clients[idx].get_state_dict()))

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        template = '-- Average Train loss {:.3f}'
        print(template.format(loss_avg))

       
        ####### FedAvg ####### START
        total_data_points = [sum([len(net_dataidx_map[r]) for r in clust]) for clust in selected_clusters]
        fed_avg_freqs = [[len(net_dataidx_map[r]) / total_data_points[clust_id] for r in selected_clusters[clust_id]] 
                         for clust_id in range(NUM_CLUSTER)]

        for i in range(NUM_CLUSTER):
            if w_locals_clusters[i] != []:
                ww = FedAvg(w_locals_clusters[i], weight_avg = fed_avg_freqs[i])
                w_glob_per_cluster[i] = copy.deepcopy(ww)
                net_glob.load_state_dict(copy.deepcopy(ww))
                _, acc = eval_test(net_glob, args, test_dl_global)
                glob_acc[i].append(acc)
                
                template = "-- Cluster {}, Global Acc: {:.3f}, Global Best Acc: {:.3f}"
                print(template.format(i+1, glob_acc[i][-1], np.max(glob_acc[i])))
            else:
                glob_acc[i].append(0)
                template = "-- Cluster {} Did not participate,Global Acc: {:.3f}, Global Best Acc: {:.3f}"
                print(template.format(i+1, glob_acc[i][-1], np.max(glob_acc[i])))
        
        avg_glob_acc = np.mean([glob_acc[i][-1] for i in range(NUM_CLUSTER)])
        avg_glob_best_acc = np.mean([np.max(glob_acc[i]) for i in range(NUM_CLUSTER)])
        
        template = "-- Avg Clusters, Global Acc: {:.3f}, Global Best Acc: {:.3f}"
        print(template.format(avg_glob_acc, avg_glob_best_acc))
        ####### FedAvg ####### END
        
        print_flag = False
        if iteration+1 in [int(0.10*args.rounds), int(0.5*args.rounds), int(0.8*args.rounds)]: 
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
    
    for i in range(NUM_CLUSTER):
        template = "-- Cluster {}, Global Final Acc: {:.2f}, Global Avg Final 10 Rounds Acc: {:.2f}, Global Best Acc: {:.2f}"
        print(template.format(i+1, glob_acc[i][-1], np.mean(glob_acc[i][-10:]), np.max(glob_acc[i])))

    avg_glob_acc = np.mean([glob_acc[i][-1] for i in range(NUM_CLUSTER)])
    avg_glob_acc_10rounds = np.mean([np.mean([glob_acc[i][-j] for i in range(NUM_CLUSTER)]) for j in range(1, 11)])
    avg_glob_best_acc = np.mean([np.max(glob_acc[i]) for i in range(NUM_CLUSTER)])
    template = "-- Avg Clusters, Global Final Acc: {:.2f}, Global Avg Final 10 Rounds Acc: {:.2f}, Global Best Acc: {:.2f}"
    print(template.format(avg_glob_acc, avg_glob_acc_10rounds, avg_glob_best_acc))

    template = ("-- Avg Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_acc)))

    template = ("-- Avg Best Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_best_acc)))

    print(f'-- FL Time: {duration/60:.2f} minutes')
    print('-'*40)
    ############################### FedAvg+ (FedAvg + FineTuning)
    print('-'*40)
    print('IFCA+ ::: IFCA + Local FineTuning')
    sys.stdout.flush()

    local_acc = []
    for idx in range(args.num_users): 
        clients[idx].set_state_dict(copy.deepcopy(w_glob)) 
        loss = clients[idx].train(is_print=False)
        _, acc = clients[idx].eval_test()
        local_acc.append(acc)
    
    ifca_ft_local = np.mean(local_acc) 
    print(f'-- IFCA+ :: AVG Local Acc: {np.mean(local_acc):.2f}')
    ############################# Saving Print Results 
    
    avg_final_local = np.mean(temp_acc)
    avg_best_local = np.mean(temp_best_acc)
    
    return avg_glob_acc, avg_glob_acc_10rounds, avg_glob_best_acc, avg_final_local, avg_best_local, ifca_ft_local, duration

def run_ifca(args, fname):
    alg_name = 'IFCA'
    
    exp_final_avg_glob=[]
    exp_avg_glob_acc_10rounds=[]
    exp_avg_glob_best=[]
    exp_avg_final_local=[]
    exp_avg_best_local=[]
    exp_ifca_ft_local=[]
    exp_fl_time=[]
    
    for trial in range(args.ntrials):
        print('*'*40)
        print(' '*20, alg_name)
        print(' '*20, 'Trial %d'%(trial+1))
        
        avg_glob_acc, avg_glob_acc_10rounds, avg_glob_best_acc, avg_final_local, avg_best_local, \
        ifca_ft_local, duration = main_ifca(args)
        
        exp_final_avg_glob.append(avg_glob_acc)
        exp_avg_glob_acc_10rounds.append(avg_glob_acc_10rounds)
        exp_avg_glob_best.append(avg_glob_best_acc)
        exp_avg_final_local.append(avg_final_local)
        exp_avg_best_local.append(avg_best_local)
        exp_ifca_ft_local.append(ifca_ft_local)
        exp_fl_time.append(duration/60)
        
        print('*'*40)
        print(' '*20, 'End of Trial %d'%(trial+1))
        print(' '*20, 'Final Results')
        
        template = "-- Avg Clusters, Global Final Acc: {:.2f}"
        print(template.format(exp_final_avg_glob[-1]))
        
        template = "-- Avg Clusters, Global Avg Final 10 Rounds Acc: {:.2f}"
        print(template.format(exp_avg_glob_acc_10rounds[-1]))

        template = "-- Avg Clusters, Global Best Acc: {:.2f}"
        print(template.format(exp_avg_glob_best[-1]))

        template = ("-- Avg Final Local Acc: {:3.2f}")
        print(template.format(exp_avg_final_local[-1]))

        template = ("-- Avg Best Local Acc: {:3.2f}")
        print(template.format(exp_avg_best_local[-1]))

        print(f'-- FedAvg+ Fine Tuning Clients AVG Local Acc: {exp_ifca_ft_local[-1]:.2f}')
        print(f'-- FL Time: {exp_fl_time[-1]:.2f} minutes')
        
    print('*'*40)
    print(' '*20, alg_name)
    print(' '*20, 'Avg %d Trial Results'%args.ntrials)
    
    template = "-- Avg Clusters, Global Final Acc: {:.2f} +- {:.2f}"
    print(template.format(np.mean(exp_final_avg_glob), np.std(exp_final_avg_glob)))

    template = "-- Avg Clusters, Global Avg Final 10 Rounds Acc: {:.2f} +- {:.2f}"
    print(template.format(np.mean(exp_avg_glob_acc_10rounds), np.std(exp_avg_glob_acc_10rounds)))

    template = "-- Avg Clusters, Global Best Acc: {:.2f} +- {:.2f}"
    print(template.format(np.mean(exp_avg_glob_best), np.std(exp_avg_glob_best)))

    template = ("-- Avg Final Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)))

    template = ("-- Avg Best Local Acc: {:3.2f} +- {:.2f}")
    print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)))

    template = '-- IFCA+ Fine Tuning Clients AVG Local Acc: {:3.2f} +- {:.3f}' 
    print(template.format(np.mean(exp_ifca_ft_local), np.std(exp_ifca_ft_local)))
    
    print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes')
    
    with open(fname+'_results_summary.txt', 'a') as text_file:
        print('*'*40, file=text_file)
        print(' '*20, alg_name, file=text_file)
        print(' '*20, 'Avg %d Trial Results'%args.ntrials, file=text_file)
    
        template = "-- Avg Clusters, Global Final Acc: {:.2f} +- {:.2f}"
        print(template.format(np.mean(exp_final_avg_glob), np.std(exp_final_avg_glob)), file=text_file)

        template = "-- Avg Clusters, Global Avg Final 10 Rounds Acc: {:.2f} +- {:.2f}"
        print(template.format(np.mean(exp_avg_glob_acc_10rounds), np.std(exp_avg_glob_acc_10rounds)), file=text_file)

        template = "-- Avg Clusters, Global Best Acc: {:.2f} +- {:.2f}"
        print(template.format(np.mean(exp_avg_glob_best), np.std(exp_avg_glob_best)), file=text_file)

        template = ("-- Avg Final Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)), file=text_file)

        template = ("-- Avg Best Local Acc: {:3.2f} +- {:.2f}")
        print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)), file=text_file)
        
        template = '-- IFCA+ Fine Tuning Clients AVG Local Acc: {:3.2f} +- {:.3f}' 
        print(template.format(np.mean(exp_ifca_ft_local), np.std(exp_ifca_ft_local)), file=text_file)
    
        print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes', file=text_file)
        print('*'*40)
        
    return