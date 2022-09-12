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

def main_fedavg(args):
    
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
            #template = "-- Weight Divergence Layer {}: {:.3f}"
            #print(template.format(key, w_div_layer_tmp[-1]))

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
            #template = "-- Angle Layer {}: {:.3f}"
            #print(template.format(key, wg_wg_tmp_angle[-1]))

            wg_vec_new.append(wg_new_tmp)
        wg_vec_new = np.hstack(wg_vec_new)

        wg_wg_angle.append(wg_wg_tmp_angle)
        w_div_glob.append(np.linalg.norm(wg_vec_new-wg_vec, ord=2))
        #template = "-- AVG Angle Glob: {:.3f}"
        #print(template.format(np.mean(wg_wg_angle[-1])))

        template = "-- Weight Divergence Glob: {:.3f}"
        print(template.format(w_div_glob[-1]))
        ####### FedAvg ####### END
        _, acc = eval_test(net_glob, args, test_dl_global)

        glob_acc.append(acc)
        template = "-- Global Acc: {:.3f}, Global Best Acc: {:.3f}"
        print(template.format(glob_acc[-1], np.max(glob_acc)))

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
    template = "-- Global Acc Final: {:.3f}" 
    print(template.format(glob_acc[-1]))

    template = "-- Global Acc Avg Final 10 Rounds: {:.3f}" 
    print(template.format(np.mean(glob_acc[-10:])))

    template = "-- Global Best Acc: {:.3f}"
    print(template.format(np.max(glob_acc)))

    template = ("-- Avg Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_acc)))

    template = ("-- Avg Best Local Acc: {:3.2f}")
    print(template.format(np.mean(temp_best_acc)))

    print(f'-- FL Time: {duration/60:.2f} minutes')
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
    
    fedavg_ft_local = np.mean(local_acc) 
    print(f'-- FedAvg+ :: AVG Local Acc: {np.mean(local_acc):.2f}')
    ############################# Saving Print Results 

    # with open(path+str(args.trial)+'_w_div.npy', 'wb') as f:
    #     w_div = np.array(w_div)
    #     np.save(f, w_div)
    #     w_div_layer = np.array(w_div_layer)
    #     np.save(f, w_div_layer)
    #     w_div_glob = np.array(w_div_glob)
    #     np.save(f, w_div_glob)
    #     wg_wg_angle = np.array(wg_wg_angle)
    #     np.save(f, wg_wg_angle)
    
    final_glob = glob_acc[-1]
    avg_final_glob = np.mean(glob_acc[-10:])
    best_glob = np.max(glob_acc)
    avg_final_local = np.mean(temp_acc)
    avg_best_local = np.mean(temp_best_acc)
    
    return final_glob, avg_final_glob, best_glob, avg_final_local, avg_best_local, fedavg_ft_local, duration

def run_fedavg(args, fname):
    alg_name = 'FedAvg'
    
    exp_final_glob=[]
    exp_avg_final_glob=[]
    exp_best_glob=[]
    exp_avg_final_local=[]
    exp_avg_best_local=[]
    exp_fedavg_ft_local=[]
    exp_fl_time=[]
    
    for trial in range(args.ntrials):
        print('*'*40)
        print(' '*20, alg_name)
        print(' '*20, 'Trial %d'%(trial+1))
        
        final_glob, avg_final_glob, best_glob, avg_final_local, avg_best_local, \
        fedavg_ft_local, duration = main_fedavg(args)
        
        exp_final_glob.append(final_glob)
        exp_avg_final_glob.append(avg_final_glob)
        exp_best_glob.append(best_glob)
        exp_avg_final_local.append(avg_final_local)
        exp_avg_best_local.append(avg_best_local)
        exp_fedavg_ft_local.append(fedavg_ft_local)
        exp_fl_time.append(duration/60)
        
        print('*'*40)
        print(' '*20, 'End of Trial %d'%(trial+1))
        print(' '*20, 'Final Results')
        
        template = "-- Global Final Acc: {:.3f}" 
        print(template.format(exp_final_glob[-1]))

        template = "-- Global Avg Final 10 Rounds Acc : {:.3f}" 
        print(template.format(exp_avg_final_glob[-1]))

        template = "-- Global Best Acc: {:.3f}"
        print(template.format(exp_best_glob[-1]))

        template = ("-- Avg Final Local Acc: {:3.2f}")
        print(template.format(exp_avg_final_local[-1]))

        template = ("-- Avg Best Local Acc: {:3.2f}")
        print(template.format(exp_avg_best_local[-1]))

        print(f'-- FedAvg+ Fine Tuning Clients AVG Local Acc: {exp_fedavg_ft_local[-1]:.2f}')
        print(f'-- FL Time: {exp_fl_time[-1]:.2f} minutes')
        
    print('*'*40)
    print(' '*20, alg_name)
    print(' '*20, 'Avg %d Trial Results'%args.ntrials)
    
    template = "-- Global Final Acc: {:.3f} +- {:.3f}" 
    print(template.format(np.mean(exp_final_glob), np.std(exp_final_glob)))

    template = "-- Global Avg Final 10 Rounds Acc: {:.3f} +- {:.3f}" 
    print(template.format(np.mean(exp_avg_final_glob), np.std(exp_avg_final_glob)))

    template = "-- Global Best Acc: {:.3f} +- {:.3f}"
    print(template.format(np.mean(exp_best_glob), np.std(exp_best_glob)))

    template = ("-- Avg Final Local Acc: {:3.2f} +- {:.3f}")
    print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)))

    template = ("-- Avg Best Local Acc: {:3.2f} +- {:.3f}")
    print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)))

    template = '-- FedAvg+ Fine Tuning Clients AVG Local Acc: {:.2f} +- {:.2f}'
    print(template.format(np.mean(exp_fedavg_ft_local), np.std(exp_fedavg_ft_local)))
    
    print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes')
    
    with open(fname+'_results_summary.txt', 'a') as text_file:
        print('*'*40, file=text_file)
        print(' '*20, alg_name, file=text_file)
        print(' '*20, 'Avg %d Trial Results'%args.ntrials, file=text_file)
    
        template = "-- Global Final Acc: {:.3f} +- {:.3f}" 
        print(template.format(np.mean(exp_final_glob), np.std(exp_final_glob)), file=text_file)

        template = "-- Global Avg Final 10 Rounds Acc: {:.3f} +- {:.3f}" 
        print(template.format(np.mean(exp_avg_final_glob), np.std(exp_avg_final_glob)), file=text_file)

        template = "-- Global Best Acc: {:.3f} +- {:.3f}"
        print(template.format(np.mean(exp_best_glob), np.std(exp_best_glob)), file=text_file)

        template = ("-- Avg Final Local Acc: {:3.2f} +- {:.3f}")
        print(template.format(np.mean(exp_avg_final_local), np.std(exp_avg_final_local)), file=text_file)

        template = ("-- Avg Best Local Acc: {:3.2f} +- {:.3f}")
        print(template.format(np.mean(exp_avg_best_local), np.std(exp_avg_best_local)), file=text_file)

        template = '-- FedAvg+ Fine Tuning Clients AVG Local Acc: {:.2f} +- {:.2f}'
        print(template.format(np.mean(exp_fedavg_ft_local), np.std(exp_fedavg_ft_local)), file=text_file)
        
        print(f'-- FL Time: {np.mean(exp_fl_time):.2f} minutes', file=text_file)
        print('*'*40)
        
    return