import numpy as np

import copy
import os 
import gc 
import pickle
import time 
import sys
import datetime

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
from src.benchmarks import *

if __name__ == '__main__':
    print('-'*40)
    
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu) ## Setting cuda on GPU 
    path = args.logdir + args.alg +'/' + args.dataset + '/' + args.partition + '/'
    mkdirs(path)
    
    if args.log_filename is None: 
        filename='logs_%s.txt' % datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    else:
        filename='logs_'+args.log_filename+'.txt'  

    sys.stdout = Logger(fname=path+filename)
    
    fname=path+filename
    fname=fname[0:-4]
    if args.alg == 'solo':
        alg_name = 'SOLO'
        run_solo(args, fname=fname)
    elif args.alg == 'fedavg':
        alg_name = 'FedAvg'
        run_fedavg(args, fname=fname)
    elif args.alg == 'fedprox':
        alg_name = 'FedProx'
        run_fedprox(args, fname=fname)
    elif args.alg == 'fednova':
        alg_name = 'FedNova'
        run_fednova(args, fname=fname)
    elif args.alg == 'scaffold':
        alg_name = 'Scaffold'
        run_scaffold(args, fname=fname)
    elif args.alg == 'lg':
        alg_name = 'LG'
        run_lg(args, fname=fname)
    elif args.alg == 'per_fedavg':
        alg_name = 'Per-FedAvg'
        run_per_fedavg(args, fname=fname)
    elif args.alg == 'ifca':
        alg_name = 'IFCA'
        run_ifca(args, fname=fname)
    elif args.alg == 'cfl':
        alg_name = 'CFL'
        run_cfl(args, fname=fname)
    else: 
        print('Algorithm Does Not Exist')
        sys.exit()
        