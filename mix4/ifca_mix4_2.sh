#!/bin/sh
#SBATCH --time=70:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --job-name=ifca_mix4_homo_4clusters_5ep_123
#SBATCH --err=results/ifca_mix4_homo_4clusters_5ep_123.err
#SBATCH --out=results/ifca_mix4_homo_4clusters_5ep_123.out

#DIR = "save_async/cifar"
#[ ! -d "$DIR" ] && mkdir -p "$DIR"

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4
ml PyTorch/1.8.0-fosscuda-2019b-Python-3.7.4
ml torchvision/0.9.1-fosscuda-2019b-PyTorch-1.8.0
ml scikit-learn/0.21.3-fosscuda-2019b-Python-3.7.4

for trial in 3 2 1
do
    dir='../save_results/ifca/homo/mix4'
    if [ ! -e $dir ]; then
    mkdir -p $dir
    fi 
    
    python ./main_ifca_mix4.py --trial=$trial \
    --rounds=50 \
    --num_users=100 \
    --frac=0.1 \
    --local_ep=5 \
    --local_bs=10 \
    --lr=0.01 \
    --momentum=0.5 \
    --model=simple-cnn \
    --dataset=mix4 \
    --datadir='../../data/' \
    --logdir='../../logs/' \
    --savedir='../save_results/' \
    --partition='homo' \
    --alg='ifca' \
    --nclusters=4 \
    --beta=0.1 \
    --local_view \
    --noise=0 \
    --gpu=0 \
    --print_freq=10 \
    2>&1 | tee $dir'/'$trial'_4clusters.txt'

done 
