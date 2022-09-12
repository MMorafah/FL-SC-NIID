#!/bin/sh
#SBATCH --time=70:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --job-name=perfedavg_cifar10_superclass_1
#SBATCH --err=results/perfedavg_cifar10_superclass_1.err
#SBATCH --out=results/perfedavg_cifar10_superclass_1.out

#DIR = "save_async/cifar"
#[ ! -d "$DIR" ] && mkdir -p "$DIR"

ml TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
ml matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
ml SciPy-bundle/2019.10-fosscuda-2019b-Python-3.7.4
ml PyTorch/1.8.0-fosscuda-2019b-Python-3.7.4
ml torchvision/0.9.1-fosscuda-2019b-PyTorch-1.8.0
ml scikit-learn/0.21.3-fosscuda-2019b-Python-3.7.4

for trial in 1
do
    python ../main.py \
    --ntrials=3 \
    --rounds=50 \
    --num_users=25 \
    --frac=0.3 \
    --local_ep=10 \
    --local_bs=10 \
    --lr=0.01 \
    --momentum=0.5 \
    --model=resnet9 \
    --dataset=stl10 \
    --partition='sc_niid_dir' \
    --datadir='../../data/' \
    --logdir='../save_results/' \
    --log_filename=$trial \
    --alg='per_fedavg' \
    --beta=0.5 \
    --local_view \
    --noise=0 \
    --gpu=0 \
    --print_freq=10
done 
