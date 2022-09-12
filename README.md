# Rethinking Data Heterogeneity in Federated Learning: Introducing a New Notion and Standard Benchmarks
This repository contains the official code for the paper ``Rethinking Data Heterogeneity in Federated Learning: Introducing a New Notion and Standard Benchmarks''. 

## Usage

We provide scripts to run the algorithms, which are put under `scripts/`. Here is an example to run the script:
```
cd scripts
bash fedavg.sh
```
To run SC-NIID, modify `partition` to `sc_niid_dir`, and `sc_niid2` for Non-IID label dir and Non-IID Label Skew respectively. To run C-NIID, modify `partition` to `sc_old_niid_dir`, and `sc_old_niid2` for Non-IID label dir and Non-IID Label Skew respectively.

The descriptions of parameters are as follows:
| Parameter | Description |
| --------- | ----------- |
| ntrials      | The number of total runs. |
| rounds       | The number of communication rounds per run. |
| num_users    | The number of clients. |
| frac         | The sampling rate of clients for each round. |
| local_ep     | The number of local training epochs. |
| local_bs     | Local batch size. |
| lr           | The learning rate for local models. |
| momentum     | The momentum for the optimizer. |
| model        | Network architecture. Options: `TODO` |
| dataset      | The dataset for training and testing. Options are discussed above. |
| partition    | How datasets are partitioned. Options: `homo`, `noniid-labeldir`, `noniid-#label1` (or 2, 3, ..., which means the fixed number of labels each party owns). |
| datadir      | The path of datasets. |
| logdir       | The path to store logs. |
| log_filename | The folder name for multiple runs. E.g., with `ntrials=3` and `log_filename=$trial`, the logs of 3 runs will be located in 3 folders named `1`, `2`, and `3`. |
| alg          | Federated learning algorithm. Options are discussed above. |
| beta         | The concentration parameter of the Dirichlet distribution for heterogeneous partition. |
| local_view   | If true puts local test set for each client |
| gpu          | The IDs of GPU to use. E.g., `TODO` |
| print_freq   | The frequency to print training logs. E.g., with `print_freq=10`, training logs are displayed every 10 communication rounds. |

