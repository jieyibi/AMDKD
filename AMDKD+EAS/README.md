# AMDKD+EAS

This is the code for [AMDKD](https://arxiv.org/abs/2210.07686) equipped with [EAS](https://github.com/ahottung/EAS), aiming to continuously improve its performance during inference (longer _T_ will yield even better performance). For EAS, we adopt its EAS-lay version (_T_=100) for demonstration purpose.

We provide codes for two combinatorial optimization problems:

- Traveling Salesman Problem (TSP)
- Capacitated Vehicle Routing Problem (CVRP)

## Usage

To test the AMDKD-POMO model with EAS strategy, please run the command as follows.

##### TSP

Take TSP-100 as an example:

```
CUDA_VISIBLE_DEVICES=0 python run_search.py -problem TSP -method eas-lay -model_path ../AMDKD-POMO/TSP/POMO/pretrained/checkpoint-tsp-100.pt  -instances_path ../data/tsp/tsp_uniform100_10000.pkl  -max_iter 100 -batch_size 100 -param_lr 0.0032 -param_lambda 0.012
```

##### CVRP

Take CVRP-100 as an example:

```
CUDA_VISIBLE_DEVICES=0 python run_search.py -problem CVRP -method eas-lay -model_path ../AMDKD-POMO/CVRP/POMO/pretrained/checkpoint-cvrp-100.pt -instances_path ../data/vrp/vrp_uniform100_10000.pkl -max_iter 100 -batch_size 100  -param_lr 0.0041 -param_lambda 0.013
```

##### Note

Set the argument `-batch_size` to change the batch size in the case of different GPUs with different memory constraint. We ran our code on the RTX 3090 GPU cards. Set the argument `CUDA_VISIBLE_DEVICES` to use specific GPUs as you want. 

If you want to test the pretrained AMDKD-POMO model on the instances in TSPLIB or CVRPLIB, add the flag `-round_distances` to normalize the initial input coordinates and `-p_runs 10` to set the number of parallel runs per instance as 10 (as suggested in the source code of EAS). If your GPU memory is limited, reduce `-p_runs` to a smaller value.

More details about the parameters, please refer to the source code for [EAS](https://github.com/ahottung/EAS). In this repo, we directly use the parameters of the original EAS, except for the `-max_iter` (_T_) that we limit to 100.

### Dependencies

- numpy==1.20.3
- torch==1.11.0+cu113
- tqdm==4.61.0

## Acknowledgements

This code is originally implemented based on [EAS](https://github.com/ahottung/EAS), which is the source code of the paper _Efficient Active Search for Combinatorial Optimization Problems_ accepted at ICLR 2022. 
