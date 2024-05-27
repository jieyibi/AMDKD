# AMDKD-AM

This is the code for [AMDKD](https://arxiv.org/abs/2210.07686) implemented on [Attention Model](https://github.com/wouterkool/attention-learn-to-route).

We provide codes for two combinatorial optimization problems:

- Traveling Salesman Problem (TSP)
- Capacitated Vehicle Routing Problem (CVRP)

## Usage

### Training

For example, training CVRP instances with 20 nodes:

```
python run.py --problem cvrp --graph_size 20 --distillation --distill_distribution
```

_options.py_ contains parameters you can modify.

##### Multiple GPUs
By default, training will happen *on GPU card(id: 0)*.
Set the argument `--CUDA_VISIBLE_ID` to use specific GPUs:

```
python run.py --problem cvrp --graph_size 20 --distillation --distill_distribution --CUDA_VISIBLE_ID 2,3
```

### Evaluation
To evaluate a model, you can use `eval.py`.

##### Greedy

```
python eval.py ../data/vrp/vrp_uniform20_10000.pkl --model pretrained/cvrp_20/epoch-best.pt --decode_strategy greedy --eval_batch_size 256
```

Note: Set the argument `--eval_batch_size` to change the batch size in the case of different GPUs with different memory constraint. We ran our code on the RTX 3090 GPU cards. Meanwhile, set the argument `--CUDA_VISIBLE_ID` to use specific GPUs as you want.

##### Sampling

To report the best of 1280 sampled solutions, use

```
python eval.py ../data/vrp/vrp_uniform20_10000.pkl --model pretrained/cvrp_20/epoch-best.pt --decode_strategy sample --width 1280 --eval_batch_size 256 --CUDA_VISIBLE_ID 1
```

### Other options and help
```
python run.py -h
python eval.py -h
```
## Dependencies

- Python>=3.8
- gurobipy==9.5.2
- matplotlib==3.5.1
- numpy==1.22.3
- ortools==9.4.1874
- scikit_learn==1.1.2
- scipy==1.8.0
- setuptools==58.0.4
- six==1.16.0
- tensorboard_logger==0.1.0
- tensorflow==2.10.0
- torch==1.11.0+cu113
- tqdm==4.62.3

## Acknowledgements

This code is originally implemented based on [Attention Model](https://github.com/wouterkool/attention-learn-to-route), which is source code of the paper _Attention, Learn to Solve Routing Problems!_ accepted at ICLR 2019.
