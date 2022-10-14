# AMDKD-AM

This is the official PyTorch code for the Adaptive Multi-Distribution Knowledge Distillation (AMDKD) scheme implemented on [Attention Model](https://github.com/wouterkool/attention-learn-to-route).

For more details, please see our paper **Learning Generalizable Models for Vehicle Routing Problems via Knowledge Distillation** which has been accepted at **NeurIPS 2022**. If this code is useful for your work, please cite our paper:

```
@inproceedings{
    bi2022learning,
    title={Learning Generalizable Models for Vehicle Routing Problems via Knowledge Distillation},
    author={Jieyi Bi and Yining Ma and Jiahai Wang and Zhiguang Cao and Jinbiao Chen and Yuan Sun and Yeow Meng Chee},
    booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
    year={2022},
    url={https://openreview.net/forum?id=sOVNpUEgKMp}
}
```
We provide codes for two combinatorial optimization problems:

- Traveling Salesman Problem (TSP)
- Capacitated Vehicle Routing Problem (CVRP)

## Usage

### Training

For example, training CVRP instances with 20 nodes:

```
python run.py --problem cvrp --graph_size 20 --distill_distribution
```

_options.py_ contains parameters you can modify.

##### Multiple GPUs
By default, training will happen *on GPU card(id: 0)*.
Set the argument `--CUDA_VISIBLE_ID` to use specific GPUs:

```
python run.py --problem cvrp --graph_size 20 --distill_distribution --CUDA_VISIBLE_ID 2,3
```

### Evaluation
To evaluate a model, you can use `eval.py`.

#####Greedy

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


## Acknowledgements
This code is originally implemented based on  [Attention Model](https://github.com/wouterkool/attention-learn-to-route) , which is source code of the paper   [Attention, Learn to Solve Routing Problems!](https://openreview.net/forum?id=ByxBFsRqYm) which has been accepted at [ICLR 2019](https://iclr.cc/Conferences/2019), cite as follows:

```
@inproceedings{
    kool2018attention,
    title={Attention, Learn to Solve Routing Problems!},
    author={Wouter Kool and Herke van Hoof and Max Welling},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=ByxBFsRqYm},
}
```
