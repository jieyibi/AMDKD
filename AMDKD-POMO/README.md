# AMDKD-POMO

This is the code for [AMDKD](https://arxiv.org/abs/2210.07686) implemented on [POMO](https://github.com/yd-kwon/POMO).

We provide codes for two combinatorial optimization problems:

- Traveling Salesman Problem (TSP)
- Capacitated Vehicle Routing Problem (CVRP)

## Usage

### Training

To train a AMDKD model:

    python distill.py

you can modify the parameters in _distill.py_. We set N=50 here.


### Evaluation

To test a AMDKD model:

    python test.py

You can specify the model as a parameter contained in *test.py*. We use the saved model (N=50) we have provided (in _pretrained_ folder), which you can easily switch to the other model you trained.


### Dependencies

- Python>=3.8
- matplotlib==3.5.1
- numpy==1.22.3
- pytz==2022.1
- rpy2==3.5.1
- tensorboard_logger==0.1.0
- torch==1.11.0+cu113


## Acknowledgements

This code is originally implemented based on [POMO](https://github.com/yd-kwon/POMO), which is source code of the paper _POMO: Policy Optimization with Multiple Optima for Reinforcement Learning_ accepted at NeurIPS 2020. We use its "NEW_py_ver" version in our paper.

