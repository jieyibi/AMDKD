# Learning Generalizable Models for Vehicle Routing Problems via Knowledge Distillation

This is the PyTorch code for the Adaptive Multi-Distribution Knowledge Distillation (AMDKD) scheme implemented on AM and POMO.

AMDKD is a generic scheme for learning more cross-distribution generalizable deep models, which leverages various knowledge from multiple teachers trained on exemplar distributions to yield a light-weight yet generalist student model. It is trained with an adaptive strategy that allows the student to concentrate on difficult distributions, so as to absorb hard-to-master knowledge more effectively. 

For more details, please see our paper [Learning Generalizable Models for Vehicle Routing Problems via Knowledge Distillation](https://arxiv.org/abs/2210.07686) which has been accepted at NeurIPS 2022. If this code is useful for your work, please cite:

```
@inproceedings{
    bi2022learning,
    title={Learning Generalizable Models for Vehicle Routing Problems via Knowledge Distillation},
    author={Bi, Jieyi and Ma, Yining and Wang, Jiahai and Cao, Zhiguang and Chen, Jinbiao and Sun, Yuan and Chee, Yeow Meng},
    booktitle = {Advances in Neural Information Processing Systems},
    year={2022}
}
```
## Paper
![architecture](./architecture.jpg)

## Data 

For the evaluation, please first download the data from [Google Drive](https://drive.google.com/drive/folders/1-Jf1Rj88zPHWoUlj71ssRiX52b6Ex0Q9?usp=sharing) due to the memory constraint of GitHub.

Put the whole directory of _data_ into _AMDKD/_.
 
