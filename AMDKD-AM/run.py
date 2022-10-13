#!/usr/bin/env python

import os
import sys
import json
import pprint as pp
import numpy as np
import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger
from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from knowledge_distillation import train_epoch_distill
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel, student_AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem
from collections import OrderedDict

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def run(opts):

    if not opts.all_cuda_visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = opts.CUDA_VISIBLE_ID
    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    np.random.seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))
        time_logger_path = os.path.join(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name), 'time')
        if not os.path.exists(time_logger_path):
            os.makedirs(time_logger_path)

    os.makedirs(opts.save_dir)

    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # # visible cuda
    # os.environ["CUDA_VISIBLE_DEVICES"] = opts.CUDA_VISIBLE_ID
    # Set the device
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)


    tb_path = os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)

    if not opts.distill_distribution:
    # if not opts.multi_teacher:
        model = model_class(
            opts.embedding_dim,
            opts.hidden_dim,
            problem,
            n_encode_layers=opts.n_encode_layers,
            mask_inner=True,
            mask_logits=True,
            normalization=opts.normalization,
            tanh_clipping=opts.tanh_clipping,
            checkpoint_encoder=opts.checkpoint_encoder,
            shrink_size=opts.shrink_size,
            tb_logger_path = tb_path,
            distill_temp=opts.distill_temperature
        ).to(opts.device)
        if opts.use_cuda and torch.cuda.device_count() > 1 and opts.all_cuda_visible:
            model = torch.nn.DataParallel(model)
        elif  not opts.all_cuda_visible and len(opts.CUDA_VISIBLE_ID) > 1:
            model = torch.nn.DataParallel(model)
    else:
        model_uniform = model_class(
            opts.embedding_dim, opts.hidden_dim, problem,
            n_encode_layers=opts.n_encode_layers, mask_inner=True, mask_logits=True,
            normalization=opts.normalization_uniform, tanh_clipping=opts.tanh_clipping,
            checkpoint_encoder=opts.checkpoint_encoder, shrink_size=opts.shrink_size,
            tb_logger_path = tb_path, distill_temp=opts.distill_temperature
        ).to(opts.device)
        model_cluster = model_class(
            opts.embedding_dim, opts.hidden_dim, problem,
            n_encode_layers=opts.n_encode_layers, mask_inner=True, mask_logits=True,
            normalization=opts.normalization_cluster, tanh_clipping=opts.tanh_clipping,
            checkpoint_encoder=opts.checkpoint_encoder, shrink_size=opts.shrink_size,
            tb_logger_path = tb_path, distill_temp=opts.distill_temperature
        ).to(opts.device)
        model_mixed = model_class(
            opts.embedding_dim, opts.hidden_dim, problem,
            n_encode_layers=opts.n_encode_layers, mask_inner=True, mask_logits=True,
            normalization=opts.normalization_mixed, tanh_clipping=opts.tanh_clipping,
            checkpoint_encoder=opts.checkpoint_encoder, shrink_size=opts.shrink_size,
            tb_logger_path=tb_path, distill_temp=opts.distill_temperature
        ).to(opts.device)

        if opts.use_cuda and torch.cuda.device_count() > 1 and opts.all_cuda_visible:
            model_uniform = torch.nn.DataParallel(model_uniform)
            model_cluster = torch.nn.DataParallel(model_cluster)
            model_mixed = torch.nn.DataParallel(model_mixed)
        elif  not opts.all_cuda_visible and len(opts.CUDA_VISIBLE_ID) > 1:
            model_uniform = torch.nn.DataParallel(model_uniform)
            model_cluster = torch.nn.DataParallel(model_cluster)
            model_mixed = torch.nn.DataParallel(model_mixed)

        model = {'uniform': model_uniform,
                'cluster': model_cluster,
                'mixed': model_mixed}

    if not opts.distillation:
        # Initialize baseline
        if opts.baseline == 'exponential':
            baseline = ExponentialBaseline(opts.exp_beta)
        elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
            assert problem.NAME == 'tsp', "Critic only supported for TSP"
            baseline = CriticBaseline(
                (
                    CriticNetworkLSTM(
                        2,
                        opts.embedding_dim,
                        opts.hidden_dim,
                        opts.n_encode_layers,
                        opts.tanh_clipping
                    )
                    if opts.baseline == 'critic_lstm'
                    else
                    CriticNetwork(
                        2,
                        opts.embedding_dim,
                        opts.hidden_dim,
                        opts.n_encode_layers,
                        opts.normalization
                    )
                ).to(opts.device)
            )
        elif opts.baseline == 'rollout':
            baseline = RolloutBaseline(model, problem, opts)
        else:
            assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
            baseline = NoBaseline()
        if opts.bl_warmup_epochs > 0:
            baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

        # Initialize optimizer
        optimizer = optim.Adam(
            [{'params': model.parameters(), 'lr': opts.lr_model}]
            + (
                [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
                if len(baseline.get_learnable_parameters()) > 0
                else []
            )
        )

    # Overwrite model parameters by parameters to load
    # model_ = get_inner_model(model)
    # model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})
    # Load data from load_path

    if not opts.distill_distribution:
        load_data = {}
        assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
        if opts.is_load or opts.distillation or opts.resume:
            load_path0 = opts.load_path if opts.load_path is not None else opts.resume
            load_path = load_path0
            print('  [*] Loading data from {}'.format(load_path))
            load_data = torch_load_cpu(load_path)
            model_ = get_inner_model(model)
            if opts.is_load_multi:
                state_dict = load_data.get('model', {})
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                model_.load_state_dict({**model_.state_dict(), **new_state_dict})
            else:
                model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

        if not opts.distillation:
            # Load baseline from data, make sure script is called with same type of baseline
            if 'baseline' in load_data:
                baseline.load_state_dict(load_data['baseline'])

            # Load optimizer state
            if 'optimizer' in load_data:
                optimizer.load_state_dict(load_data['optimizer'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        # if isinstance(v, torch.Tensor):
                        if torch.is_tensor(v):
                            state[k] = v.to(opts.device)

        if opts.resume:
            epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
            torch.set_rng_state(load_data['rng_state'])
            if opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
            # Set the random states
            # Dumping of state was done before epoch callback, so do that now (model is loaded)
            baseline.epoch_callback(model, epoch_resume)
            print("Resuming after {}".format(epoch_resume))
            opts.epoch_start = epoch_resume + 1


    if opts.multi_test:
        val_dataset = [problem.make_dataset(
            size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset_uniform,
            distribution='uniform', n_cluster=opts.n_cluster, n_cluster_mix=opts.n_cluster_mix),
            problem.make_dataset(
                size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset_cluster,
                distribution='cluster',n_cluster=opts.n_cluster, n_cluster_mix=opts.n_cluster_mix),
            problem.make_dataset(
                size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset_mixed,
                distribution='mixed', n_cluster=opts.n_cluster, n_cluster_mix=opts.n_cluster_mix)]
    else:
        # Start the actual training loop
        val_dataset = problem.make_dataset(
            size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset,
            distribution=opts.data_distribution, n_cluster = opts.n_cluster, n_cluster_mix=opts.n_cluster_mix)

    if opts.eval_only:
        if opts.multi_test:
            validate(model, val_dataset[0], opts, print_log='uniform')
            validate(model, val_dataset[1], opts, print_log='cluster')
            validate(model, val_dataset[2], opts, print_log='mixed')
        else:
            validate(model, val_dataset, opts)

    elif opts.distillation:
        # Initialize student model
        print("AMDKD is working...")
        student_model = student_AttentionModel(
                                    opts.student_embedding_dim,
                                    opts.student_hidden_dim,
                                    problem,
                                    n_encode_layers=opts.student_n_encode_layers,
                                    mask_inner=True,
                                    mask_logits=True,
                                    normalization=opts.student_normalization,
                                    tanh_clipping=opts.tanh_clipping,
                                    checkpoint_encoder=opts.checkpoint_encoder,
                                    shrink_size=opts.shrink_size,
                                    tb_logger_path=tb_path,
                                    distill_temp=opts.distill_temperature,
                                    feed_forward_hidden=opts.student_feed_forward_hidden
        ).to(opts.device)

        if opts.use_cuda and torch.cuda.device_count() > 1 and opts.all_cuda_visible:
            student_model = torch.nn.DataParallel(student_model)
        elif not opts.all_cuda_visible and len(opts.CUDA_VISIBLE_ID) > 1:
            student_model = torch.nn.DataParallel(student_model)

        # Initialize baseline
        if opts.baseline == 'exponential':
            baseline = ExponentialBaseline(opts.exp_beta)
        elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
            assert problem.NAME == 'tsp', "Critic only supported for TSP"
            baseline = CriticBaseline(
                (
                    CriticNetworkLSTM(
                        2,
                        opts.embedding_dim,
                        opts.hidden_dim,
                        opts.n_encode_layers,
                        opts.tanh_clipping
                    )
                    if opts.baseline == 'critic_lstm'
                    else
                    CriticNetwork(
                        2,
                        opts.embedding_dim,
                        opts.hidden_dim,
                        opts.n_encode_layers,
                        opts.normalization
                    )
                ).to(opts.device)
            )
        elif opts.baseline == 'rollout':
            baseline = RolloutBaseline(student_model, problem, opts)
        else:
            assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
            baseline = NoBaseline()
        if opts.bl_warmup_epochs > 0:
            baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

        # Initialize optimizer
        optimizer = optim.Adam(
            [{'params': student_model.parameters(), 'lr': opts.lr_model}]
            + (
                [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
                if len(baseline.get_learnable_parameters()) > 0
                else []
            )
        )

        # Initialize learning rate scheduler, decay by lr_decay once per epoch!
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

        # load student model
        if opts.student_load or opts.resume:
            load_path = opts.student_load_path if opts.student_load_path is not None else opts.resume
            print('  [*] Loading data from {}'.format(load_path))
            load_data = torch_load_cpu(load_path)
            model_ = get_inner_model(student_model)
            if opts.is_load_multi:
                state_dict = load_data.get('model', {})
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                model_.load_state_dict({**model_.state_dict(), **new_state_dict})
            else:
                model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

            # Load baseline from data, make sure script is called with same type of baseline
            if 'baseline' in load_data:
                baseline.load_state_dict(load_data['baseline'])

            # Load optimizer state
            if 'optimizer' in load_data:
                optimizer.load_state_dict(load_data['optimizer'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        # if isinstance(v, torch.Tensor):
                        if torch.is_tensor(v):
                            state[k] = v.to(opts.device)

        if opts.resume:
            epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
            torch.set_rng_state(load_data['rng_state'])
            if opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
            # Set the random states
            # Dumping of state was done before epoch callback, so do that now (model is loaded)
            baseline.epoch_callback(student_model, epoch_resume)
            print("Resuming after {}".format(epoch_resume))
            opts.epoch_start = epoch_resume + 1

        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            val_result = train_epoch_distill(
                    model,
                    student_model,
                    optimizer,
                    baseline,
                    lr_scheduler,
                    epoch,
                    val_dataset,
                    problem,
                    tb_logger,
                    opts
            )
            if opts.adaptive_prob:
                opts.save_best = False
                gap = [(val_result[i].item()-opts.LKH3_optimal[i])/opts.LKH3_optimal[i] for i in range(len(val_result))]
                if opts.adaptive_prob_type == 'softmax':
                    opts.teacher_prob = softmax(gap)
                elif opts.adaptive_prob_type == 'sum':
                    opts.teacher_prob = [gap[i]/sum(gap) for i in range(len(gap))]
                print('Gap: ',[gap[i]*100 for i in range(len(gap))])
                print('mean_gap: ', np.mean(gap)*100)
                if opts.best ==0:
                    opts.best = np.mean(gap)*100
                elif np.mean(gap)*100 < opts.best:
                    opts.best = np.mean(gap) * 100
                    opts.save_best = True


    else:
        # original AM
        # Initialize learning rate scheduler, decay by lr_decay once per epoch!
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )



if __name__ == "__main__":
    run(get_options())

