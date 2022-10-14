import os
import time
from tqdm import tqdm
import torch
import math
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from utils import torch_load_cpu
from collections import OrderedDict
from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to
import warnings
warnings.simplefilter("ignore", UserWarning)

# default_collate_func = DataLoader.default_collate
# def default_collate_override(batch):
#     DataLoader._use_shared_memory = False
#     return default_collate_func(batch)
# setattr(DataLoader, 'default_collate', default_collate_override)

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts, print_log=None):
    # Validate
    if not print_log: print('Validating...')
    cost = rollout(model, dataset, opts, progress_bar=True)
    avg_cost = cost.mean()
    if print_log is None:
        print('Validation overall avg_cost: {} +- {}'.format(avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    else:
        print('Validation {} avg_cost: {} +- {}'.format(print_log, avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts, progress_bar=None):
    if progress_bar is None:
        progress_bar = opts.no_progress_bar
    # Put in greedy evaluation mode!
    set_decode_type(model, opts.test_type)
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device),opts=opts)
        return cost.data.cpu()

    if isinstance(dataset, dict):
        out = None
        for k, v in dataset.items():
            if out is None:
                out = torch.cat([
                    eval_model_bat(bat)
                    for bat
                    in tqdm(DataLoader(dataset[k], batch_size=opts.eval_batch_size), disable=progress_bar)
                ], 0)
            else:
                tmp = torch.cat([
                    eval_model_bat(bat)
                    for bat
                    in tqdm(DataLoader(dataset[k], batch_size=opts.eval_batch_size), disable=progress_bar)
                ], 0)
                out = torch.cat([out, tmp], dim=0)
        return out
    return  torch.cat([
            eval_model_bat(bat)
            for bat
            in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=progress_bar)
        ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch_distill(teacher_model, student_model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem,
                        tb_logger, opts):
    print("****************************************************************************************")
    print("Start train AMDKD student model epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'],opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()
    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # load the teacher model
    if opts.multi_teacher and opts.distill_distribution:
        # load the teacher model
        for i in ['uniform','cluster','mixed']:
            load_path = opts.load_path_multi[i]
            print('  [*] Loading data from {}'.format(load_path))
            load_data = torch_load_cpu(load_path)
            model_ = get_inner_model(teacher_model[i])
            if opts.is_load_multi:
                state_dict = load_data.get('model', {})
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                model_.load_state_dict({**model_.state_dict(), **new_state_dict})
            else:
                model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})
            teacher_model[i].eval()
            set_decode_type(teacher_model[i], "sampling")

    elif opts.distill_distribution:

        if opts.adaptive_prob and epoch >=opts.start_adaptive_epoch:
            if opts.random_adaptive_prob!=0: # randomly choose the equal prob mode OR the adaptive mode
                if np.random.rand()<opts.random_adaptive_prob:# choose the adaptive prob mode randomly
                    class_type = np.random.choice(['uniform', 'cluster', 'mixed'], size=1, p=opts.teacher_prob)
                    load_path = opts.load_path_multi[class_type.item()]
                    print('  [*] Loading data from {}, prob: {} [randomly choose]'.format(load_path, opts.teacher_prob))
                else: # choose the equal prob mode randomly
                    class_type = np.random.choice(['uniform', 'cluster', 'mixed'], 1)
                    load_path = opts.load_path_multi[class_type.item()]
                    print('  [*] Loading data from {} [randomly choose]'.format(load_path))
            else: # directly use the adaptive prob mode
                class_type = np.random.choice(['uniform', 'cluster', 'mixed'], size=1, p=opts.teacher_prob)
                load_path = opts.load_path_multi[class_type.item()]
                print('  [*] Loading data from {}, prob: {}'.format(load_path, opts.teacher_prob))
        else: # directly use the equal prob mode
            class_type = np.random.choice(['uniform','cluster','mixed'],1)
            load_path = opts.load_path_multi[class_type.item()]
            print('  [*] Loading data from {}'.format(load_path))

        load_data = torch_load_cpu(load_path)
        model_ = get_inner_model(teacher_model[class_type.item()])
        if opts.is_load_multi:
            state_dict = load_data.get('model', {})
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model_.load_state_dict({**model_.state_dict(), **new_state_dict})
        else:
            model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

        teacher_model = teacher_model[class_type.item()]
        # set model to training mode
        teacher_model.eval()
        set_decode_type(teacher_model, "sampling")

    # set model to training mode
    student_model.train()
    set_decode_type(student_model, "sampling")


    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution,
        n_cluster=opts.n_cluster, mix_data=opts.generate_mix_data))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0)

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        train_batch_distill(
                            teacher_model,
                            student_model,
                            optimizer,
                            baseline,
                            step,
                            batch,
                            tb_logger,
                            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))


    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(student_model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

        if opts.distill_distribution and not opts.multi_teacher:
            model_path = os.path.join(opts.save_dir, 'epoch-{}-{}-model-only.pt'.format(epoch,class_type.item()))
        else:
            model_path = os.path.join(opts.save_dir, 'epoch-{}-model-only.pt'.format(epoch))

        torch.save(
            {
                'model': get_inner_model(student_model).state_dict()
            },
            model_path
        )
    if opts.save_best:
        print("Saving best trained_model")
        torch.save(
            {
                'best':opts.best,
                'epoch':epoch,
                'model': get_inner_model(student_model).state_dict(),
            },
            os.path.join(opts.save_dir, 'epoch-best.pt')
        )


    if opts.multi_test:
        avg_reward_uniform = validate(student_model, val_dataset[0], opts, print_log='uniform')
        avg_reward_cluster = validate(student_model, val_dataset[1], opts, print_log='cluster')
        avg_reward_mixed = validate(student_model, val_dataset[2], opts, print_log='mixed')
        if not opts.no_tensorboard:
            tb_logger.log_value('val_avg_reward_uniform', avg_reward_uniform, step)
            tb_logger.log_value('val_avg_reward_cluster', avg_reward_cluster, step)
            tb_logger.log_value('val_avg_reward_mixed', avg_reward_mixed, step)
    else:
        avg_reward = validate(student_model, val_dataset, opts)
        if not opts.no_tensorboard:
            tb_logger.log_value('val_avg_reward', avg_reward, step)

    if not opts.no_tensorboard:
        tb_logger.log_value('epoch_duration', epoch_duration, step)

    baseline.epoch_callback(student_model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()

    if opts.adaptive_prob:
        return [avg_reward_uniform,avg_reward_cluster,avg_reward_mixed]


def train_batch_distill(
        teacher_model,
        student_model,
        optimizer,
        baseline,
        step,
        batch,
        tb_logger,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    if opts.router == 'teacher':
        if opts.multi_teacher:
            # Teacher
            teacher_embeddings, teacher_hidden, teacher_logp, teacher_pi, teacher_attn = {},{},{},{},{}
            for i in ['uniform','cluster','mixed']:
                with torch.no_grad():
                    teacher_embeddings[i], teacher_hidden[i],teacher_attn[i], teacher_logp[i], _, _, teacher_pi[i] = teacher_model[i](x, distillation=True,return_pi=True,opts=opts)
            # Student
            # Evaluate model, get costs and log probabilities
            class_type = np.random.choice(['uniform', 'cluster', 'mixed'], 1).item()
            print('Randomly choose a {} teacher to route if you use teacher as router in multi-teacher'.format(class_type))
            student_embeddings, student_hidden, student_attn, student_logp, cost, log_likelihood, student_pi = student_model(x, return_pi=True, log_time=False, distillation=True,route=teacher_pi[class_type],opts=opts)

            assert torch.equal(teacher_pi[class_type], student_pi), "Teacher route and student route are not same!"
        else:
            with torch.no_grad():
                teacher_embeddings, teacher_hidden,teacher_attn, teacher_logp, _, _, teacher_pi = teacher_model(x, distillation=True,return_pi=True,opts=opts)

            # Student
            # Evaluate model, get costs and log probabilities
            student_embeddings, student_hidden, student_logp, cost, log_likelihood, student_pi = student_model(x, return_pi=True, log_time=False, distillation=True,route=teacher_pi,opts=opts)

            assert torch.equal(teacher_pi, student_pi), "Teacher route and student route are not same!"

    elif opts.router == 'student':
        # Student
        # Evaluate model, get costs and log probabilities
        student_embeddings, student_hidden, student_attn, student_logp, cost, log_likelihood, student_pi = student_model(x,return_pi=True,log_time=False,distillation=True,opts=opts)


        # Teacher
        if opts.multi_teacher:
            # Teacher
            teacher_embeddings, teacher_hidden, teacher_logp, teacher_pi, teacher_attn = {},{},{},{},{}
            for i in ['uniform','cluster','mixed']:
                with torch.no_grad():
                    teacher_embeddings[i], teacher_hidden[i],teacher_attn[i], teacher_logp[i], _, _, teacher_pi[i] = teacher_model[i](x, return_pi=True, distillation=True, route=student_pi,opts=opts)

                assert torch.equal(teacher_pi[i], student_pi), "Teacher route and student route are not same!"
        else:
            with torch.no_grad():
                teacher_embeddings, teacher_hidden, teacher_attn, teacher_logp, _, _, teacher_pi = teacher_model(x,return_pi=True,distillation=True,route=student_pi,opts=opts)

            assert torch.equal(teacher_pi, student_pi), "Teacher route and student route are not same!"



    # Student go on getting its reinforcement task loss
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    task_loss = reinforce_loss + bl_loss

    if opts.multi_teacher:
        soft_loss0 = [nn.KLDivLoss()(student_logp, teacher_logp[_].exp()) for _ in ['uniform', 'cluster', 'mixed']]
        soft_loss = torch.zeros(1).to(opts.device)
        for i in range(3):
            soft_loss.add_(soft_loss0[i])
    else:
        if opts.meaningful_KLD:
            soft_loss = (teacher_logp.exp() * ((teacher_logp.exp()+0.00001).log() - (student_logp.exp()+0.00001).log())).sum(dim=1).mean()\
                if not opts.twist_kldloss else \
                (student_logp.exp() * ((student_logp.exp()+0.00001).log() - (teacher_logp.exp()+0.00001).log())).sum(dim=1).mean()
        else:
            soft_loss = nn.KLDivLoss()(student_logp,teacher_logp.exp()) if not opts.twist_kldloss else nn.KLDivLoss()(teacher_logp, student_logp.exp())

        # loss function from Hinton et. al, 2015 (soft * alpha*T^2 + hard * (1-alpha))
        # loss = soft_loss * (opts.distill_alpha * opts.distill_temperature * opts.distill_temperature) + task_loss \
        #     if opts.hinton_t2 else task_loss  + soft_loss * opts.distill_alpha
        loss = task_loss * opts.rl_alpha  + soft_loss * opts.distill_alpha


    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, step, log_likelihood,
                   reinforce_loss, bl_loss, soft_loss,loss, tb_logger, opts)

