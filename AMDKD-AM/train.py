import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts, print_log=None):
    # Validate
    print('Validating...')
    torch.cuda.synchronize()
    tik = time.time()
    cost = rollout(model, dataset, opts)
    print(time.time() - tik)
    avg_cost = cost.mean()
    if print_log is None:
        print('Validation overall avg_cost: {} +- {}'.format(avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    else:
        print('Validation {} avg_cost: {} +- {}'.format(print_log, avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts, progress_bar = None):
    if progress_bar is None:
        progress_bar = opts.no_progress_bar
    # Put in greedy evaluation mode!
    set_decode_type(model, opts.test_type)
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device), opts = opts)
        return cost.data.cpu()

    if isinstance(dataset, dict):
        out = None
        for _, data in dataset.items():
            if out is None:
                out = torch.cat([
                            eval_model_bat(bat)
                            for bat
                            in tqdm(DataLoader(data, batch_size=opts.eval_batch_size), disable=progress_bar)
                        ], 0)
            else:
                tmp = torch.cat([
                            eval_model_bat(bat)
                            for bat
                            in tqdm(DataLoader(data, batch_size=opts.eval_batch_size), disable=progress_bar)
                        ], 0)
                out = torch.cat([out, tmp], dim=0)
        return out
    else:
        return torch.cat([
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


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)

    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution,
         n_cluster = opts.n_cluster, mix_data=opts.generate_mix_data))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
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
        # torch.save(get_inner_model(model).state_dict(),os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch)))
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    if opts.multi_test:
        avg_reward_uniform = validate(model, val_dataset[0], opts, print_log='uniform')
        avg_reward_cluster = validate(model, val_dataset[1], opts, print_log='cluster')
        avg_reward_mixed = validate(model, val_dataset[2], opts, print_log='mixed')
        if not opts.no_tensorboard:
            tb_logger.log_value('val_avg_reward_uniform', avg_reward_uniform, step)
            tb_logger.log_value('val_avg_reward_cluster', avg_reward_cluster, step)
            tb_logger.log_value('val_avg_reward_mixed', avg_reward_mixed, step)
    else:
        avg_reward = validate(model, val_dataset, opts)
        if not opts.no_tensorboard:
            tb_logger.log_value('val_avg_reward', avg_reward, step)

    if not opts.no_tensorboard:
        tb_logger.log_value('epoch_duration', epoch_duration, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()

def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    cost, log_likelihood = model(x,log_time = True, opts = opts)

    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, 0, 0,
                   tb_logger, opts)
