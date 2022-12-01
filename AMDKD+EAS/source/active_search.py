import math
import time

import numpy as np
import torch.optim as optim
from tqdm import tqdm

from source.MODEL_HYPER_PARAMS import *  # NOTE : You much edit HYPER_PARAMS to match the model you are loading
from source.TORCH_OBJECTS import *

AUG_S = 8  # Number of augmentations


def run_active_search(grouped_actor, instance_data, problem_size, config, get_episode_data_fn,
                      augment_and_repeat_episode_data_fn):
    """
    Original active search as proposed by Bello et al. Search for solutions by using reinforcement learning to fine
    tune all model weights (encoder + decoder) to a single test instance.
    """
    dataset_size = len(instance_data[0])

    assert config.batch_size == 1

    instance_solutions = torch.zeros(dataset_size, problem_size * 2, dtype=torch.int)
    instance_costs = np.zeros((dataset_size))

    if config.problem == "TSP":
        from source.tsp.env import GROUP_ENVIRONMENT
    elif config.problem == "CVRP":
        from source.cvrp.env import GROUP_ENVIRONMENT

    original_actor_state_dict = grouped_actor.state_dict()

    for episode in tqdm(range(math.ceil(dataset_size / config.batch_size))):

        # Load the instances
        ###############################################

        episode_data = get_episode_data_fn(instance_data, episode * config.batch_size, config.batch_size, problem_size)
        batch_size = episode_data[0].shape[0]  # Number of instances considered in this iteration

        p_runs = config.p_runs  # Number of parallel runs per instance
        batch_r = batch_size * p_runs  # Search runs per batch
        batch_s = AUG_S * batch_r  # Model batch size (nb. of instances * the number of augmentations * p_runs)
        group_s = problem_size  # Number of different rollouts per instance (+1 for incumbent solution construction)

        with torch.no_grad():
            aug_data = augment_and_repeat_episode_data_fn(episode_data, problem_size, p_runs, AUG_S)
            env = GROUP_ENVIRONMENT(aug_data, problem_size, config.round_distances)
            grouped_actor.load_state_dict(
                state_dict=original_actor_state_dict)  # Reset the model weights before solving the next instance

        optimizer = optim.Adam(grouped_actor.parameters(), lr=config.param_lr,
                               weight_decay=ACTOR_WEIGHT_DECAY)  # We want to optimize all model weights

        max_reward = torch.full((batch_size,), -np.inf, device="cuda")
        incumbent_solutions = torch.zeros(batch_size, problem_size * 2, dtype=torch.int)

        # Start the search
        ###############################################

        t_start = time.time()
        for iter in range(config.max_iter):
            group_state, reward, done = env.reset(group_size=group_s)
            grouped_actor.reset(group_state)  # Generate embeddings

            # Start generating batch_s * group_s solutions
            ###############################################

            solutions = []

            if config.problem == "CVRP":
                # First Move is given
                first_action = LongTensor(np.zeros((batch_s, group_s)))  # start from node_0-depot
                group_state, reward, done = env.step(first_action)
                solutions.append(first_action.unsqueeze(2))

            # First/Second Move is given
            second_action = LongTensor(np.arange(group_s) % problem_size)[None, :].expand(batch_s, group_s).clone()

            group_state, reward, done = env.step(second_action)
            solutions.append(second_action.unsqueeze(2))

            group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
            while not done:
                action_probs = grouped_actor.get_action_probabilities(group_state)
                # shape = (batch_s, group_s, problem)
                action = action_probs.reshape(batch_s * group_s, -1).multinomial(1) \
                    .squeeze(dim=1).reshape(batch_s, group_s)
                # shape = (batch_s, group_s)

                if config.problem == "CVRP":
                    action[group_state.finished] = 0  # stay at depot, if you are finished
                group_state, reward, done = env.step(action)
                solutions.append(action.unsqueeze(2))

                batch_idx_mat = torch.arange(int(batch_s))[:, None].expand(batch_s, group_s)
                group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
                chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)
                # shape = (batch_s, group_s)

                if config.problem == "CVRP":
                    chosen_action_prob[group_state.finished] = 1  # done episode will gain no more probability
                group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)

            # Solution generation finished. Update incumbent solutions and best rewards
            ###############################################

            group_reward = reward.reshape(AUG_S, batch_r, group_s)
            solutions = torch.cat(solutions, dim=2)

            max_reward_iter = group_reward.max()
            if max_reward_iter > max_reward:
                max_idx = torch.argmax(reward)
                best_solution_iter = solutions.reshape(-1, solutions.shape[2])
                best_solution_iter = best_solution_iter[max_idx]
                incumbent_solutions[0, :best_solution_iter.shape[0]] = best_solution_iter
                max_reward = max_reward_iter

            # LEARNING - Actor
            # Use the same reinforcement learning method as during the training of the model
            ###############################################
            group_log_prob = group_prob_list.log().sum(dim=2)
            # shape = (batch_s, group_s)

            advantage = reward - reward.mean(dim=1, keepdim=True)

            group_loss = -advantage * group_log_prob
            # shape = (batch_s, group_s)
            loss = group_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if time.time() - t_start > config.max_runtime:
                break

        # Store incumbent solutions and their objective function value
        instance_solutions[episode * config.batch_size: episode * config.batch_size + batch_size] = incumbent_solutions
        instance_costs[
        episode * config.batch_size: episode * config.batch_size + batch_size] = -max_reward.cpu().numpy()

    return instance_costs, instance_solutions
