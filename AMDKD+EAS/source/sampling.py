import math
import time

import numpy as np
from tqdm import tqdm

from source.TORCH_OBJECTS import *

AUG_S = 8  # Number of augmentations


def run_sampling(grouped_actor, instance_data, problem_size, config, get_episode_data_fn,
                 augment_and_repeat_episode_data_fn):
    """
    Perform a search by sampling multiple solutions for each instance can return the best one. This is the baseline
    for our experiments.
    """

    dataset_size = len(instance_data[0])

    assert config.batch_size <= dataset_size

    instance_solutions = torch.zeros(dataset_size, problem_size * 2, dtype=torch.int)
    instance_costs = np.zeros((dataset_size))

    if config.problem == "TSP":
        from source.tsp.env import GROUP_ENVIRONMENT
    elif config.problem == "CVRP":
        from source.cvrp.env import GROUP_ENVIRONMENT

    for episode in tqdm(range(math.ceil(dataset_size / config.batch_size))):
        with torch.no_grad():
            # Load the instances
            ###############################################

            episode_data = get_episode_data_fn(instance_data, episode * config.batch_size, config.batch_size, problem_size)
            batch_size = episode_data[0].shape[0]  # Number of instances considered in this iteration

            p_runs = config.p_runs  # Number of parallel runs per instance
            batch_r = batch_size * p_runs  # Search runs per batch
            batch_s = AUG_S * batch_r  # Model batch size (nb. of instances * the number of augmentations * p_runs)
            group_s = problem_size # Number of different rollouts per instance

            aug_data = augment_and_repeat_episode_data_fn(episode_data, problem_size, p_runs, AUG_S)
            env = GROUP_ENVIRONMENT(aug_data, problem_size, config.round_distances)
            group_state, reward, done = env.reset(group_size=group_s)
            grouped_actor.reset(group_state)

            max_reward = torch.full((batch_size,), -np.inf, device="cuda")
            incumbent_solutions = torch.zeros(batch_size, problem_size * 2, dtype=torch.int)

            # Start the search
            ###############################################

            t_start = time.time()
            for iter in range(config.max_iter):
                group_state, reward, done = env.reset(group_size=group_s)

                # Start generating batch_s * group_s solutions
                ###############################################

                solutions = []

                if config.problem == "CVRP":
                    # First Move is given
                    first_action = LongTensor(np.zeros((batch_s, group_s)))  # start from node_0-depot
                    group_state, reward, done = env.step(first_action)
                    solutions.append(first_action.unsqueeze(2))

                # Second Move is given
                second_action = LongTensor(np.arange(group_s) % problem_size)[None, :].expand(batch_s, group_s).clone()
                group_state, reward, done = env.step(second_action)
                solutions.append(second_action.unsqueeze(2))

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

                group_reward = reward.reshape(AUG_S, int(batch_s / AUG_S), group_s)
                solutions = torch.cat(solutions, dim=2)

                # Solution generation finished. Update incumbent solutions and best rewards
                ###############################################

                if config.batch_size == 1:
                    # Single instance search. Only a single incumbent solution exists that needs to be updated
                    max_reward_iter = group_reward.max()
                    if max_reward_iter > max_reward:
                        max_idx = torch.argmax(reward)
                        best_solution_iter = solutions.reshape(-1, solutions.shape[2])
                        best_solution_iter = best_solution_iter[max_idx]
                        incumbent_solutions[0, :best_solution_iter.shape[0]] = best_solution_iter
                        max_reward = max_reward_iter

                else:
                    # Batch search. Update incumbent etc. separately for each instance
                    max_reward_iter, _ = group_reward.max(dim=2)
                    max_reward_iter, _ = max_reward_iter.max(dim=0)
                    improved_idx = max_reward < max_reward_iter

                    if improved_idx.any():
                        reward_g = group_reward.permute(1, 0, 2).reshape(batch_size, -1)[improved_idx]
                        iter_max_k, iter_best_k = torch.topk(reward_g, k=1, dim=1)
                        solutions = solutions.reshape(8, batch_size, group_s, -1)
                        solutions = solutions.permute(1, 0, 2, 3).reshape(batch_size, 8 * group_s, -1)[[improved_idx]]
                        best_solutions_iter = torch.gather(solutions, 1, iter_best_k.unsqueeze(2).expand(-1, -1,
                                                                                                         solutions.shape[
                                                                                                             2])).squeeze(
                            1)
                        max_reward[improved_idx] = max_reward_iter[improved_idx]
                        incumbent_solutions[improved_idx,
                        :best_solutions_iter.shape[1]] = best_solutions_iter.int().cpu()


                if time.time() - t_start > config.max_runtime:
                    break

            # Store incumbent solutions and their objective function value
            instance_solutions[episode * config.batch_size: episode * config.batch_size + batch_size] = incumbent_solutions
            instance_costs[episode * config.batch_size: episode * config.batch_size + batch_size] = -max_reward.cpu().numpy()

    return instance_costs, instance_solutions
