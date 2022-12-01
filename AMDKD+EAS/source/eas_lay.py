import math
import time

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from source.MODEL_HYPER_PARAMS import *  # NOTE : You much edit HYPER_PARAMS to match the model you are loading
from source.TORCH_OBJECTS import *
from source.cvrp.grouped_actors import multi_head_attention as multi_head_attention_CVRP
from source.cvrp.grouped_actors import reshape_by_heads as reshape_by_heads_CVRP
from source.tsp.grouped_actors import multi_head_attention as multi_head_attention_TSP
from source.tsp.grouped_actors import reshape_by_heads as reshape_by_heads_TSP

AUG_S = 8


class prob_calc_added_layers_CVRP(nn.Module):
    """New nn.Module with added layers for the CVRP.

    same as source.MODEL__Actor.grouped.actors.Next_Node_Probability_Calculator_for_group with added layers.
    """

    def __init__(self, batch_s):
        super().__init__()

        self.Wq_last = nn.Linear(EMBEDDING_DIM + 1, HEAD_NUM * KEY_DIM, bias=False)
        self.Wk = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wv = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)

        self.new = nn.Parameter(torch.zeros((batch_s, HEAD_NUM * KEY_DIM, HEAD_NUM * KEY_DIM), requires_grad=True))
        self.new_bias = nn.Parameter(torch.zeros((batch_s, 1, HEAD_NUM * KEY_DIM), requires_grad=True))
        self.new_2 = nn.Parameter(torch.zeros((batch_s, HEAD_NUM * KEY_DIM, HEAD_NUM * KEY_DIM), requires_grad=True))
        self.new_bias_2 = nn.Parameter(torch.zeros((batch_s, 1, HEAD_NUM * KEY_DIM), requires_grad=True))

        torch.nn.init.xavier_uniform_(self.new)
        torch.nn.init.xavier_uniform_(self.new_bias)

        self.multi_head_combine = nn.Linear(HEAD_NUM * KEY_DIM, EMBEDDING_DIM)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

    def reset(self, encoded_nodes):
        # encoded_nodes.shape = (batch, problem+1, EMBEDDING_DIM)

        self.k = reshape_by_heads_CVRP(self.Wk(encoded_nodes), head_num=HEAD_NUM)
        self.v = reshape_by_heads_CVRP(self.Wv(encoded_nodes), head_num=HEAD_NUM)
        # shape = (batch, HEAD_NUM, problem+1, KEY_DIM)
        self.single_head_key = encoded_nodes.transpose(1, 2).detach()
        self.single_head_key.requires_grad = True
        # shape = (batch, EMBEDDING_DIM, problem+1)

#         self.encoded_graph = encoded_graph

    def forward(self, input2, remaining_loaded, ninf_mask=None):
        # input1.shape = (batch, 1, EMBEDDING_DIM)
        # input2.shape = (batch, group, EMBEDDING_DIM)
        # remaining_loaded.shape = (batch, group, 1)
        # ninf_mask.shape = (batch, group, problem+1)

        with torch.no_grad():
            group_s = input2.size(1)

            #  Multi-Head Attention
            #######################################################
            input_cat = torch.cat((input2, remaining_loaded), dim=2)
            # shape = (batch, group, 2*EMBEDDING_DIM+1)

            q = reshape_by_heads_CVRP(self.Wq_last(input_cat), head_num=HEAD_NUM)
            # shape = (batch, HEAD_NUM, group, KEY_DIM)

            out_concat = multi_head_attention_CVRP(q, self.k, self.v, ninf_mask=ninf_mask)
            # shape = (batch, n, HEAD_NUM*KEY_DIM)

        # Added layers start
        ###############################################

        residual = out_concat.detach()
        # out_concat = torch.matmul(out_concat.permute(1, 0, 2).unsqueeze(2), self.new)
        # out_concat = out_concat.squeeze(2).permute(1, 0, 2)
        out_concat = F.relu(torch.matmul(out_concat, self.new) + self.new_bias.expand_as(out_concat))
        out_concat = torch.matmul(out_concat, self.new_2) + self.new_bias_2.expand_as(out_concat)
        out_concat += residual
        # shape = (batch, n, HEAD_NUM*KEY_DIM)

        # Added layers end
        ###############################################

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape = (batch, n, EMBEDDING_DIM)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key.detach())
        # shape = (batch, n, problem+1)

        score_scaled = score / np.sqrt(EMBEDDING_DIM)
        # shape = (batch_s, group, problem+1)

        score_clipped = LOGIT_CLIPPING * torch.tanh(score_scaled)

        if ninf_mask is None:
            score_masked = score_clipped
        else:
            score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape = (batch, group, problem+1)

        return probs


class prob_calc_added_layers_TSP(nn.Module):
    """New nn.Module with added layers for the TSP.

    same as source.MODEL__Actor.grouped.actors.Next_Node_Probability_Calculator_for_group with added layers.
    """

    def __init__(self, batch_s):
        super().__init__()

#         self.Wq_graph = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wq_first = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wq_last = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wk = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wv = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)

        self.new = nn.Parameter(torch.zeros((batch_s, HEAD_NUM * KEY_DIM, HEAD_NUM * KEY_DIM), requires_grad=True))
        self.new_bias = nn.Parameter(torch.zeros((batch_s, 1, HEAD_NUM * KEY_DIM), requires_grad=True))
        self.new_2 = nn.Parameter(torch.zeros((batch_s, HEAD_NUM * KEY_DIM, HEAD_NUM * KEY_DIM), requires_grad=True))
        self.new_bias_2 = nn.Parameter(torch.zeros((batch_s, 1, HEAD_NUM * KEY_DIM), requires_grad=True))

        torch.nn.init.xavier_uniform(self.new)
        torch.nn.init.xavier_uniform(self.new_bias)

        self.multi_head_combine = nn.Linear(HEAD_NUM * KEY_DIM, EMBEDDING_DIM)

#         self.q_graph = None  # saved q1, for multi-head attention
        self.q_first = None  # saved q2, for multi-head attention
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

    def reset(self, encoded_nodes):
        # encoded_nodes.shape = (batch_s, TSP_SIZE, EMBEDDING_DIM)

#         self.q_graph = reshape_by_heads_TSP(self.Wq_graph(encoded_graph), head_num=HEAD_NUM)
        # shape = (batch_s, HEAD_NUM, 1, KEY_DIM)
        self.q_first = None
        # shape = (batch_s, HEAD_NUM, group, KEY_DIM)
        self.k = reshape_by_heads_TSP(self.Wk(encoded_nodes), head_num=HEAD_NUM)
        self.v = reshape_by_heads_TSP(self.Wv(encoded_nodes), head_num=HEAD_NUM)
        # shape = (batch_s, HEAD_NUM, TSP_SIZE, KEY_DIM)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape = (batch_s, EMBEDDING_DIM, TSP_SIZE)
        # self.group_ninf_mask = group_ninf_mask
        # shape = (batch_s, group, TSP_SIZE)

    def forward(self, encoded_LAST_NODE, group_ninf_mask):
        # encoded_LAST_NODE.shape = (batch_s, group, EMBEDDING_DIM)

        with torch.no_grad():
            if self.q_first is None:
                self.q_first = reshape_by_heads_TSP(self.Wq_first(encoded_LAST_NODE), head_num=HEAD_NUM)
            # shape = (batch_s, HEAD_NUM, group, KEY_DIM)

            #  Multi-Head Attention
            #######################################################
            q_last = reshape_by_heads_TSP(self.Wq_last(encoded_LAST_NODE), head_num=HEAD_NUM)
            # shape = (batch_s, HEAD_NUM, group, KEY_DIM)

            q = self.q_first + q_last
            # shape = (batch_s, HEAD_NUM, group, KEY_DIM)

            out_concat = multi_head_attention_TSP(q, self.k, self.v, group_ninf_mask=group_ninf_mask)
            # shape = (batch_s, group, HEAD_NUM*KEY_DIM)

        # Added layers start
        ###############################################

        residual = out_concat.detach()
        # out_concat = torch.matmul(out_concat.permute(1, 0, 2).unsqueeze(2), self.new)
        # out_concat = out_concat.squeeze(2).permute(1, 0, 2)
        out_concat = F.relu(torch.matmul(out_concat, self.new) + self.new_bias.expand_as(out_concat))
        out_concat = torch.matmul(out_concat, self.new_2) + self.new_bias_2.expand_as(out_concat)
        out_concat += residual

        # Added layers end
        ###############################################

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape = (batch_s, group, EMBEDDING_DIM)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key.detach())
        # shape = (batch_s, group, TSP_SIZE)

        score_scaled = score / np.sqrt(EMBEDDING_DIM)
        # shape = (batch_s, group, TSP_SIZE)

        score_clipped = LOGIT_CLIPPING * torch.tanh(score_scaled)

        score_masked = score_clipped + group_ninf_mask.clone()

        probs = F.softmax(score_masked, dim=2)
        # shape = (batch_s, group, TSP_SIZE)

        return probs


def replace_decoder(grouped_actor, batch_s, state, problem):
    """Function to add layers to pretrained model while retaining weights from other layers."""

    # update decoder
    if problem == "CVRP":
        grouped_actor.decoder = prob_calc_added_layers_CVRP(batch_s)
    elif problem == "TSP":
        grouped_actor.decoder = prob_calc_added_layers_TSP(batch_s)
    grouped_actor.decoder.load_state_dict(state_dict=state, strict=False)

    return grouped_actor


#######################################################################################################################
# Search process
#######################################################################################################################

def run_eas_lay(grouped_actor, instance_data, problem_size, config, get_episode_data_fn,
                augment_and_repeat_episode_data_fn):
    """
    Efficient active search using added layer updates
    """

    dataset_size = len(instance_data[0])
    print("Dataset size: ", dataset_size)

    assert config.batch_size <= dataset_size

    original_decoder_state_dict = grouped_actor.decoder.state_dict()

    instance_solutions = torch.zeros(dataset_size, problem_size * 2, dtype=torch.int)
    instance_costs = np.zeros((dataset_size))

    if config.problem == "TSP":
        from source.tsp.env import GROUP_ENVIRONMENT
    elif config.problem == "CVRP":
        from source.cvrp.env import GROUP_ENVIRONMENT

    for episode in tqdm(range(math.ceil(dataset_size / config.batch_size))):

        # Load the instances
        ###############################################

        episode_data = get_episode_data_fn(instance_data, episode * config.batch_size, config.batch_size, problem_size)
        batch_size = episode_data[0].shape[0]  # Number of instances considered in this iteration

        p_runs = config.p_runs  # Number of parallel runs per instance
        batch_r = batch_size * p_runs  # Search runs per batch
        batch_s = AUG_S * batch_r  # Model batch size (nb. of instances * the number of augmentations * p_runs)
        group_s = problem_size + 1  # Number of different rollouts per instance (+1 for incumbent solution construction)

        with torch.no_grad():
            aug_data = augment_and_repeat_episode_data_fn(episode_data, problem_size, p_runs, AUG_S)
            if config.problem == "CVRP":
                env = GROUP_ENVIRONMENT(aug_data, problem_size, config.round_distances)
            elif config.problem == "TSP":
                env = GROUP_ENVIRONMENT(aug_data, problem_size, config.round_distances,config.max_scaler)
            # Replace the decoder of the loaded model with the modified decoder with added layers
            grouped_actor_modified = replace_decoder(grouped_actor, batch_s, original_decoder_state_dict,
                                                     config.problem).cuda()

            group_state, reward, done = env.reset(group_size=group_s)
            grouped_actor_modified.reset(group_state)  # Generate the embeddings

        # Only update the weights of the added layer during training
        optimizer = optim.Adam(
            [grouped_actor_modified.decoder.new, grouped_actor_modified.decoder.new_2,
             grouped_actor_modified.decoder.new_bias,
             grouped_actor_modified.decoder.new_bias_2], lr=config.param_lr,
            weight_decay=ACTOR_WEIGHT_DECAY)

        incumbent_solutions = torch.zeros(batch_size, problem_size * 2, dtype=torch.int)

        # Start the search
        ###############################################

        t_start = time.time()
        for iter in range(config.max_iter):
            group_state, reward, done = env.reset(group_size=group_s)

            incumbent_solutions_expanded = incumbent_solutions.repeat(AUG_S, 1).repeat(p_runs, 1)

            # Start generating batch_s * group_s solutions
            ###############################################
            solutions = []

            step = 0
            if config.problem == "CVRP":
                # First Move is given
                first_action = LongTensor(np.zeros((batch_s, group_s)))  # start from node_0-depot
                group_state, reward, done = env.step(first_action)
                solutions.append(first_action.unsqueeze(2))
                step += 1

            # First/Second Move is given
            second_action = LongTensor(np.arange(group_s) % problem_size)[None, :].expand(batch_s, group_s).clone()

            if iter > 0:
                second_action[:, -1] = incumbent_solutions_expanded[:,
                                       step]  # Teacher forcing the imitation learning loss
            group_state, reward, done = env.step(second_action)
            solutions.append(second_action.unsqueeze(2))
            step += 1

            group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
            while not done:
                action_probs = grouped_actor_modified.get_action_probabilities(group_state)
                # shape = (batch_s, group_s, problem)
                action = action_probs.reshape(batch_s * group_s, -1).multinomial(1) \
                    .squeeze(dim=1).reshape(batch_s, group_s)
                # shape = (batch_s, group_s)
                if iter > 0:
                    action[:, -1] = incumbent_solutions_expanded[:, step]  # Teacher forcing the imitation learning loss

                if config.problem == "CVRP":
                    action[group_state.finished] = 0  # stay at depot, if you are finished
                group_state, reward, done = env.step(action)
                solutions.append(action.unsqueeze(2))

                batch_idx_mat = torch.arange(int(batch_s))[:, None].expand(batch_s,
                                                                           group_s)
                group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
                chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)
                # shape = (batch_s, group_s)
                if config.problem == "CVRP":
                    chosen_action_prob[group_state.finished] = 1  # done episode will gain no more probability
                group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)
                step += 1

            # Solution generation finished. Update incumbent solutions and best rewards
            ###############################################

            group_reward = reward.reshape(AUG_S, batch_r, group_s)
            solutions = torch.cat(solutions, dim=2)
            if config.batch_size == 1:
                # Single instance search. Only a single incumbent solution exists that needs to be updated
                max_idx = torch.argmax(reward)
                best_solution_iter = solutions.reshape(-1, solutions.shape[2])
                best_solution_iter = best_solution_iter[max_idx]
                incumbent_solutions[0, :best_solution_iter.shape[0]] = best_solution_iter
                max_reward = reward.max()

            else:
                # Batch search. Update incumbent etc. separately for each instance
                max_reward, _ = group_reward.max(dim=2)
                max_reward, _ = max_reward.max(dim=0)

                reward_g = group_reward.permute(1, 0, 2).reshape(batch_r, -1)
                iter_max_k, iter_best_k = torch.topk(reward_g, k=1, dim=1)
                solutions = solutions.reshape(AUG_S, batch_r, group_s, -1)
                solutions = solutions.permute(1, 0, 2, 3).reshape(batch_r, AUG_S * group_s, -1)
                best_solutions_iter = torch.gather(solutions, 1,
                                                   iter_best_k.unsqueeze(2).expand(-1, -1, solutions.shape[2])).squeeze(
                    1)
                incumbent_solutions[:, :best_solutions_iter.shape[1]] = best_solutions_iter

            # LEARNING - Actor
            # Use the same reinforcement learning method as during the training of the model
            # to update only the weights of the newly added layers
            ###############################################
            group_reward = reward[:, :group_s - 1]
            # shape = (batch_s, group_s - 1)
            group_log_prob = group_prob_list.log().sum(dim=2)
            # shape = (batch_s, group_s)

            group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

            group_loss = -group_advantage * group_log_prob[:, :group_s - 1]
            # shape = (batch_s, group_s - 1)
            loss_1 = group_loss.mean()  # Reinforcement learning loss
            loss_2 = -group_log_prob[:, group_s - 1].mean()  # Imitation learning loss
            loss = loss_1 + loss_2 * config.param_lambda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

#             print(max_reward)

            if time.time() - t_start > config.max_runtime:
                break

        # Store incumbent solutions and their objective function value
        instance_solutions[episode * config.batch_size: episode * config.batch_size + batch_size] = incumbent_solutions
        instance_costs[
        episode * config.batch_size: episode * config.batch_size + batch_size] = -max_reward.cpu().numpy()

    return instance_costs, instance_solutions
