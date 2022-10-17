
import torch
import numpy as np
import os
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from utils.utils import *


class TSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        if '.pt' in model_load['path']:
            checkpoint_fullname = model_load['path']
        else:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        print(checkpoint_fullname)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            # infertime_all=[]
            score, aug_score, infertime = self._test_one_batch(batch_size, episode)
            # infertime_all.append(infertime)
            # print(np.mean(infertime_all))

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {} ".format(aug_score_AM.avg))

        # print(np.mean(infertime_all))
        # print(torch.cuda.max_memory_reserved())
        return aug_score_AM.avg

    def _test_one_batch(self, batch_size,episode=None):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor,load_path=self.env_params['load_path'],episode=episode)

            import time
            tik = time.time()

            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        torch.cuda.synchronize()
        tok = time.time()
        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item(), tok-tik

def validate(model, env, batch_size, augment = True, load_path = None):

    # Augmentation
    ###############################################
    if augment:
        aug_factor = 8
    else:
        aug_factor = 1
    # Ready
    ###############################################
    model.eval()
    with torch.no_grad():
        env.load_problems(batch_size, aug_factor,load_path = load_path)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

    # POMO Rollout
    ###############################################
    state, reward, done = env.pre_step()
    while not done:
        selected, _ = model(state)
        # shape: (batch, pomo)
        state, reward, done = env.step(selected)

    # Return
    ###############################################
    aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
    # shape: (augmentation, batch, pomo)

    max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
    # shape: (augmentation, batch)
    no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

    max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
    # shape: (batch,)
    aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

    return no_aug_score.item(), aug_score.item()
