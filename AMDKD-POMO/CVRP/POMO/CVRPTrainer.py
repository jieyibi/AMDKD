import torch
from torch import nn
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPTester import validate
from CVRPModel import CVRPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from tensorboard_logger import Logger as TbLogger

from utils.utils import *

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class CVRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        if trainer_params['distillation']:
            self.student_model_params = trainer_params['student_model_param']

        # result folder, logger
        self.logger = getLogger(name='trainer')
        if self.trainer_params['logging']['tb_logger']:
            self.tb_logger = TbLogger('./log/' + get_start_time() +self.trainer_params['tb_path'])
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # Main Components
        if trainer_params['distillation'] and trainer_params['distill_param']['multi_teacher']:# 3 teachers
            model_uniform = Model(**self.model_params)
            model_cluster = Model(**self.model_params)
            model_mixed = Model(**self.model_params)
            self.model =  {'uniform': model_uniform,
                            'cluster': model_cluster,
                            'mixed': model_mixed}
            self.env = Env(**self.env_params)
        else:
            self.model = Model(**self.model_params)
            self.env = Env(**self.env_params)

        if trainer_params['distillation']: # student model and env
            self.student_model = Model(**self.student_model_params)
            self.student_env = Env(**self.env_params)
            # only student model needs to be trained
            self.optimizer = Optimizer(self.student_model.parameters(), **self.optimizer_params['optimizer'])
        else:
            self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])

        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            if trainer_params['distillation']:
                checkpoint_fullname = model_load['path']
                checkpoint = torch.load(checkpoint_fullname, map_location=device)
                self.student_model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info('Saved Student Model Loaded !!')
            else:
                checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
                checkpoint = torch.load(checkpoint_fullname, map_location=device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info('Saved Model Loaded !!')

            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1


        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            if self.trainer_params['distillation']:
                if self.trainer_params['distill_param']['distill_distribution']:
                    train_score, train_loss, RL_loss, KLD_loss, class_type = self._distill_one_epoch(epoch,teacher_prob=self.trainer_params['distill_param']['teacher_prob'])
                    locals()[class_type + '_train_score'] = train_score
                    locals()[class_type + '_train_loss'] = train_loss
                    locals()[class_type + '_RL_loss'] = RL_loss
                    locals()[class_type + '_KLD_loss'] = KLD_loss
                    self.trainer_params['distill_param']['count'][class_type] += 1
                else:
                    train_score, train_loss, RL_loss, KLD_loss = self._distill_one_epoch(epoch)
            else:
                train_score, train_loss = self._train_one_epoch(epoch)

            # Test in three dataset
            if self.trainer_params['distillation'] and self.trainer_params['distill_param']['adaptive_prob']:
                if epoch ==1 or (epoch % self.trainer_params['distill_param']['adaptive_interval'])==0:
                    is_test = True # test every I epoch to get the gap for adaptive prob
            elif self.trainer_params['multi_test'] and (epoch % 50) == 0 :
                    is_test = True # test every 50 epoch if the adaptive prob is not needed
            else:
                is_test = False
            # is_test = True
            if is_test:
                if self.trainer_params['distillation']:
                    val_model, val_env = self.student_model,self.student_env
                else:
                    val_model, val_env = self.model, self.env
                val_no_aug, val_aug, gap_no_aug, gap_aug  = [], [], [], []
                g = 0
                for k, v in self.trainer_params['val_dataset_multi'].items():
                    # torch.cuda.synchronize()
                    # tik = time.time()
                    no_aug, aug = validate(model=val_model, env=val_env,
                                           batch_size=self.trainer_params['val_batch_size'],
                                           augment=True, load_path=v)
                    # print(time.time()-tik)
                    val_no_aug.append(no_aug)
                    val_aug.append(aug)
                    gap_no_aug.append((no_aug - self.trainer_params['LKH3_optimal'][g]) / self.trainer_params['LKH3_optimal'][g])
                    gap_aug.append((aug - self.trainer_params['LKH3_optimal'][g]) / self.trainer_params['LKH3_optimal'][g])
                    g += 1

            # Calculate the adaptive prob
            if self.trainer_params['distillation'] and self.trainer_params['distill_param']['adaptive_prob']:
                gap = gap_aug if self.trainer_params['distill_param']['aug_gap'] else gap_no_aug
                if any(map(lambda x: x < 0, gap)):
                    self.trainer_params['distill_param']['teacher_prob'] = [1/3, 1/3, 1/3]
                    self.logger.info('Gap is negative!')
                else:
                    if self.trainer_params['distill_param']['adaptive_prob_type'] == 'softmax':
                        self.trainer_params['distill_param']['teacher_prob'] = softmax(gap)
                    elif self.trainer_params['distill_param']['adaptive_prob_type'] == 'sum':
                        self.trainer_params['distill_param']['teacher_prob'] = [gap[i] / sum(gap) for i in range(len(gap))]

            # log
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)
            if self.trainer_params['distillation']:
                self.result_log.append('RL_loss', epoch, RL_loss)
                self.result_log.append('KLD_loss', epoch, KLD_loss)
                if self.trainer_params['distill_param']['distill_distribution']:
                    self.result_log.append(class_type + '_train_score', epoch, train_score)
                    self.result_log.append(class_type + '_train_loss', epoch, train_loss)
                    self.result_log.append(class_type + '_RL_loss', epoch, RL_loss)
                    self.result_log.append(class_type + '_KLD_loss', epoch, KLD_loss)
                    self.result_log.append('class_type', epoch, class_type)
            if is_test:
                note  = ['uniform', 'cluster', 'mixed']
                for i in range(3):
                    self.result_log.append(note[i] + '_val_score_noAUG', epoch, val_no_aug[i])
                    self.result_log.append(note[i] + '_val_score_AUG', epoch, val_aug[i])
                    self.result_log.append(note[i] + '_val_gap_noAUG', epoch, gap_no_aug[i])
                    self.result_log.append(note[i] + '_val_gap_AUG', epoch, gap_aug[i])
                self.result_log.append('val_gap_AUG_mean', epoch, np.mean(gap_aug))
                self.result_log.append('val_gap_noAUG_mean', epoch, np.mean(gap_no_aug))


            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))
            if self.trainer_params['logging']['tb_logger']:
                self.tb_logger.log_value('train_score', train_score, epoch)
                self.tb_logger.log_value('train_loss', train_loss, epoch)
                if self.trainer_params['distillation']:
                    self.tb_logger.log_value('RL_loss', RL_loss, epoch)
                    self.tb_logger.log_value('KLD_loss', KLD_loss, epoch)
                    if self.trainer_params['distill_param']['distill_distribution']:
                        self.tb_logger.log_value(class_type + '_train_score', train_score,
                                                 self.trainer_params['distill_param']['count'][class_type])
                        self.tb_logger.log_value(class_type + '_train_loss', train_loss,
                                                 self.trainer_params['distill_param']['count'][class_type])
                        self.tb_logger.log_value(class_type + '_RL_loss', RL_loss,
                                                 self.trainer_params['distill_param']['count'][class_type])
                        self.tb_logger.log_value(class_type + '_KLD_loss', KLD_loss,
                                                 self.trainer_params['distill_param']['count'][class_type])
                if is_test:
                    note = ['uniform', 'cluster', 'mixed']
                    for i in range(3):
                        self.tb_logger.log_value(note[i] + '_val_score_noAUG', val_no_aug[i], epoch)
                        self.tb_logger.log_value(note[i] + '_val_score_AUG', val_aug[i], epoch)
                        self.tb_logger.log_value(note[i] + '_val_gap_noAUG', gap_no_aug[i], epoch)
                        self.tb_logger.log_value(note[i] + '_val_gap_AUG', gap_aug[i], epoch)
                    self.tb_logger.log_value('val_gap_AUG_mean', np.mean(gap_aug), epoch)
                    self.tb_logger.log_value('val_gap_noAUG_mean', np.mean(gap_no_aug), epoch)

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            # Save latest images, every epoch
            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])
                if self.trainer_params['distillation']:
                    util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                                   self.result_log, labels=['RL_loss'])
                    util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                                   self.result_log, labels=['KLD_loss'])
                    if self.trainer_params['distill_param']['distill_distribution']:
                        log_image_params = { 'json_foldername': 'log_image_style',
                                             'filename': 'style_cvrp_{}_{}.json'.format(self.env_params['problem_size'],class_type)}
                        util_save_log_image_with_label(image_prefix, log_image_params,
                                                       self.result_log, labels=[class_type + '_train_score'])
                        util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                                       self.result_log, labels=[class_type + '_train_loss'])
                        util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                                       self.result_log, labels=[class_type + '_RL_loss'])
                        util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                                       self.result_log, labels=[class_type + '_KLD_loss'])
                if is_test:
                    gap_image_params = {'json_foldername': 'log_image_style',
                                        'filename': 'style_gap.json'}
                    note = ['uniform', 'cluster', 'mixed']
                    for i in range(3):
                        log_image_params = { 'json_foldername': 'log_image_style',
                                             'filename': 'style_cvrp_{}_{}.json'.format(self.env_params['problem_size'],note[i])}
                        util_save_log_image_with_label(image_prefix, log_image_params, self.result_log, labels=[note[i] + '_val_score_noAUG'])
                        util_save_log_image_with_label(image_prefix, log_image_params, self.result_log, labels=[note[i] + '_val_score_AUG'])
                        util_save_log_image_with_label(image_prefix, gap_image_params, self.result_log, labels=[note[i] + '_val_gap_noAUG'])
                        util_save_log_image_with_label(image_prefix, gap_image_params, self.result_log, labels=[note[i] + '_val_gap_AUG'])
                    util_save_log_image_with_label(image_prefix, gap_image_params, self.result_log,labels=['val_gap_AUG_mean'])
                    util_save_log_image_with_label(image_prefix, gap_image_params, self.result_log,labels=['val_gap_noAUG_mean'])


            # Save Model
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                if self.trainer_params['distillation']:
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.student_model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'result_log': self.result_log.get_raw_data()
                    }
                else:
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'result_log': self.result_log.get_raw_data()
                    }

                if self.trainer_params['distillation'] and self.trainer_params['distill_param']['distill_distribution']:
                    torch.save(checkpoint_dict, '{}/checkpoint-{}-{}.pt'.format(self.result_folder, epoch,class_type))
                else:
                    torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            # Save Image
            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])
                if self.trainer_params['distillation']:
                    util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                                   self.result_log, labels=['RL_loss'])
                    util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                                   self.result_log, labels=['KLD_loss'])
                    if self.trainer_params['distill_param']['distill_distribution']:
                        log_image_params = { 'json_foldername': 'log_image_style',
                                             'filename': 'style_cvrp_{}_{}.json'.format(self.env_params['problem_size'],class_type)}
                        util_save_log_image_with_label(image_prefix, log_image_params,
                                                       self.result_log, labels=[class_type + '_train_score'])
                        util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                                       self.result_log, labels=[class_type + '_train_loss'])
                        util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                                       self.result_log, labels=[class_type + '_RL_loss'])
                        util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                                       self.result_log, labels=[class_type + '_KLD_loss'])
                if is_test:
                    gap_image_params = {'json_foldername': 'log_image_style',
                                        'filename': 'style_gap.json'}
                    note = ['uniform', 'cluster', 'mixed']
                    for i in range(3):
                        log_image_params = { 'json_foldername': 'log_image_style',
                                             'filename': 'style_cvrp_{}_{}.json'.format(self.env_params['problem_size'],note[i])}
                        util_save_log_image_with_label(image_prefix, log_image_params, self.result_log, labels=[note[i] + '_val_score_noAUG'])
                        util_save_log_image_with_label(image_prefix, log_image_params, self.result_log, labels=[note[i] + '_val_score_AUG'])
                        util_save_log_image_with_label(image_prefix, gap_image_params, self.result_log, labels=[note[i] + '_val_gap_noAUG'])
                        util_save_log_image_with_label(image_prefix, gap_image_params, self.result_log, labels=[note[i] + '_val_gap_AUG'])
                    util_save_log_image_with_label(image_prefix, gap_image_params, self.result_log,labels=['val_gap_AUG_mean'])
                    util_save_log_image_with_label(image_prefix, gap_image_params, self.result_log,labels=['val_gap_noAUG_mean'])

            # logger
            if is_test:
                note = ['uniform', 'cluster', 'mixed']
                for i in range(3):
                    self.logger.info("Epoch {:3d}/{:3d} Validate in {}: Gap: noAUG[{:.3f}] AUG[{:.3f}]; Score: noAUG[{:.3f}] AUG[{:.3f}]".format(
                        epoch, self.trainer_params['epochs'],note[i], gap_no_aug[i], gap_aug[i],val_no_aug[i],val_aug[i]))
                self.logger.info("Epoch {:3d}/{:3d} Validate! mean Gap: noAUG[{:.3f}] AUG[{:.3f}]".format(epoch,
                        self.trainer_params['epochs'], np.mean(gap_no_aug)*100, np.mean(gap_aug)*100))
                if self.trainer_params['best']==0:
                    print(self.trainer_params['best'])
                    self.trainer_params['best'] = np.mean(gap_aug)*100
                elif  np.mean(gap_aug)*100 < self.trainer_params['best']:
                    self.trainer_params['best'] = np.mean(gap_aug) * 100
                    self.logger.info("Saving best trained_model")
                    if self.trainer_params['distillation']:
                        checkpoint_dict = {
                            'epoch': epoch,
                            'best_gap': np.mean(gap_aug) * 100,
                            'model_state_dict': self.student_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'result_log': self.result_log.get_raw_data()
                        }
                    else:
                        checkpoint_dict = {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'result_log': self.result_log.get_raw_data()
                        }

                    if self.trainer_params['distillation'] and self.trainer_params['distill_param']['distill_distribution']:
                        torch.save(checkpoint_dict,
                                   '{}/checkpoint-{}-best.pt'.format(self.result_folder,class_type))
                    else:
                        torch.save(checkpoint_dict, '{}/checkpoint-best.pt'.format(self.result_folder))


            # All-done announcement
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()

    def _distill_one_epoch(self, epoch, teacher_prob = 0):
        distill_param = self.trainer_params['distill_param']
        self.logger.info("Start train student model epoch {}".format(epoch))

        # init
        if distill_param['distill_distribution']:
            uniform_score_AM, uniform_loss_AM, uniform_RL_loss_AM, uniform_KLD_loss_AM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            cluster_score_AM, cluster_loss_AM, cluster_RL_loss_AM, cluster_KLD_loss_AM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            mixed_score_AM, mixed_loss_AM, mixed_RL_loss_AM, mixed_KLD_loss_AM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        else:
            score_AM = AverageMeter()
            loss_AM = AverageMeter()
            RL_loss_AM = AverageMeter()
            KLD_loss_AM = AverageMeter()

        # load the teacher model
        if distill_param['multi_teacher'] and distill_param['distill_distribution']: # multi teacher
            for i in ['uniform', 'cluster', 'mixed']:
                load_path = self.trainer_params['model_load']['load_path_multi'][i]
                self.logger.info(' [*] Loading model from {}'.format(load_path))
                checkpoint = torch.load(load_path, map_location=self.device)
                self.model[i].load_state_dict(checkpoint['model_state_dict'])
                if distill_param['adaptive_prob'] and epoch > distill_param['start_adaptive_epoch']:  # adaptive prob based on the gap
                    class_type = np.random.choice(['uniform', 'cluster', 'mixed'], size=1,p=distill_param['teacher_prob'])
                else:  # equal prob
                    class_type = np.random.choice(['uniform', 'cluster', 'mixed'], 1)
        elif distill_param['distill_distribution']: # randomly choose a teacher
            if distill_param['adaptive_prob'] and epoch > distill_param['start_adaptive_epoch']: # adaptive prob based on the gap
                class_type = np.random.choice(['uniform', 'cluster', 'mixed'], size=1, p=distill_param['teacher_prob'])
                load_path = self.trainer_params['model_load']['load_path_multi'][class_type.item()]
                self.logger.info(' [*] Loading model from {}, prob: {}'.format(load_path,distill_param['teacher_prob']))
            else: # equal prob
                class_type = np.random.choice(['uniform', 'cluster', 'mixed'], 1)
                load_path = self.trainer_params['model_load']['load_path_multi'][class_type.item()]
                self.logger.info(' [*] Loading model from {}'.format(load_path))
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else: # distill one teacher
            load_path = self.trainer_params['model_load']['path']
            self.logger.info(' [*] Loading model from {}'.format(load_path))
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            class_type = None if not distill_param['distill_distribution'] else class_type
            if not isinstance(class_type, str) and class_type is not None:
                class_type = class_type.item()

            avg_score, avg_loss, RL_loss, KLD_loss  = self._distill_one_batch(batch_size, distribution=class_type)

            # update variables
            if distill_param['distill_distribution']:
                locals()[class_type + '_score_AM'].update(avg_score, batch_size)
                locals()[class_type + '_loss_AM'].update(avg_loss, batch_size)
                locals()[class_type + '_RL_loss_AM'].update(RL_loss, batch_size)
                locals()[class_type + '_KLD_loss_AM'].update(KLD_loss, batch_size)
            else:
                score_AM.update(avg_score, batch_size)
                loss_AM.update(avg_loss, batch_size)
                RL_loss_AM.update(RL_loss, batch_size)
                KLD_loss_AM.update(KLD_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    if distill_param['distill_distribution']:
                        self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f},  RL_Loss: {:.4f},  KLD_Loss: {:.4f}'
                                         .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                                 locals()[class_type + '_score_AM'].avg, locals()[class_type + '_loss_AM'].avg,
                                                 locals()[class_type + '_RL_loss_AM'].avg, locals()[class_type + '_KLD_loss_AM'].avg))
                    else:
                        self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f},  RL_Loss: {:.4f},  KLD_Loss: {:.4f}'
                                         .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                                 score_AM.avg, loss_AM.avg, RL_loss_AM.avg, KLD_loss_AM.avg))

        torch.cuda.empty_cache()
        # Log Once, for each epoch
        if distill_param['distill_distribution']:
            self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  RL_Loss: {:.4f},  KLD_Loss: {:.4f}'
                             .format(epoch, 100. * episode / train_num_episode,
                                     locals()[class_type + '_score_AM'].avg,
                                     locals()[class_type + '_loss_AM'].avg,
                                     locals()[class_type + '_RL_loss_AM'].avg,
                                     locals()[class_type + '_KLD_loss_AM'].avg))
            torch.cuda.empty_cache()
            return locals()[class_type + '_score_AM'].avg, locals()[class_type + '_loss_AM'].avg,\
                   locals()[class_type + '_RL_loss_AM'].avg, locals()[class_type + '_KLD_loss_AM'].avg, class_type
        else:
            self.logger.info(
                'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  RL_Loss: {:.4f},  KLD_Loss: {:.4f}'
                .format(epoch, 100. * episode / train_num_episode,
                        score_AM.avg, loss_AM.avg, RL_loss_AM.avg, KLD_loss_AM.avg))
            torch.cuda.empty_cache()
            return score_AM.avg, loss_AM.avg, RL_loss_AM.avg, KLD_loss_AM.avg

    def _distill_one_batch(self, batch_size,distribution=None):
        distill_param = self.trainer_params['distill_param']
        # Prep
        ###############################################
        self.student_model.train()
        if distill_param['multi_teacher'] and distill_param['distill_distribution']:
            for i in ['uniform', 'cluster', 'mixed']:
                self.model[i].eval()
        else:
            self.model.eval()

        self.env.load_problems(batch_size, distribution=distribution)
        depot_xy = self.env.reset_state.depot_xy
        node_xy = self.env.reset_state.node_xy
        node_demand = self.env.reset_state.node_demand
        self.student_env.load_problems(batch_size, copy=[depot_xy, node_xy, node_demand])

        # total_num = sum(p.numel() for p in self.student_model.parameters())
        # trainable_num = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
        # print({'Total': total_num, 'Trainable': trainable_num})
        # total_num = sum(p.numel() for p in self.model.parameters())
        # trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print({'Total': total_num, 'Trainable': trainable_num})

        if distill_param['router'] == 'teacher':# Teacher as the router

            # Teacher
            with torch.no_grad():
                reset_state, _, _ = self.env.reset()
                self.model.pre_forward(reset_state, attn_type=None)  # No return!

                teacher_prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))# shape: (batch, pomo, 0~problem)
                # POMO Rollout
                ###############################################
                state, reward, done = self.env.pre_step()
                teacher_pi = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
                teacher_probs = torch.zeros(size=(batch_size, self.env.pomo_size, self.env.problem_size + 1, 0))
                # Decoding
                while not done:
                    selected, prob, probs = self.model(state, return_probs=True, teacher=True)
                    # shape: (batch, pomo)
                    state, reward, done = self.env.step(selected)
                    teacher_prob_list = torch.cat((teacher_prob_list, prob[:, :, None]), dim=2)
                    teacher_pi = torch.cat((teacher_pi, selected[:, :, None]), dim=2)
                    teacher_probs = torch.cat((teacher_probs, probs[:, :, :, None]), dim=3)
                teacher_probs = teacher_probs + 0.00001 # avoid log0

            # Student
            student_reset_state, _, _ = self.student_env.reset()
            self.student_model.pre_forward(student_reset_state, attn_type=None) # No return!

            student_prob_list = torch.zeros(size=(batch_size, self.student_env.pomo_size, 0)) # shape: (batch, pomo, 0~problem)
            # POMO Rollout
            ###############################################
            student_state, student_reward, student_done = self.student_env.pre_step()
            student_pi = torch.zeros(size=(batch_size, self.student_env.pomo_size, 0))
            student_probs = torch.zeros(size=(batch_size, self.student_env.pomo_size, self.student_env.problem_size+1, 0))
            # Decoding
            while not student_done:
                student_selected, student_prob, probs = self.student_model(student_state, route=teacher_pi, return_probs=True)
                # shape: (batch, pomo)
                student_state, student_reward, student_done = self.student_env.step(student_selected)
                student_prob_list = torch.cat((student_prob_list, student_prob[:, :, None]), dim=2)
                student_pi = torch.cat((student_pi, student_selected[:, :, None]), dim=2)
                student_probs = torch.cat((student_probs, probs[:, :, :, None]), dim=3)
            student_probs = student_probs + 0.00001 # avoid log0

        else:# Student as the router
            if self.trainer_params['distill_param']['multi_teacher']:
                student_reset_state, _, _ = self.student_env.reset()
                self.student_model.pre_forward(student_reset_state, attn_type=None)  # No return!

                student_prob_list = torch.zeros(
                    size=(batch_size, self.env.pomo_size, 0))  # shape: (batch, pomo, 0~problem)
                # POMO Rollout
                ###############################################
                student_state, student_reward, student_done = self.student_env.pre_step()
                student_pi = torch.zeros(size=(batch_size, self.student_env.pomo_size, 0))
                student_probs = torch.zeros(
                    size=(batch_size, self.student_env.pomo_size, self.student_env.problem_size + 1, 0))
                # Decoding
                while not student_done:
                    student_selected, student_prob, probs = self.student_model(student_state, return_probs=True)
                    # shape: (batch, pomo)
                    student_state, student_reward, student_done = self.student_env.step(student_selected)
                    student_prob_list = torch.cat((student_prob_list, student_prob[:, :, None]), dim=2)
                    student_pi = torch.cat((student_pi, student_selected[:, :, None]), dim=2)
                    student_probs = torch.cat((student_probs, probs[:, :, :, None]), dim=3)
                # if not distill_param['KLD_student_to_teacher']:
                student_probs = student_probs + 0.00001  # avoid log0

                # Teacher follow the route of student (multi)
                teacher_probs_multi=[]
                for i in ['uniform', 'cluster', 'mixed']:
                    with torch.no_grad():
                        reset_state, _, _ = self.env.reset()
                        self.model[i].pre_forward(reset_state, attn_type=None)  # No return!

                        teacher_prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))  # shape: (batch, pomo, 0~problem)
                        # POMO Rollout
                        ###############################################
                        state, reward, done = self.env.pre_step()
                        teacher_pi = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
                        teacher_probs = torch.zeros(size=(batch_size, self.env.pomo_size, self.env.problem_size + 1, 0))

                        # Decoding
                        while not done:
                            selected, prob, probs = self.model[i](state, route=student_pi, return_probs=True)
                            # shape: (batch, pomo)
                            state, reward, done = self.env.step(selected)
                            teacher_prob_list = torch.cat((teacher_prob_list, prob[:, :, None]), dim=2)
                            teacher_pi = torch.cat((teacher_pi, selected[:, :, None]), dim=2)
                            teacher_probs = torch.cat((teacher_probs, probs[:, :, :, None]), dim=3)
                        # if distill_param['KLD_student_to_teacher']:
                        teacher_probs = teacher_probs + 0.00001  # avoid log0
                        teacher_probs_multi.append(teacher_probs)


            else:
                # Student
                student_reset_state, _, _ = self.student_env.reset()
                self.student_model.pre_forward(student_reset_state, attn_type=None)  # No return!

                student_prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))  # shape: (batch, pomo, 0~problem)
                # POMO Rollout
                ###############################################
                student_state, student_reward, student_done = self.student_env.pre_step()
                student_pi = torch.zeros(size=(batch_size, self.student_env.pomo_size, 0))
                student_probs = torch.zeros(
                    size=(batch_size, self.student_env.pomo_size, self.student_env.problem_size + 1, 0))
                # Decoding
                while not student_done:
                    student_selected, student_prob, probs = self.student_model(student_state,return_probs=True)
                    # shape: (batch, pomo)
                    student_state, student_reward, student_done = self.student_env.step(student_selected)
                    student_prob_list = torch.cat((student_prob_list, student_prob[:, :, None]), dim=2)
                    student_pi = torch.cat((student_pi, student_selected[:, :, None]), dim=2)
                    student_probs = torch.cat((student_probs, probs[:, :, :, None]), dim=3)
                # if not distill_param['KLD_student_to_teacher']:
                student_probs = student_probs + 0.00001  # avoid log0

                # Teacher follow the route of student
                with torch.no_grad():
                    reset_state, _, _ = self.env.reset()
                    self.model.pre_forward(reset_state, attn_type=None)  # No return!

                    teacher_prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))  # shape: (batch, pomo, 0~problem)
                    # POMO Rollout
                    ###############################################
                    state, reward, done = self.env.pre_step()
                    teacher_pi = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
                    teacher_probs = torch.zeros(size=(batch_size, self.env.pomo_size, self.env.problem_size + 1, 0))
                    # Decoding
                    while not done:
                        selected, prob, probs = self.model(state, route=student_pi, return_probs=True)
                        # shape: (batch, pomo)
                        state, reward, done = self.env.step(selected)
                        teacher_prob_list = torch.cat((teacher_prob_list, prob[:, :, None]), dim=2)
                        teacher_pi = torch.cat((teacher_pi, selected[:, :, None]), dim=2)
                        teacher_probs = torch.cat((teacher_probs, probs[:, :, :, None]), dim=3)
                    teacher_probs = teacher_probs + 0.00001  # avoid log0

        assert torch.equal(teacher_pi, student_pi), "Teacher route and student route are not the same!"

        # Loss for student model
        ###############################################
        advantage = student_reward - student_reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = student_prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        task_loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        task_loss = task_loss.mean()

        if distill_param['meaningful_KLD']:
            if distill_param['multi_teacher']:
                for i in range(len(teacher_probs_multi)):
                    if i == 0:
                        soft_loss = (student_probs * (student_probs.log() - teacher_probs_multi[i].log())).sum(dim=2).mean() if distill_param['KLD_student_to_teacher'] \
                            else (teacher_probs_multi[i] * (teacher_probs_multi[i].log() - student_probs.log())).sum(dim=2).mean()
                    else:
                        soft_loss = soft_loss + (student_probs * (student_probs.log() - teacher_probs_multi[i].log())).sum(dim=2).mean() if distill_param['KLD_student_to_teacher'] \
                            else (teacher_probs_multi[i] * (teacher_probs_multi[i].log() - student_probs.log())).sum(dim=2).mean()
                soft_loss = soft_loss / 3
            else:
                soft_loss = (student_probs * (student_probs.log() - teacher_probs.log())).sum(dim=2).mean() if \
                distill_param['KLD_student_to_teacher'] \
                    else (teacher_probs * (teacher_probs.log() - student_probs.log())).sum(dim=2).mean()
        else:
            soft_loss = nn.KLDivLoss()(student_probs.log(), teacher_probs) if not distill_param['KLD_student_to_teacher'] \
                else nn.KLDivLoss()(teacher_probs.log(), student_probs)
        loss = task_loss * distill_param['rl_alpha'] + soft_loss * distill_param['distill_alpha']

        # Score
        ###############################################
        max_pomo_reward, _ = student_reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################---==
        self.student_model.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()

        return score_mean.item(), loss.item(), task_loss.item(), soft_loss.item()

