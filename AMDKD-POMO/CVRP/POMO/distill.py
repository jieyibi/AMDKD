##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 2

##########################################################################################
# Path Config

import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src
from CVRPTrainer import CVRPTrainer as Trainer


##########################################################################################
# parameters

env_params = {
    'problem_size': 50,
    'pomo_size': 50,
    'distribution': {
        'data_type': 'uniform',  # cluster, mixed, uniform
        'n_cluster': 3,
        'n_cluster_mix': 1,
        'lower': 0.2,
        'upper': 0.8,
        'std': 0.07,
    }
}
env_params['load_raw']=None
LKH3_optimal = {
    20: [6.156523, 3.05725599999999, 5.439149],
    50: [10.417558, 5.155511, 9.354149],#1000
    100: [15.740834, 7.909336, 14.294179] #1000
}

model_params = {
    'normalization':'instance',
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones':[29901,29951],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 30000,
    'train_episodes': 10* 1000,
    'train_batch_size': 512,
    'prev_model_path': None,
    'distillation': True,
    'multi_test': False,
    'logging': {
        'model_save_interval': 100,
        'img_save_interval': 100,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_cvrp_{}_{}.json'.format(env_params['problem_size'],env_params['distribution']['data_type'])
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
        'tb_logger': True
    },
    'model_load': {
        'enable': False,
        # 'path': '', #directory path of pre-trained model and log files saved.
        # 'epoch': 0,  # epoch version of pre-trained model to laod.
    },
    'best': 0
}

#################################################
# distillation params
if trainer_params['distillation']:
    trainer_params['student_model_param'] = {
        'normalization':'instance',
        'embedding_dim': 64,
        'sqrt_embedding_dim': 64 ** (1 / 2),
        'encoder_layer_num': 6,
        'qkv_dim': 8,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
    }
    trainer_params['distill_param'] =  {
        'distill_distribution':True,
        'router': 'student',
        'temperature': 1,
        'rl_alpha': 0.5,
        'distill_alpha': 0.5,
        'adaptive_prob':True,
        'teacher_prob': 0,
        'KLD_student_to_teacher': False,
        'meaningful_KLD':True,
        'adaptive_prob_type': 'softmax', #'sum'
        'start_adaptive_epoch': 1,
        'aug_gap':True,
        'adaptive_interval': 1,
        'multi_teacher':False,
        'hinton_t2': False,
        'attn_type': 'qk_scaled', # choose: no: no attn loss; qk_scaled; add_mask
     }

    if trainer_params['distill_param']['adaptive_prob']:
        trainer_params['multi_test'] = True

    if trainer_params['distill_param']['distill_distribution']:
        trainer_params['distill_param']['count'] = {
            'uniform': 0,
            'cluster': 0,
            'mixed': 0
        }
        trainer_params['model_load']['load_path_multi'] = {
            'uniform': './teacher/cvrp50-uniform-checkpoint.pt',
            'cluster': './teacher/cvrp50-cluster-checkpoint.pt',
            'mixed': './teacher/cvrp50-mixed-checkpoint.pt'
        }
    else:
        load_path_multi = {
            'uniform': './teacher/cvrp50-uniform-checkpoint.pt',
            'cluster': './teacher/cvrp50-cluster-checkpoint.pt',
            'mixed': './teacher/cvrp50-mixed-checkpoint.pt'
        }
        trainer_params['model_load']['path'] = load_path_multi[env_params['distribution']['data_type']]
        print(trainer_params['model_load']['path'])

if trainer_params['multi_test']:
    # uniform, cluster, mixed
    trainer_params['LKH3_optimal'] = LKH3_optimal[env_params['problem_size']]
    trainer_params['val_batch_size'] = 1000
    trainer_params['val_dataset_multi'] = {
        'uniform': './data/vrp_uniform{}_1000_seed1234.pkl'.format(env_params['problem_size']),
        'cluster': './data/vrp_cluster{}_1000_seed1234.pkl'.format(env_params['problem_size']),
        'mixed': './data/vrp_mixed{}_1000_seed1234.pkl'.format(env_params['problem_size'])
    }

###################################################
logger_params = {
    'log_file': {
        'desc': 'train_cvrp_n{}_epoch{}_{}_batchsize{}_{}Norm'.format(env_params['problem_size'],trainer_params['epochs'],
                                                      env_params['distribution']['data_type'],trainer_params['train_batch_size'],
                                                                      model_params['normalization']),
        'filename': 'run_log'
    }
}

if trainer_params['distillation']:
    if trainer_params['distill_param']['distill_distribution']:
        if trainer_params['distill_param']['adaptive_prob']:
            dir0 = 'distill_cvrp_n{}_epoch{}_router_{}_layer{}_dim{}_adaptive_{}_start{}_every{}epochs_{}Norm'.format(
                env_params['problem_size'], trainer_params['epochs'],
                trainer_params['distill_param']['router'], trainer_params['student_model_param']['encoder_layer_num'],
                trainer_params['student_model_param']['embedding_dim'],
                trainer_params['distill_param']['adaptive_prob_type'],
                trainer_params['distill_param']['start_adaptive_epoch'],
                trainer_params['distill_param']['adaptive_interval'],
                trainer_params['student_model_param']['normalization'])
            logger_params['log_file']['desc'] = dir0
        else:
            logger_params['log_file']['desc'] = 'distill_cvrp_n{}_epoch{}_router_{}_layer{}_dim{}_{}Norm'.format(env_params['problem_size'], trainer_params['epochs'],
                                                trainer_params['distill_param']['router'], trainer_params['student_model_param']['encoder_layer_num'],
                                                trainer_params['student_model_param']['embedding_dim'],trainer_params['student_model_param']['normalization'])
    else:
        logger_params['log_file']['desc'] = 'distill_cvrp_n{}_epoch{}_{}_{}Norm'.format(env_params['problem_size'],trainer_params['epochs'],
                                                          env_params['distribution']['data_type'],trainer_params['student_model_param']['normalization'])

trainer_params['tb_path'] = logger_params['log_file']['desc']
##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
