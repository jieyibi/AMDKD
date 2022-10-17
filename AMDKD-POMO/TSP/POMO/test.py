##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 1


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSPTester import TSPTester as Tester


##########################################################################################
# parameters

env_params = {
    'problem_size': 50,
    'pomo_size': 50,
    'distribution': {
        'data_type': 'uniform',  # cluster, mixed, uniform, mix_three
        'n_cluster': 3,
        'n_cluster_mix': 1,
        'lower': 0.2,
        'upper': 0.8,
        'std': 0.07,
        'use_LHS': False,
        'centroid_file': None
    },
    'load_path': '../../../data/tsp/tsp_uniform50_10000.pkl',
    'load_raw': None

}

model_params = {
    'embedding_dim': 64,
    'sqrt_embedding_dim': 64**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 8,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': 'pretrained/checkpoint-tsp-50.pt',  # directory path of pre-trained model and log files saved.
        'epoch': 'best',  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 10000,
    'test_batch_size': 2500,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 2500,
}

assert tester_params['test_episodes'] % tester_params['test_batch_size'] == 0, "Number of instances must be divisible by batch size!"
assert tester_params['test_episodes'] % tester_params['aug_batch_size'] == 0, "Number of instances must be divisible by batch size!"

if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test_tsp{}_{}_epoch{}'.format(env_params['problem_size'],tester_params['model_load']['path'].split('/')[-1],tester_params['model_load']['epoch']),
        'filename': 'run_log'
    }
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)
    # no_aug, aug ,time = tester.run()
    tester.run()

    copy_all_src(tester.result_folder)



def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
