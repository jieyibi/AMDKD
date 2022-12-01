import argparse
import datetime
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.cuda as cutorch
import warnings
warnings.filterwarnings("ignore")

from source.active_search import run_active_search
from source.cvrp.grouped_actors import ACTOR as CVRP_ACTOR
from source.cvrp.read_data import read_instance_pkl as CVRP_read_instance_pkl
from source.cvrp.read_data import read_instance_vrp
from source.cvrp.utilities import augment_and_repeat_episode_data as CVRP__augment_and_repeat_episode_data
from source.cvrp.utilities import get_episode_data as CVRP_get_episode_data
# from source.eas_emb import run_eas_emb
from source.eas_lay import run_eas_lay
# from source.eas_tab import run_eas_tab
from source.sampling import run_sampling
from source.tsp.grouped_actors import ACTOR as TSP_ACTOR
from source.tsp.read_data import read_instance_pkl as TSP_read_instance_pkl
from source.tsp.read_data import read_instance_tsp
from source.tsp.utilities import augment_and_repeat_episode_data as TSP_augment_and_repeat_episode_data
from source.tsp.utilities import get_episode_data as TSP_get_episode_data


def get_config():
    parser = argparse.ArgumentParser(description='Efficient Active Search')

    parser.add_argument('-problem', default="CVRP", type=str, help="TSP or CVRP")
    parser.add_argument('-method', default="eas_lay", type=str, help="sampling, eas-emb, eas-lay, or eas-tab")
    parser.add_argument('-model_path', default="", type=str, help="Path of the trained model weights")

    parser.add_argument('-instances_path', default="", type=str, help="Path of the instances")
    parser.add_argument('-nb_instances', default=100000, type=int,
                        help="Maximum number of instances that should be solved")
    parser.add_argument('-instances_offset', default=0, type=int)

    parser.add_argument('-round_distances', default=False, action='store_true',
                        help="Round distances to the nearest integer. Required to solve .vrp instances")

    parser.add_argument('-max_iter', default=10000, type=int, help="Maximum number of EAS iterations")
    parser.add_argument('-max_runtime', default=100000, type=int, help="Maximum runtime of EAS per batch in seconds")

    parser.add_argument('-batch_size', default=25, type=int)  # Set to 1 for single instance search
    parser.add_argument('-p_runs', default=1,
                        type=int)  # If batch_size is 1, set this to > 1 to do multiple runs for the instance in parallel

    # EAS-Emb and EAS-Lay parameters
    parser.add_argument('-param_lambda', default=0.013, type=float)
    parser.add_argument('-param_lr', default=0.0041, type=float)

    # EAS-Tab parameters
    parser.add_argument('-param_alpha', default=0.539, type=float)
    parser.add_argument('-param_sigma', default=9.55, type=float)

    parser.add_argument('-output_path', default="", type=str)

    config = parser.parse_args()
    config.max_scaler = None
    return config


def read_instance_data(config):
    logging.info(f"Reading in instances from {config.instances_path}")

    if config.instances_path.endswith(".pkl"):
        # Read in an instance file that has been created with
        # https://github.com/wouterkool/attention-learn-to-route/blob/master/generate_data.py

        if config.problem == "TSP":
            instance_data = TSP_read_instance_pkl(config.instances_path)
            instance_data = instance_data[config.instances_offset:config.instances_offset + config.nb_instances]
            problem_size = instance_data.shape[1]
            instance_data_scaled = (instance_data, None)

        elif config.problem == "CVRP":
            instance_data = CVRP_read_instance_pkl(config.instances_path)
            demand_scaler = instance_data[2]
            instance_data = (instance_data[0][config.instances_offset:config.instances_offset + config.nb_instances],
                             instance_data[1][config.instances_offset:config.instances_offset + config.nb_instances])
            problem_size = instance_data[0].shape[1] - 1

            # The vehicle capacity (here called demand_scaler) is hardcoded for these instances as follows
            # if problem_size == 20:
            #     demand_scaler = 30
            # elif problem_size == 50:
            #     demand_scaler = 40
            # elif problem_size == 100:
            #     demand_scaler = 50
            # elif problem_size == 125:
            #     demand_scaler = 55
            # elif problem_size == 150:
            #     demand_scaler = 60
            # elif problem_size == 200:
            #     demand_scaler = 70
            # else:
            #     raise NotImplementedError

            instance_data_scaled = instance_data[0], instance_data[1] / demand_scaler

    else:
        # Read in .vrp instance(s) that have the VRPLIB format. In this case the distances between customers
        # should be rounded.

        assert config.round_distances

        if config.instances_path.endswith(".vrp") or config.instances_path.endswith(".tsp"):
            # Read in a single instance
            instance_file_paths = [config.instances_path]
        elif os.path.isdir(config.instances_path):
            # or all instances in the given directory.
            instance_file_paths = [os.path.join(config.instances_path, f) for f in
                                   sorted(os.listdir(config.instances_path))]
            instance_file_paths = instance_file_paths[
                                  config.instances_offset:config.instances_offset + config.nb_instances]

        # Read in the first instance only to determine the problem_size
        if config.instances_path.endswith(".vrp"):
            _, locations, _, _ = read_instance_vrp(instance_file_paths[0])
            problem_size = locations.shape[1] - 1
            # Prepare empty numpy array to store instance data
            instance_data_scaled = (np.zeros((len(instance_file_paths), locations.shape[1], 2)),
                                    np.zeros((len(instance_file_paths), locations.shape[1] - 1)))

            # Read in all instances
            for idx, file in enumerate(instance_file_paths):
                # logging.info(f'Instance: {os.path.split(file)[-1]}')
                original_locations, locations, demand, capacity = read_instance_vrp(file)
                instance_data_scaled[0][idx], instance_data_scaled[1][idx] = locations, demand / capacity

        elif config.instances_path.endswith(".tsp"):
            _, locations, _ = read_instance_tsp(config.instances_path)
            problem_size =  locations.shape[1]
            # Prepare empty numpy array to store instance data
            instance_data_scaled = (np.zeros((len(instance_file_paths), locations.shape[1], 2)),None)
            # Read in all instances
            for idx, file in enumerate(instance_file_paths):
                # logging.info(f'Instance: {os.path.split(file)[-1]}')
                original_locations, locations, max_scaler = read_instance_tsp(file)
                instance_data_scaled[0][idx] = locations
                config.max_scaler = max_scaler

    return instance_data_scaled, problem_size


def search(run_id, config, tuning=False):
    # Creating output directories
    if config.output_path == "":
        config.output_path = os.getcwd()
    now = datetime.datetime.now()
    config.output_path = os.path.join(config.output_path, "runs", f"run_{now.day}.{now.month}.{now.year}_{run_id}")
    os.makedirs(os.path.join(config.output_path))

    # Create logger and log run parameters
    logging.basicConfig(
        filename=os.path.join(config.output_path, "log_" + str(run_id) + ".txt"), filemode='w',
        level=logging.INFO, format='[%(levelname)s]%(message)s')

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Call: {0}".format(' '.join(sys.argv)))

    # Load models
    if config.problem == "TSP":
        grouped_actor = TSP_ACTOR().cuda()
    elif config.problem == "CVRP":
        grouped_actor = CVRP_ACTOR().cuda()
    else:
        raise NotImplementedError("Unknown problem")
    grouped_actor.load_state_dict(torch.load(config.model_path, map_location="cuda")['model_state_dict'])
    grouped_actor.eval()

    instance_data_scaled, problem_size = read_instance_data(config)

    if config.problem == "TSP":
        get_episode_data_fn = TSP_get_episode_data
        augment_and_repeat_episode_data_fn = TSP_augment_and_repeat_episode_data
    elif config.problem == "CVRP":
        get_episode_data_fn = CVRP_get_episode_data
        augment_and_repeat_episode_data_fn = CVRP__augment_and_repeat_episode_data

    if config.method == "sampling":
        start_search_fn = run_sampling
    elif config.method.startswith("as"):
        start_search_fn = run_active_search
    elif config.method.startswith("eas-emb"):
        start_search_fn = run_eas_emb
    elif config.method.startswith("eas-lay"):
        start_search_fn = run_eas_lay
    elif config.method.startswith("eas-tab"):
        start_search_fn = run_eas_tab
    else:
        raise NotImplementedError("Unknown search method")

    if config.batch_size == 1:
        logging.info("Starting single instance search. 1 instance is solved per episode.")
    else:
        assert config.p_runs == 1
        logging.info(f"Starting batch search. {config.batch_size} instances are solved per episode.")

    # Run the actual search
    start_t = time.time()
    perf, best_solutions = start_search_fn(grouped_actor, instance_data_scaled, problem_size, config,
                                           get_episode_data_fn, augment_and_repeat_episode_data_fn)
    runtime = time.time() - start_t

    if config.problem == "CVRP" and not config.instances_path.endswith(".pkl"):
        # For instances with the CVRPLIB format the costs need to be adjusted to match the original coordinates
        perf = np.round(perf * 1000).astype('int')
    elif config.problem == "TSP" and not config.instances_path.endswith(".pkl"):
        perf = np.round(perf * config.max_scaler).astype('int')

    logging.info(f"Mean costs: {np.mean(perf)}")
    logging.info(f"Runtime: {runtime}")
    logging.info("MEM: " + str(cutorch.max_memory_cached(
        0) / 1024 / 1024))  # TODO update to torch.max_memory_reserved
    logging.info(f"Nb. instances: {len(perf)}")

    pickle.dump([runtime, perf],
                open(os.path.join(config.output_path, "results.pkl"), 'wb'))

    return np.mean(perf)


VERSION = "1.0.0"
if __name__ == '__main__':
    run_id = np.random.randint(10000, 99999)
    config = get_config()
    search(run_id, config)
