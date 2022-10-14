import argparse
import os
import numpy as np
import torch
from utils.data_utils import check_extension, save_dataset


def generate_tsp_data(dataset_size, tsp_size, distribution,n_cluster=None,n_cluster_mix=None):
    if distribution == 'uniform':
        data = np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()
    elif distribution == 'cluster':
        center = np.array([list(np.random.rand(n_cluster * 2)) for _ in range(dataset_size)])
        lower, upper = 0.2, 0.8
        center = lower + (upper - lower) * center
        data = []
        for j in range(dataset_size):
            std = 0.07
            mean_x, mean_y = center[j, ::2], center[j, 1::2]
            coords = torch.zeros(tsp_size, 2)
            for i in range(n_cluster):
                if i < n_cluster - 1:
                    coords[int(tsp_size / n_cluster) * i:int(tsp_size / n_cluster) * (i + 1)] = \
                        torch.cat((torch.FloatTensor(int(tsp_size/ n_cluster), 1).normal_(mean_x[i], std),
                                   torch.FloatTensor(int(tsp_size / n_cluster), 1).normal_(mean_y[i], std)), dim=1)
                elif i == n_cluster - 1:
                    coords[int(tsp_size / n_cluster) * i:] = \
                        torch.cat(
                            (torch.FloatTensor(tsp_size - int(tsp_size / n_cluster) * i, 1).normal_(mean_x[i], std),
                             torch.FloatTensor(tsp_size - int(tsp_size/ n_cluster) * i, 1).normal_(mean_y[i],std)), dim=1)
            coords = torch.where(coords > 1, torch.ones_like(coords), coords)
            coords = torch.where(coords < 0, torch.zeros_like(coords), coords)
            data.append(tuple(coords.numpy().tolist()))
    elif distribution == 'mixed':
        center = np.array([list(np.random.rand(n_cluster_mix * 2)) for _ in range(dataset_size)])
        lower, upper = 0.2, 0.8
        center = lower + (upper - lower) * center
        data = []
        for j in range(dataset_size):
            std = 0.07
            mean_x, mean_y = center[j, ::2], center[j, 1::2]
            mutate_idx = np.random.choice(range(tsp_size), int(tsp_size / 2), replace=False)
            coords = torch.FloatTensor(tsp_size, 2).uniform_(0, 1)
            for i in range(n_cluster_mix):
                if i < n_cluster_mix - 1:
                    coords[mutate_idx[int(tsp_size / n_cluster_mix / 2) * i:int(tsp_size / n_cluster_mix / 2) * (i + 1)]] = \
                        torch.cat((torch.FloatTensor(int(tsp_size / n_cluster_mix / 2), 1).normal_(mean_x[i], std),
                                   torch.FloatTensor(int(tsp_size / n_cluster_mix / 2), 1).normal_(mean_y[i], std)), dim=1)
                elif i == n_cluster_mix - 1:
                    coords[mutate_idx[int(tsp_size / n_cluster_mix / 2) * i:]] = \
                        torch.cat((torch.FloatTensor(int(tsp_size / 2) - int(tsp_size / n_cluster_mix / 2) * i, 1).normal_(mean_x[i], std),
                                   torch.FloatTensor(int(tsp_size / 2) - int(tsp_size / n_cluster_mix / 2) * i, 1).normal_(mean_y[i], std)), dim=1)
            coords = torch.where(coords > 1, torch.ones_like(coords), coords)
            coords = torch.where(coords < 0, torch.zeros_like(coords), coords)
            data.append(tuple(coords.numpy().tolist()))
    return data


def generate_vrp_data(dataset_size, vrp_size, distribution,n_cluster=None,n_cluster_mix=None):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.,
    }
    if distribution == 'uniform':
        data = list(zip(
            np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
            np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
            np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
            # np.full((dataset_size, vrp_size) , 5).tolist(),
            np.full(dataset_size, CAPACITIES[vrp_size]).tolist()  # Capacity, same for whole dataset
        ))
    elif distribution == 'cluster':
        center = np.array( [list(np.random.rand(n_cluster * 2)) for _ in range(dataset_size)])
        lower, upper = 0.2, 0.8
        center = lower + (upper - lower) * center
        data = []
        for j in range(dataset_size):
            std = 0.07
            mean_x, mean_y = center[j, ::2], center[j, 1::2]
            coords = torch.zeros(vrp_size + 1, 2)
            for i in range(n_cluster):
                if i < n_cluster - 1:
                    coords[int((vrp_size + 1) / n_cluster) * i:int((vrp_size + 1) / n_cluster) * (i + 1)] = \
                        torch.cat((torch.FloatTensor(int((vrp_size + 1) / n_cluster), 1).normal_(mean_x[i], std),
                                   torch.FloatTensor(int((vrp_size + 1) / n_cluster), 1).normal_(mean_y[i], std)), dim=1)
                elif i == n_cluster - 1:
                    coords[int((vrp_size + 1) / n_cluster) * i:] = \
                        torch.cat(
                            (torch.FloatTensor((vrp_size + 1) - int((vrp_size + 1) / n_cluster) * i, 1).normal_(mean_x[i], std),
                             torch.FloatTensor((vrp_size + 1) - int((vrp_size + 1) / n_cluster) * i, 1).normal_(mean_y[i],std)), dim=1)
            coords = torch.where(coords > 1, torch.ones_like(coords), coords)
            coords = torch.where(coords < 0, torch.zeros_like(coords), coords)
            depot_idx = int(np.random.choice(range(coords.shape[0]), 1))
            data.append(tuple([
                coords[depot_idx].numpy().tolist(),
                coords[torch.arange(coords.size(0)) != depot_idx].numpy().tolist(),
                np.random.randint(1, 10, size=(vrp_size)).tolist(),
                CAPACITIES[vrp_size]
            ]))
    elif distribution == 'mixed':
        center = np.array([list(np.random.rand(n_cluster_mix * 2)) for _ in range(dataset_size)])
        lower, upper = 0.2, 0.8
        center = lower + (upper - lower) * center
        data = []
        for j in range(dataset_size):
            std = 0.07
            mean_x, mean_y = center[j, ::2], center[j, 1::2]
            mutate_idx = np.random.choice(range(vrp_size), int(vrp_size / 2), replace=False)
            coords = torch.FloatTensor(vrp_size, 2).uniform_(0, 1)
            for i in range(n_cluster_mix):
                if i < n_cluster_mix - 1:
                    coords[mutate_idx[int(vrp_size / n_cluster_mix / 2) * i:int(vrp_size / n_cluster_mix / 2) * (i + 1)]] = \
                        torch.cat((torch.FloatTensor(int(vrp_size / n_cluster_mix / 2), 1).normal_(mean_x[i], std),
                                   torch.FloatTensor(int(vrp_size / n_cluster_mix / 2), 1).normal_(mean_y[i], std)), dim=1)
                elif i == n_cluster_mix - 1:
                    coords[mutate_idx[int(vrp_size / n_cluster_mix / 2) * i:]] = \
                        torch.cat((torch.FloatTensor(int(vrp_size / 2) - int(vrp_size / n_cluster_mix / 2) * i, 1).normal_(mean_x[i], std),
                                   torch.FloatTensor(int(vrp_size / 2) - int(vrp_size / n_cluster_mix / 2) * i, 1).normal_(mean_y[i], std)), dim=1)
            coords = torch.where(coords > 1, torch.ones_like(coords), coords)
            coords = torch.where(coords < 0, torch.zeros_like(coords), coords)
            data.append(tuple([
                np.random.uniform(size=2).tolist(),
                coords.numpy().tolist(),
                np.random.randint(1, 10, size=(vrp_size)).tolist(),
                CAPACITIES[vrp_size]
            ]))
    return data


def generate_op_data(dataset_size, op_size, prize_type='const'):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, op_size, 2))

    # Methods taken from Fischetti et al. 1998
    if prize_type == 'const':
        prize = np.ones((dataset_size, op_size))
    elif prize_type == 'unif':
        prize = (1 + np.random.randint(0, 100, size=(dataset_size, op_size))) / 100.
    else:  # Based on distance to depot
        assert prize_type == 'dist'
        prize_ = np.linalg.norm(depot[:, None, :] - loc, axis=-1)
        prize = (1 + (prize_ / prize_.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.

    # Max length is approximately half of optimal TSP tour, such that half (a bit more) of the nodes can be visited
    # which is maximally difficult as this has the largest number of possibilities
    MAX_LENGTHS = {
        20: 2.,
        50: 3.,
        100: 4.
    }

    return list(zip(
        depot.tolist(),
        loc.tolist(),
        prize.tolist(),
        np.full(dataset_size, MAX_LENGTHS[op_size]).tolist()  # Capacity, same for whole dataset
    ))


def generate_pctsp_data(dataset_size, pctsp_size, penalty_factor=3):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, pctsp_size, 2))

    # For the penalty to make sense it should be not too large (in which case all nodes will be visited) nor too small
    # so we want the objective term to be approximately equal to the length of the tour, which we estimate with half
    # of the nodes by half of the tour length (which is very rough but similar to op)
    # This means that the sum of penalties for all nodes will be approximately equal to the tour length (on average)
    # The expected total (uniform) penalty of half of the nodes (since approx half will be visited by the constraint)
    # is (n / 2) / 2 = n / 4 so divide by this means multiply by 4 / n,
    # However instead of 4 we use penalty_factor (3 works well) so we can make them larger or smaller
    MAX_LENGTHS = {
        20: 2.,
        50: 3.,
        100: 4.
    }
    penalty_max = MAX_LENGTHS[pctsp_size] * (penalty_factor) / float(pctsp_size)
    penalty = np.random.uniform(size=(dataset_size, pctsp_size)) * penalty_max

    # Take uniform prizes
    # Now expectation is 0.5 so expected total prize is n / 2, we want to force to visit approximately half of the nodes
    # so the constraint will be that total prize >= (n / 2) / 2 = n / 4
    # equivalently, we divide all prizes by n / 4 and the total prize should be >= 1
    deterministic_prize = np.random.uniform(size=(dataset_size, pctsp_size)) * 4 / float(pctsp_size)

    # In the deterministic setting, the stochastic_prize is not used and the deterministic prize is known
    # In the stochastic setting, the deterministic prize is the expected prize and is known up front but the
    # stochastic prize is only revealed once the node is visited
    # Stochastic prize is between (0, 2 * expected_prize) such that E(stochastic prize) = E(deterministic_prize)
    stochastic_prize = np.random.uniform(size=(dataset_size, pctsp_size)) * deterministic_prize * 2

    return list(zip(
        depot.tolist(),
        loc.tolist(),
        penalty.tolist(),
        deterministic_prize.tolist(),
        stochastic_prize.tolist()
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=False, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='tsp',
                        help="Problem, 'tsp', 'vrp', 'pctsp' or 'op_const', 'op_unif' or 'op_dist'"
                             " or 'all' to generate all")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")
    parser.add_argument("--dataset_size", type=int, default=2000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[50],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")
    parser.add_argument('--n_cluster', type=int, default=3, help='n_cluster for cluster distribution')
    parser.add_argument('--n_cluster_mix', type=int, default=1, help='n_cluster for mixed cluster distribution')

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    distributions_per_problem = {
        'tsp': ['uniform','cluster','mixed'],
        'vrp': ['uniform','cluster','mixed'],
        # 'vrp': [ 'uniform'],
        'pctsp': [None],
        'op': ['const', 'unif', 'dist']
    }
    if opts.problem == 'all':
        problems = distributions_per_problem
    else:
        problems = {
            opts.problem:
                distributions_per_problem[opts.problem]
                if opts.data_distribution == 'all'
                else [opts.data_distribution]
        }

    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            for graph_size in opts.graph_sizes:

                datadir = os.path.join(opts.data_dir, problem)
                os.makedirs(datadir, exist_ok=True)

                if opts.filename is None:
                    file = "{}{}{}_{}_seed{}.pkl"
                    # file = 'cvrp2000_demand5_capacity50.pkl'
                    filename = os.path.join(datadir, file.format(
                        problem,
                        "_{}".format(distribution) if distribution is not None else "",
                        graph_size, opts.dataset_size, opts.seed))
                else:
                    filename = check_extension(opts.filename)

                assert opts.f or not os.path.isfile(check_extension(filename)), \
                    "File already exists! Try running with -f option to overwrite."

                torch.manual_seed(opts.seed)
                torch.cuda.manual_seed_all(opts.seed)
                np.random.seed(opts.seed)
                if problem == 'tsp':
                    dataset = generate_tsp_data(opts.dataset_size, graph_size, distribution, n_cluster=opts.n_cluster, n_cluster_mix=opts.n_cluster_mix)
                elif problem == 'vrp':
                    dataset = generate_vrp_data(opts.dataset_size, graph_size,distribution,n_cluster=opts.n_cluster, n_cluster_mix=opts.n_cluster_mix)
                elif problem == 'pctsp':
                    dataset = generate_pctsp_data(opts.dataset_size, graph_size)
                elif problem == "op":
                    dataset = generate_op_data(opts.dataset_size, graph_size, prize_type=distribution)
                else:
                    assert False, "Unknown problem: {}".format(problem)

                print(dataset[0])

                save_dataset(dataset, filename)
