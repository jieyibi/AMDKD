from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search
import numpy as np

class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, n_cluster = 1, n_cluster_mix = 1, mix_data=False):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # # Sample points randomly in [0, 1] square
            # self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]
            if mix_data:
                self.data = []
                center = np.array([list(np.random.rand(n_cluster * 2)) for _ in range(num_samples)])
                lower, upper = 0.2, 0.8
                center = lower + (upper - lower) * center
                std = 0.07
                for j in range(num_samples):
                    class_type = np.random.choice(['uniform', 'cluster', 'mixed'], 1)
                    if class_type.item() == 'uniform':
                        temp = torch.FloatTensor(size, 2).uniform_(0, 1)
                    elif class_type.item() == 'cluster':
                        mean_x, mean_y = center[j, ::2], center[j, 1::2]
                        coords = torch.zeros(size, 2)
                        for i in range(n_cluster):
                            if i < n_cluster - 1:
                                coords[int(size/ n_cluster) * i:int(size / n_cluster) * (i + 1)] = \
                                    torch.cat(
                                        (torch.FloatTensor(int(size / n_cluster), 1).normal_(mean_x[i], std),
                                         torch.FloatTensor(int(size  / n_cluster), 1).normal_(mean_y[i], std)),
                                        dim=1)
                            elif i == n_cluster - 1:
                                coords[int(size / n_cluster) * i:] = \
                                    torch.cat((torch.FloatTensor(size - int(size / n_cluster) * i,1).normal_(mean_x[i], std),
                                               torch.FloatTensor(size  - int(size / n_cluster) * i,1).normal_(mean_y[i], std)), dim=1)
                        coords = torch.where(coords > 1, torch.ones_like(coords), coords)
                        coords = torch.where(coords < 0, torch.zeros_like(coords), coords)
                        temp = coords
                    elif class_type.item() == 'mixed':
                        mean_x, mean_y = center[j, ::2], center[j, 1::2]
                        mutate_idx = np.random.choice(range(size), int(size / 2), replace=False)
                        coords = torch.FloatTensor(size, 2).uniform_(0, 1)
                        for i in range(n_cluster_mix):
                            if i < n_cluster_mix - 1:
                                coords[mutate_idx[int(size / n_cluster_mix / 2) * i:int(size / n_cluster_mix / 2) * (i + 1)]] = \
                                    torch.cat((torch.FloatTensor(int(size / n_cluster_mix / 2), 1).normal_(mean_x[i],std),
                                              torch.FloatTensor(int(size / n_cluster_mix / 2), 1).normal_(mean_y[i],std)),dim=1)
                            elif i == n_cluster_mix - 1:
                                coords[mutate_idx[int(size / n_cluster_mix / 2) * i:]] = \
                                    torch.cat((torch.FloatTensor(int(size / 2) - int(size / n_cluster_mix / 2) * i,1).normal_(mean_x[i], std),
                                               torch.FloatTensor(int(size / 2) - int(size / n_cluster_mix / 2) * i,1).normal_(mean_y[i], std)), dim=1)
                        coords = torch.where(coords > 1, torch.ones_like(coords), coords)
                        coords = torch.where(coords < 0, torch.zeros_like(coords), coords)
                        temp =  coords
                    self.data.append(temp)
            if distribution == 'uniform':
                self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

            elif distribution == 'cluster':
                self.data = []
                center = np.array([list(np.random.rand(n_cluster * 2)) for _ in range(num_samples)])
                lower, upper = 0.2, 0.8
                center = lower + (upper - lower) * center
                for j in range(num_samples):
                    std = 0.07
                    mean_x, mean_y = center[j,::2], center[j,1::2]
                    coords = torch.zeros(size, 2)
                    for i in range(n_cluster):
                        if i < n_cluster - 1:
                            coords[int(size / n_cluster) * i:int(size / n_cluster) * (i + 1)] = \
                                torch.cat((torch.FloatTensor(int(size/ n_cluster),1).normal_(mean_x[i],std),
                                           torch.FloatTensor(int(size / n_cluster),1).normal_(mean_y[i],std)),dim=1)
                        elif i == n_cluster - 1:
                            coords[int(size / n_cluster) * i:] = \
                                torch.cat((torch.FloatTensor(size - int(size / n_cluster) * i, 1).normal_(mean_x[i], std),
                                           torch.FloatTensor(size - int(size / n_cluster) * i, 1).normal_(mean_y[i], std)), dim=1)
                    coords = torch.where(coords > 1, torch.ones_like(coords), coords)
                    coords = torch.where(coords < 0, torch.zeros_like(coords), coords)
                    self.data.append(coords)

            elif distribution == 'mixed':
                self.data = []
                center = np.array([list(np.random.rand(n_cluster_mix * 2)) for _ in range(num_samples)])
                lower, upper = 0.2, 0.8
                center = lower + (upper - lower) * center
                for j in range(num_samples):
                    std = 0.07
                    mean_x, mean_y = center[j,::2], center[j,1::2]
                    mutate_idx = np.random.choice(range(size),int(size / 2), replace=False )
                    coords = torch.FloatTensor(size, 2).uniform_(0, 1)
                    for i in range(n_cluster_mix):
                        if i < n_cluster_mix - 1:
                            coords[mutate_idx[int(size / n_cluster_mix / 2) * i:int(size / n_cluster_mix / 2) * (i + 1)]] = \
                                torch.cat((torch.FloatTensor(int(size / n_cluster_mix / 2), 1).normal_(mean_x[i], std),
                                           torch.FloatTensor(int(size / n_cluster_mix / 2), 1).normal_(mean_y[i],std)), dim=1)
                        elif i == n_cluster_mix - 1:
                            coords[mutate_idx[int(size / n_cluster_mix / 2) * i:]] = \
                                torch.cat((torch.FloatTensor(int(size / 2) - int(size / n_cluster_mix / 2) * i, 1).normal_(mean_x[i], std),
                                           torch.FloatTensor(int(size / 2) - int(size / n_cluster_mix / 2) * i,1).normal_(mean_y[i], std)),dim=1)
                    coords = torch.where(coords > 1, torch.ones_like(coords), coords)
                    coords = torch.where(coords < 0, torch.zeros_like(coords), coords)
                    self.data.append(coords)

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
