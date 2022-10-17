import torch
import numpy as np


def get_random_problems(batch_size, problem_size, distribution, load_path=None,episode=None):
    if load_path is not None:
        import os
        import pickle
        filename = load_path
        assert os.path.splitext(filename)[1] == '.pkl'
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        if episode is not None:
            data = data[episode: episode+batch_size]
            # if episode==0:
            #     print(episode,episode+batch_size,len(data))
        problems = torch.FloatTensor(data).cuda()
    else:
        # print(distribution['data_type'])
        if distribution['data_type'] == 'uniform':
            problems = torch.rand(size=(batch_size, problem_size, 2))
            # problems.shape: (batch, problem, 2)
        elif distribution['data_type'] == 'cluster':
            n_cluster = distribution['n_cluster']
            center = np.array([list(np.random.rand(n_cluster * 2)) for _ in range(batch_size)])
            center = distribution['lower'] + (distribution['upper'] - distribution['lower']) * center
            std = distribution['std']
            for j in range(batch_size):
                mean_x, mean_y = center[j, ::2], center[j, 1::2]
                coords = torch.zeros(problem_size, 2)
                for i in range(n_cluster):
                    if i < n_cluster - 1:
                        coords[int((problem_size) / n_cluster) * i:int((problem_size) / n_cluster) * (i + 1)] = \
                            torch.cat((torch.FloatTensor(int((problem_size) / n_cluster), 1).normal_(mean_x[i], std),
                                 torch.FloatTensor(int((problem_size) / n_cluster), 1).normal_(mean_y[i], std)),dim=1)
                    elif i == n_cluster - 1:
                        coords[int((problem_size) / n_cluster) * i:] = \
                            torch.cat((torch.FloatTensor((problem_size) - int((problem_size) / n_cluster) * i,1).normal_(mean_x[i], std),
                                 torch.FloatTensor((problem_size) - int((problem_size) / n_cluster) * i,1).normal_(mean_y[i], std)), dim=1)
                coords = torch.where(coords > 1, torch.ones_like(coords), coords)
                coords = torch.where(coords < 0, torch.zeros_like(coords), coords)
                problems = coords.unsqueeze(0) if j == 0 else torch.cat((problems, coords.unsqueeze(0)), dim=0)
        elif distribution['data_type'] == 'mixed':
            n_cluster_mix = distribution['n_cluster_mix']
            center = np.array([list(np.random.rand(n_cluster_mix * 2)) for _ in range(batch_size)])
            center = distribution['lower'] + (distribution['upper'] - distribution['lower']) * center
            std = distribution['std']
            for j in range(batch_size):
                mean_x, mean_y = center[j, ::2], center[j, 1::2]
                mutate_idx = np.random.choice(range(problem_size), int(problem_size / 2), replace=False)
                coords = torch.FloatTensor(problem_size, 2).uniform_(0, 1)
                for i in range(n_cluster_mix):
                    if i < n_cluster_mix - 1:
                        coords[mutate_idx[int(problem_size / n_cluster_mix / 2) * i:int(problem_size / n_cluster_mix / 2) * (i + 1)]] = \
                            torch.cat((torch.FloatTensor(int(problem_size / n_cluster_mix / 2), 1).normal_(mean_x[i], std),
                                 torch.FloatTensor(int(problem_size / n_cluster_mix / 2), 1).normal_(mean_y[i], std)),dim=1)
                    elif i == n_cluster_mix - 1:
                        coords[mutate_idx[int(problem_size / n_cluster_mix / 2) * i:]] = \
                            torch.cat((torch.FloatTensor(int(problem_size / 2) - int(problem_size / n_cluster_mix / 2) * i,1).normal_(mean_x[i], std),
                                 torch.FloatTensor(int(problem_size / 2) - int(problem_size / n_cluster_mix / 2) * i,1).normal_(mean_y[i], std)), dim=1)
                coords = torch.where(coords > 1, torch.ones_like(coords), coords)
                coords = torch.where(coords < 0, torch.zeros_like(coords), coords).cuda()
                problems = coords.unsqueeze(0) if j == 0 else torch.cat((problems, coords.unsqueeze(0)), dim=0)
        elif distribution['data_type'] == 'mix_three':
            center =  np.array([list(np.random.rand(3 * 2)) for _ in range(batch_size)])
            center = distribution['lower'] + (distribution['upper'] - distribution['lower']) * center
            std = distribution['std']
            for j in range(batch_size):
                class_type = np.random.choice(['uniform', 'cluster', 'mixed'], 1)
                if class_type.item() == 'uniform':
                    problems_temp = torch.rand(size=(1, problem_size, 2))
                elif class_type.item() == 'cluster':
                    n_cluster = distribution['n_cluster']
                    mean_x, mean_y = center[j, ::2], center[j, 1::2]
                    coords = torch.zeros(problem_size, 2)
                    for i in range(n_cluster):
                        if i < n_cluster - 1:
                            coords[int((problem_size) / n_cluster) * i:int((problem_size) / n_cluster) * (i + 1)] = \
                                torch.cat((torch.FloatTensor(int((problem_size) / n_cluster), 1).normal_(mean_x[i], std),
                                     torch.FloatTensor(int((problem_size) / n_cluster), 1).normal_(mean_y[i], std)),dim=1)
                        elif i == n_cluster - 1:
                            coords[int((problem_size) / n_cluster) * i:] = torch.cat(
                                (torch.FloatTensor((problem_size) - int((problem_size) / n_cluster) * i, 1).normal_(mean_x[i],std),
                                torch.FloatTensor((problem_size) - int((problem_size) / n_cluster) * i, 1).normal_(mean_y[i], std)), dim=1)
                    coords = torch.where(coords > 1, torch.ones_like(coords), coords)
                    coords = torch.where(coords < 0, torch.zeros_like(coords), coords)
                    problems_temp = coords.unsqueeze(0)
                elif class_type.item() == 'mixed':
                    n_cluster_mix = distribution['n_cluster_mix']
                    mean_x, mean_y = center[j, ::2], center[j, 1::2]
                    mutate_idx = np.random.choice(range(problem_size), int(problem_size / 2), replace=False)
                    coords = torch.FloatTensor(problem_size, 2).uniform_(0, 1)
                    for i in range(n_cluster_mix):
                        if i < n_cluster_mix - 1:
                            coords[mutate_idx[int(problem_size / n_cluster_mix / 2) * i:int(problem_size / n_cluster_mix / 2) * (i + 1)]] = \
                                torch.cat((torch.FloatTensor(int(problem_size / n_cluster_mix / 2), 1).normal_(mean_x[i], std),
                                           torch.FloatTensor(int(problem_size / n_cluster_mix / 2), 1).normal_(mean_y[i], std)), dim=1)
                        elif i == n_cluster_mix - 1:
                            coords[mutate_idx[int(problem_size / n_cluster_mix / 2) * i:]] = \
                                torch.cat((torch.FloatTensor(int(problem_size / 2) - int(problem_size / n_cluster_mix / 2) * i, 1).normal_(mean_x[i], std),
                                           torch.FloatTensor(int(problem_size / 2) - int(problem_size / n_cluster_mix / 2) * i,1).normal_(mean_y[i], std)), dim=1)
                    coords = torch.where(coords > 1, torch.ones_like(coords), coords)
                    coords = torch.where(coords < 0, torch.zeros_like(coords), coords)
                    problems_temp = coords.unsqueeze(0)
                    # print('mixed')
                problems = problems_temp.cuda() if j == 0 else torch.cat((problems, problems_temp.cuda()), dim=0)
        else:
            assert 0, 'Distribution not defined!'

    return problems


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems