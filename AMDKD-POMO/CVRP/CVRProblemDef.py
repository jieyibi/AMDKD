
import torch
import numpy as np
import rpy2.robjects as robjects


def get_random_problems(batch_size, problem_size, distribution, load_path=None,episode=None):

    if load_path is not None:
        import os
        import pickle
        filename = load_path
        # print(episode)
        # if episode is not None:
        #     if episode == 0:
        #         print('Load from: ', filename)
        # print('Load from: ', filename)
        assert os.path.splitext(filename)[1] == '.pkl'
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        # episode = None
        if episode is not None:
            data = data[episode: episode + batch_size]
            # if episode==0:
            #     print(episode,episode+batch_size,len(data))
        for i in range(len(data)):
            depot_xy = torch.FloatTensor(data[i][0]).unsqueeze(0) if i == 0 else torch.cat(
                (depot_xy, torch.FloatTensor(data[i][0]).unsqueeze(0)), dim=0)
            node_xy = torch.FloatTensor(data[i][1]).unsqueeze(0).cuda() if i == 0 else torch.cat(
                (node_xy, torch.FloatTensor(data[i][1]).unsqueeze(0).cuda()), dim=0)
            node_demand = torch.FloatTensor(data[i][2]).unsqueeze(0) if i == 0 else torch.cat(
                (node_demand, torch.FloatTensor(data[i][2]).unsqueeze(0)), dim=0)
        depot_xy = depot_xy.unsqueeze(1).cuda()
        node_demand = node_demand.cuda() / float(data[0][3])
    else:
        if distribution['data_type'] == 'uniform':
            depot_xy = torch.rand(size=(batch_size, 1, 2))
            # shape: (batch, 1, 2)
            node_xy = torch.rand(size=(batch_size, problem_size, 2))
            # shape: (batch, problem, 2)

        elif distribution['data_type'] == 'cluster':
            n_cluster = distribution['n_cluster']
            center = np.array([list(np.random.rand(n_cluster * 2)) for _ in range(batch_size)])
            center = distribution['lower'] + (distribution['upper'] - distribution['lower']) * center
            std = distribution['std']
            for j in range(batch_size):
                mean_x, mean_y = center[j, ::2], center[j, 1::2]
                coords = torch.zeros(problem_size + 1, 2)
                for i in range(n_cluster):
                    if i < n_cluster - 1:
                        coords[int((problem_size + 1) / n_cluster) * i:int((problem_size + 1) / n_cluster) * (i + 1)] = \
                            torch.cat((torch.FloatTensor(int((problem_size + 1) / n_cluster), 1).normal_(mean_x[i], std),
                                       torch.FloatTensor(int((problem_size + 1) / n_cluster), 1).normal_(mean_y[i], std)),
                                      dim=1)
                    elif i == n_cluster - 1:
                        coords[int((problem_size + 1) / n_cluster) * i:] = \
                            torch.cat(
                                (torch.FloatTensor((problem_size + 1) - int((problem_size + 1) / n_cluster) * i, 1).normal_(
                                    mean_x[i], std),
                                 torch.FloatTensor((problem_size + 1) - int((problem_size + 1) / n_cluster) * i, 1).normal_(
                                     mean_y[i], std)), dim=1)

                coords = torch.where(coords > 1, torch.ones_like(coords), coords)
                coords = torch.where(coords < 0, torch.zeros_like(coords), coords)
                depot_idx = int(np.random.choice(range(coords.shape[0]), 1))
                node_xy = coords[torch.arange(coords.size(0)) != depot_idx].unsqueeze(0) if j == 0 else \
                    torch.cat((node_xy, coords[torch.arange(coords.size(0)) != depot_idx].unsqueeze(0)), dim=0)
                depot_xy = coords[depot_idx].unsqueeze(0).unsqueeze(0) if j == 0 else \
                    torch.cat((depot_xy, coords[depot_idx].unsqueeze(0).unsqueeze(0)), dim=0)

        elif distribution['data_type'] == 'mixed':
            depot_xy = torch.rand(size=(batch_size, 1, 2))  # shape: (batch, 1, 2)
            n_cluster_mix = distribution['n_cluster_mix']
            center = np.array([list(np.random.rand(n_cluster_mix * 2)) for _ in range(batch_size)])
            center = distribution['lower'] + (distribution['upper'] - distribution['lower']) * center
            std = distribution['std']
            for j in range(batch_size):
                mean_x, mean_y = center[j, ::2], center[j, 1::2]
                mutate_idx = np.random.choice(range(problem_size), int(problem_size / 2), replace=False)
                coords = torch.FloatTensor(problem_size, 2).uniform_(0, 1).to(depot_xy.device)
                for i in range(n_cluster_mix):
                    if i < n_cluster_mix - 1:
                        coords[mutate_idx[
                               int(problem_size / n_cluster_mix / 2) * i:int(problem_size / n_cluster_mix / 2) * (i + 1)]] = \
                            torch.cat((torch.FloatTensor(int(problem_size / n_cluster_mix / 2), 1).normal_(mean_x[i], std),
                                       torch.FloatTensor(int(problem_size / n_cluster_mix / 2), 1).normal_(mean_y[i], std)),
                                      dim=1).to(depot_xy.device)
                    elif i == n_cluster_mix - 1:
                        coords[mutate_idx[int(problem_size / n_cluster_mix / 2) * i:]] = \
                            torch.cat((torch.FloatTensor(int(problem_size / 2) - int(problem_size / n_cluster_mix / 2) * i,
                                                         1).normal_(mean_x[i], std),
                                       torch.FloatTensor(int(problem_size / 2) - int(problem_size / n_cluster_mix / 2) * i,
                                                         1).normal_(mean_y[i], std)), dim=1).to(depot_xy.device)

                coords = torch.where(coords > 1, torch.ones_like(coords), coords).to(depot_xy.device)
                coords = torch.where(coords < 0, torch.zeros_like(coords), coords).to(depot_xy.device)
                node_xy = coords.unsqueeze(0) if j == 0 else torch.cat((node_xy, coords.unsqueeze(0)), dim=0)

        elif distribution['data_type'] == 'mix_three':
            center = np.array([list(np.random.rand(3 * 2)) for _ in range(batch_size)])
            center = distribution['lower'] + (distribution['upper'] - distribution['lower']) * center
            std = distribution['std']
            for j in range(batch_size):
                class_type = np.random.choice(['uniform', 'cluster', 'mixed'], 1)
                if class_type.item() == 'uniform':
                    depot_xy_temp = torch.rand(size=(1, 1, 2))
                    node_xy_temp = torch.rand(size=(1, problem_size, 2))
                    # print('uniform')
                elif class_type.item() == 'cluster':
                    n_cluster = distribution['n_cluster']
                    mean_x, mean_y = center[j, ::2], center[j, 1::2]
                    coords = torch.zeros(problem_size + 1, 2)
                    for i in range(n_cluster):
                        if i < n_cluster - 1:
                            coords[
                            int((problem_size + 1) / n_cluster) * i:int((problem_size + 1) / n_cluster) * (i + 1)] = \
                                torch.cat(
                                    (torch.FloatTensor(int((problem_size + 1) / n_cluster), 1).normal_(mean_x[i], std),
                                     torch.FloatTensor(int((problem_size + 1) / n_cluster), 1).normal_(mean_y[i], std)),
                                    dim=1)
                        elif i == n_cluster - 1:
                            coords[int((problem_size + 1) / n_cluster) * i:] = \
                                torch.cat((torch.FloatTensor(
                                    (problem_size + 1) - int((problem_size + 1) / n_cluster) * i, 1).normal_(mean_x[i],
                                                                                                             std),
                                           torch.FloatTensor(
                                               (problem_size + 1) - int((problem_size + 1) / n_cluster) * i, 1).normal_(
                                               mean_y[i], std)), dim=1)
                    coords = torch.where(coords > 1, torch.ones_like(coords), coords)
                    coords = torch.where(coords < 0, torch.zeros_like(coords), coords)
                    depot_idx = int(np.random.choice(range(coords.shape[0]), 1))
                    node_xy_temp = coords[torch.arange(coords.size(0)) != depot_idx].unsqueeze(0)
                    depot_xy_temp = coords[depot_idx].unsqueeze(0).unsqueeze(0)
                    # print('cluster')
                elif class_type.item() == 'mixed':
                    n_cluster_mix = distribution['n_cluster_mix']
                    depot_xy_temp = torch.rand(size=(1, 1, 2))  # shape: (1, 1, 2)
                    mean_x, mean_y = center[j, ::2], center[j, 1::2]
                    mutate_idx = np.random.choice(range(problem_size), int(problem_size / 2), replace=False)
                    coords = torch.FloatTensor(problem_size, 2).uniform_(0, 1).to(depot_xy_temp.device)
                    for i in range(n_cluster_mix):
                        if i < n_cluster_mix - 1:
                            coords[mutate_idx[int(problem_size / n_cluster_mix / 2) * i:int(
                                problem_size / n_cluster_mix / 2) * (i + 1)]] = \
                                torch.cat((torch.FloatTensor(int(problem_size / n_cluster_mix / 2), 1).normal_(
                                    mean_x[i], std),
                                           torch.FloatTensor(int(problem_size / n_cluster_mix / 2), 1).normal_(
                                               mean_y[i], std)), dim=1).to(depot_xy_temp.device)
                        elif i == n_cluster_mix - 1:
                            coords[mutate_idx[int(problem_size / n_cluster_mix / 2) * i:]] = \
                                torch.cat((torch.FloatTensor(
                                    int(problem_size / 2) - int(problem_size / n_cluster_mix / 2) * i, 1).normal_(
                                    mean_x[i], std),
                                           torch.FloatTensor(
                                               int(problem_size / 2) - int(problem_size / n_cluster_mix / 2) * i,
                                               1).normal_(mean_y[i], std)), dim=1).to(depot_xy_temp.device)

                    coords = torch.where(coords > 1, torch.ones_like(coords), coords).to(depot_xy_temp.device)
                    coords = torch.where(coords < 0, torch.zeros_like(coords), coords).to(depot_xy_temp.device)
                    node_xy_temp = coords.unsqueeze(0)
                    # print('mixed')
                depot_xy = depot_xy_temp if j == 0 else torch.cat((depot_xy, depot_xy_temp), dim=0)
                node_xy = node_xy_temp if j == 0 else torch.cat((node_xy, node_xy_temp), dim=0)
        elif distribution['data_type'] == 'expansion' or distribution['data_type'] == 'explosion': # NORM
            robjects.r(
                '''
                get_instances = function(distribution, n.point){
                  library("netgen")
                  library("gridExtra")
                  library("tspgen")
                  RUE = generateRandomNetwork(n.point, lower = 0, upper = 1)
                  ##########################################
                  if(distribution=='explosion'){
                    x = RUE
                    x$coordinates = doExplosionMutation(x$coordinates, min.eps=0.3, max.eps=0.3)
                    return(x$coordinates)
                  }
                  #######################################
                  if(distribution=='implosion'){
                    x = RUE
                    x$coordinates = doImplosionMutation(x$coordinates, min.eps=0.3, max.eps=0.3)
                    return(x$coordinates)
                  }
                  #########################################
                  if(distribution=='expansion'){
                    x = RUE
                    x$coordinates = doExpansionMutation(x$coordinates, min.eps=0.3, max.eps=0.3)
                    return(x$coordinates)
                  }
                  ########################################
                  if(distribution=='grid'){
                    x = RUE
                    x$coordinates = doGridMutation(x$coordinates, box.min=0.3, box.max=0.3, p.rot=0, p.jitter=0, jitter.sd=0.05)
                    return(x$coordinates)
                  }
                }
                '''
            )
            for j in range(batch_size):
                instances = robjects.r('get_instances')(distribution['data_type'], problem_size + 1)
                width = torch.FloatTensor(np.array(instances)).max()-torch.FloatTensor(np.array(instances)).min()
                min_coord = torch.FloatTensor(np.array(instances)).min()
                coords = (torch.FloatTensor(np.array(instances)).cuda()-min_coord.cuda())/width.cuda()
                depot_idx = int(np.random.choice(range(coords.shape[0]), 1))
                # shape: (batch, problem, 2)
                node_xy = coords[torch.arange(coords.size(0)) != depot_idx].unsqueeze(0) if j == 0 else \
                    torch.cat((node_xy, coords[torch.arange(coords.size(0)) != depot_idx].unsqueeze(0)), dim=0)
                # shape: (batch, 1, 2)
                depot_xy = coords[depot_idx].unsqueeze(0).unsqueeze(0) if j == 0 else \
                    torch.cat((depot_xy, coords[depot_idx].unsqueeze(0).unsqueeze(0)), dim=0)
        elif distribution['data_type'] == 'grid' or distribution['data_type'] == 'implosion': # NO NORM
            robjects.r(
                '''
                get_instances = function(distribution, n.point){
                  library("netgen")
                  library("gridExtra")
                  library("tspgen")
                  RUE = generateRandomNetwork(n.point, lower = 0, upper = 1)
                  ##########################################
                  if(distribution=='explosion'){
                    x = RUE
                    x$coordinates = doExplosionMutation(x$coordinates, min.eps=0.3, max.eps=0.3)
                    return(x$coordinates)
                  }
                  #######################################
                  if(distribution=='implosion'){
                    x = RUE
                    x$coordinates = doImplosionMutation(x$coordinates, min.eps=0.3, max.eps=0.3)
                    return(x$coordinates)
                  }
                  #########################################
                  if(distribution=='expansion'){
                    x = RUE
                    x$coordinates = doExpansionMutation(x$coordinates, min.eps=0.3, max.eps=0.3)
                    return(x$coordinates)
                  }
                  ########################################
                  if(distribution=='grid'){
                    x = RUE
                    x$coordinates = doGridMutation(x$coordinates, box.min=0.3, box.max=0.3, p.rot=0, p.jitter=0, jitter.sd=0.05)
                    return(x$coordinates)
                  }
                }
                '''
            )
            for j in range(batch_size):
                instances = robjects.r('get_instances')(distribution['data_type'], problem_size + 1)
                coords = torch.FloatTensor(np.array(instances)).cuda()
                depot_idx = int(np.random.choice(range(coords.shape[0]), 1))
                # shape: (batch, problem, 2)
                node_xy = coords[torch.arange(coords.size(0)) != depot_idx].unsqueeze(0).cuda() if j == 0 else \
                    torch.cat((node_xy, coords[torch.arange(coords.size(0)) != depot_idx].unsqueeze(0)), dim=0)
                # shape: (batch, 1, 2)
                depot_xy = coords[depot_idx].unsqueeze(0).unsqueeze(0).cuda() if j == 0 else \
                    torch.cat((depot_xy, coords[depot_idx].unsqueeze(0).unsqueeze(0)), dim=0).cuda()
        else:
            assert 0, 'Distribution not defined!'

        if problem_size == 20:
            demand_scaler = 30
        elif problem_size == 50:
            demand_scaler = 40
        elif problem_size == 100:
            demand_scaler = 50
        else:
            raise NotImplementedError

        node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    return depot_xy, node_xy, node_demand


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data