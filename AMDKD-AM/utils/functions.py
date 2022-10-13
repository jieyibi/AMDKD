import warnings

import torch
import numpy as np
import os
import json
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import torch.nn.functional as F
from collections import OrderedDict

def load_problem(name):
    from problems import TSP, CVRP, SDVRP, OP, PCTSPDet, PCTSPStoch
    problem = {
        'tsp': TSP,
        'cvrp': CVRP,
        'sdvrp': SDVRP,
        'op': OP,
        'pctsp_det': PCTSPDet,
        'pctsp_stoch': PCTSPStoch,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    if isinstance(load_path,list):
        print('  [*] Loading encoder from {}'.format(load_path[0]))
        print('  [*] Loading decoder from {}'.format(load_path[1]))
    else:
        print('  [*] Loading model from {}'.format(load_path))
        load_data = torch.load(
            os.path.join(
                os.getcwd(),
                load_path
            ), map_location=lambda storage, loc: storage)
        if isinstance(load_data, dict):
            load_optimizer_state_dict = load_data.get('optimizer', None)
            # load_model_state_dict = load_data.get('model', load_data)
            load_model_state_dict = load_data.get('model', {})
            new_state_dict = OrderedDict()
            for k, v in load_model_state_dict.items():
                if 'W_hidden' not in k and 'W_embed' not in k:
                    new_state_dict[k] = v
            load_model_state_dict = new_state_dict
        else:
            load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()
    state_dict.update(load_model_state_dict)
    # model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args


def load_model(path, epoch=None,use_adjacent=False):
    from nets.attention_model import AttentionModel,student_AttentionModel
    from nets.pointer_network import PointerNetwork

    if isinstance(path,list):
        encoder_model_filename, decoder_model_filename = path[0], path[1]
        encoder_path, decoder_path = os.path.dirname(encoder_model_filename), os.path.dirname(decoder_model_filename)
        args = load_args(os.path.join(encoder_path, 'args.json'))
    else:
        if os.path.isfile(path):
            model_filename = path
            path = os.path.dirname(model_filename)
        elif os.path.isdir(path):
            if epoch is None:
                epoch = max(
                    int(os.path.splitext(filename)[0].split("-")[1])
                    for filename in os.listdir(path)
                    if os.path.splitext(filename)[1] == '.pt'
                )
            model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
        else:
            assert False, "{} is not a valid directory or file".format(path)

        args = load_args(os.path.join(path, 'args.json'))

    problem = load_problem(args['problem'])

    model_class = {
        'attention': AttentionModel if not use_adjacent else student_AttentionModel,
        'pointer': PointerNetwork
    }.get(args.get('model', 'attention'), None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    # print(args['normalization'])
    model = model_class(
        args['student_embedding_dim'],
        args['student_hidden_dim'],
        problem,
        n_encode_layers=args['student_n_encode_layers'],
        mask_inner=True,
        mask_logits=True,
        normalization=args['student_normalization'],
        tanh_clipping=args['tanh_clipping'],
        checkpoint_encoder=args.get('checkpoint_encoder', False),
        shrink_size=args.get('shrink_size', None),

    )


    if isinstance(path,list):
        print('  [*] Loading encoder from {}'.format(path[0]))
        print('  [*] Loading decoder from {}'.format(path[1]))
        enc_load_data , dec_load_data = torch_load_cpu(encoder_model_filename), torch_load_cpu(decoder_model_filename)
        enc_state_dict = enc_load_data.get('model', {}) if not isinstance(enc_load_data,OrderedDict) else enc_load_data
        dec_state_dict = dec_load_data.get('model', {}) if not isinstance(dec_load_data,OrderedDict) else dec_load_data
        new_state_dict = OrderedDict()
        for k, v in enc_state_dict.items():
            if 'project' in k:
                break
            new_state_dict[k] = v
        for k, v in dec_state_dict.items():
            if 'project' in k and 'W_hidden' not in k and 'W_embed' not in k:
                new_state_dict[k] = v

    else:
        # Overwrite model parameters by parameters to load
        load_data = torch_load_cpu(model_filename)
        # model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})
        state_dict = load_data.get('model', {})
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'W_hidden' not in k and 'W_embed' not in k:
                new_state_dict[k] = v

    model.load_state_dict({**model.state_dict(), **new_state_dict})

    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    # model.load_state_dict({**model.state_dict(), **new_state_dict})

    if not isinstance(path, list):
        model, *_ = _load_model_file(model_filename, model)

    model.eval()  # Put in eval mode

    return model, args


def parse_softmax_temperature(raw_temp):
    # Load from file
    if os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    """
    :param input: (batch_size, graph_size, node_dim) input node features
    :return:
    """
    input = do_batch_rep(input, batch_rep)

    costs = []
    pis = []
    for i in range(iter_rep):
        _log_p, pi = inner_func(input)
        # pi.view(-1, batch_rep, pi.size(-1))
        cost, mask = get_cost_func(input, pi)

        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)
    # (batch_size * batch_rep, iter_rep, max_length) => (batch_size, batch_rep * iter_rep, max_length)
    pis = torch.cat(
        [F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis],
        1
    )  # .view(embeddings.size(0), batch_rep * iter_rep, max_length)
    costs = torch.cat(costs, 1)

    # (batch_size)
    mincosts, argmincosts = costs.min(-1)
    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]

    return minpis, mincosts
