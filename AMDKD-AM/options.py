import os
import time
import argparse
import torch
import sys

def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Adaptive Multi-Distribution Knowledge Distillation (AMDKD) scheme for AM")

    # Data
    parser.add_argument('--problem', default='cvrp', help="The problem to solve, or 'tsp'")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--batch_size', type=int, default=512, help='Number of instances per batch during training')
    parser.add_argument('--epoch_size', type=int, default=128000, help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')
    parser.add_argument('--test_type',type=str,default='greedy',help='test type')

    # Model
    parser.add_argument('--model', default='attention', help="Model, 'attention' (default) or 'pointer'")
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh.'
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")

    # Knowledge distillation parameters
    parser.add_argument('--distillation', action='store_true',default=True, help='whether to use knowledge distillation')
    parser.add_argument('--adaptive_prob',action='store_true',default=True,help='randomly choose teacher considering the gap of each distribution')
    parser.add_argument('--random_adaptive_prob',type=float, default=0 ,help='randomly choose whether considering the gap or not')
    parser.add_argument('--adaptive_prob_type',type=str, default="softmax", help='the way to calculate the adaptive prob, softmax or sum')
    parser.add_argument('--start_adaptive_epoch',type=int, default=500)

    parser.add_argument('--student_embedding_dim', type=int, default=64, help='Dimension of input embedding in student net')
    parser.add_argument('--student_hidden_dim', type=int, default=64, help='Dimension of hidden layers in Enc/Dec in student net')
    parser.add_argument('--student_n_encode_layers', type=int, default=1,
                        help='Number of layers in the encoder/critic network in student net')
    parser.add_argument('--student_normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")
    parser.add_argument('--student_feed_forward_hidden',default=512,type=int,help='FFN dim in student graph encoder(Transformer block)')
    parser.add_argument('--student_load',action='store_true',default=False,help='whether to load student model')
    parser.add_argument('--student_load_path',type=str, help='load path of the student model')

    parser.add_argument('--n_cluster', type=int, default=3, help='n_cluster for cluster distribution')
    parser.add_argument('--n_cluster_mix', type=int, default=1, help='n_cluster for mixed cluster distribution')
    parser.add_argument('--generate_mix_data', action='store_true', default=False, help='whether to generate mix data')
    parser.add_argument('--distill_temperature', type=int, default=1)
    parser.add_argument('--rl_alpha',type = float, default=0.5, help='weight for RL loss (task loss)')
    parser.add_argument('--distill_alpha', type=float, default=0.5, help='weight for KLD loss (distill loss)')
    parser.add_argument('--twist_kldloss',action='store_true',default=False)
    parser.add_argument('--meaningful_KLD',action='store_true',default=False)
    parser.add_argument('--router',type=str,default='student',help='Teacher or student acts as the router')
    parser.add_argument('--hinton_t2', action='store_true',default=False, help='soft target loss * temperature^2')
    parser.add_argument('--distill_distribution', action='store_true',default=False, help='AMDKD')
    parser.add_argument('--load_path_uniform', type=str, help='teacher model under uniform distribution')
    parser.add_argument('--load_path_cluster', type=str, help='teacher model under cluster distribution')
    parser.add_argument('--load_path_mixed', type=str, help='teacher model under mixed distribution')
    parser.add_argument('--multi_distribution_baseline', action='store_true', default=False, help='whether to use mix data during baseline evaluation')
    parser.add_argument('--multi_teacher', action='store_true', default=False, help='whether to use the multi-teacher loss[1]')
    # [1] Chuhan Wu et al. One Teacher is Enough? Pre-trained Language Model Distillation from Multiple Teachers. arXiv preprint arXiv:2106.01023. 2021.
    parser.add_argument('--multi_test', action='store_true',default=False, help='whether to test in different distributions')
    parser.add_argument('--normalization_uniform', default='batch',help="Normalization type, 'batch' (default) or 'instance'")
    parser.add_argument('--normalization_cluster', default='instance', help="Normalization type, 'batch' (default) or 'instance'")
    parser.add_argument('--normalization_mixed', default='batch', help="Normalization type, 'batch' (default) or 'instance'")
    parser.add_argument('--val_dataset_uniform', type=str, default=None, help='Dataset file to use for validation')
    parser.add_argument('--val_dataset_cluster', type=str, default=None, help='Dataset file to use for validation')
    parser.add_argument('--val_dataset_mixed', type=str, default=None, help='Dataset file to use for validation')

    # Training
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network(1e-4 for basic AM)")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch')
    parser.add_argument('--eval_only', action='store_true', default=False, help='Set this value to only evaluate model')
    parser.add_argument('--n_epochs', type=int, default=10000, help='The number of epochs to train')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay (default 0.8)')
    parser.add_argument('--baseline', default='rollout',
                        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                        help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
                             'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                             ' to save memory (default None means no shrinking)')
    parser.add_argument('--data_distribution', type=str, default='uniform',
                        help='Data distribution to use during training, defaults and options depend on problem.')

    # Misc
    parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run_AMDKD', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--is_load',action='store_true',default=False, help='whether to load model parameters and optimizer state')
    parser.add_argument('--is_load_multi', action='store_true',default=False)
    parser.add_argument('--load_path', type=str, help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--all_cuda_visible',action='store_true',help='Whether to use all available cuda')
    parser.add_argument('--CUDA_VISIBLE_ID', default="0",
                        help='Make specific id of cuda visible and use them instead of all available cuda')


    opts = parser.parse_args(args)
    if not opts.all_cuda_visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = opts.CUDA_VISIBLE_ID
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda

    opts.best = 0
    opts.save_best = False
    if opts.multi_test:
        if opts.problem == 'tsp':
            opts.val_dataset_uniform = 'data/tsp/tsp_uniform{}_1000_seed1234.pkl'.format(opts.graph_size)
            opts.val_dataset_cluster = 'data/tsp/tsp_cluster{}_1000_seed1234_noLHS.pkl'.format(opts.graph_size)
            opts.val_dataset_mixed = 'data/tsp/tsp_mixed{}_1000_seed1234_noLHS.pkl'.format(opts.graph_size)
        elif opts.problem == 'cvrp':
            opts.val_dataset_uniform = 'data/vrp/vrp_uniform{}_1000_seed1234.pkl'.format(opts.graph_size)
            opts.val_dataset_cluster = 'data/vrp/vrp_cluster{}_1000_seed1234_noLHS.pkl'.format(opts.graph_size)
            opts.val_dataset_mixed = 'data/vrp/vrp_mixed{}_1000_seed1234_noLHS.pkl'.format(opts.graph_size)
    if opts.adaptive_prob:
        if opts.problem == 'tsp':
            LKH3_optimal = {
                20: [3.84485, 1.824836, 3.2708310000000003],
                50: [5.686744, 2.666433, 4.912134],
                100: [7.753418, 3.667576, 6.729566]
            }
        elif  opts.problem == 'cvrp':
            LKH3_optimal = {
                20: [6.156523, 3.05725599999999, 5.439149],
                50: [10.417558, 5.155511, 9.354149],
                100: [15.740834, 7.909336, 14.294179]
            }
        opts.LKH3_optimal = LKH3_optimal[opts.graph_size]


    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    )

    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0

    if opts.distillation:
        if opts.distill_distribution:
            opts.multi_test = True
            if not opts.load_path_uniform:
                opts.load_path_uniform = 'teacher/{}{}/{}{}-uniform-epoch-99.pt'.format(opts.problem, opts.graph_size, opts.problem, opts.graph_size)
                print("Using given Uniform teacher")
            if not opts.load_path_cluster:
                opts.load_path_cluster = 'teacher/{}{}/{}{}-cluster-epoch-99.pt'.format(opts.problem, opts.graph_size, opts.problem, opts.graph_size)
                print("Using given Cluster teacher")
            if not opts.load_path_mixed:
                opts.load_path_mixed = 'teacher/{}{}/{}{}-mixed-epoch-99.pt'.format(opts.problem, opts.graph_size, opts.problem, opts.graph_size)
                print("Using given Mixed teacher")
            opts.load_path_multi = {
                'uniform': opts.load_path_uniform,
                'cluster': opts.load_path_cluster,
                'mixed': opts.load_path_mixed
            }
        else:
            assert opts.load_path is not None, "Knowledge Distillation for a single model must load a teacher model!"

    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"


    return opts

if __name__ == "__main__":
    get_options()