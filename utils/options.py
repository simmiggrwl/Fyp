import os
import time
import torch
import argparse

from utils.data_utils import set_seed, str2bool


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention model (Transformer) to solve the Combinatorial Optimization Problems, specially the OP "
                    "(Orienteering Problem), with Reinforcement Learning. The code is mainly adapted for the OP.")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed to use")

    # Problem
    parser.add_argument('--problem', default='op', help="The problem to solve. Options: op, tsp, pctsp, vrp, top")
    parser.add_argument('--num_agents', type=int, default=1, help="Number of agents (only for OP-MP-TN)")
    parser.add_argument('--num_depots', type=int, default=1, help="Number of depots. Options are 1 or 2. num_depots=1"
                        "means that the start and end depot are the same. num_depots=2 means that they are different")

    # Data
    parser.add_argument('--max_length', type=float, default=2, help="Normalized time limit to solve the problem")
    parser.add_argument('--data_dist', type=str, default='coop', help="Data dist (rewards of regions)"
                        "of OP. Options: const, unif, dist, coop (OP-MP-TN), nocoop")
    parser.add_argument('--cluster', type=str, default='kmc', help="For OP with coop/nocoop data dist,"
                        "choose cluster method: K-Means(km), K-Means constrained(kmc), Fuzzy C-Means(fcm)")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--epoch_size', type=int, default=1280000, help="Number of instances per epoch during training")
    parser.add_argument('--train_dataset', type=str, default=None, help='Dataset file to use for training')
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')
    parser.add_argument('--val_size', type=int, default=10000, help="Number of instances to report val performance")

    # Model
    parser.add_argument('--model', default='attention', help="Model, 'attention' (default), 'pointer' or 'gpn'")
    parser.add_argument('--batch_size', type=int, default=512, help="Number of instances per batch during training")
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")
    parser.add_argument('--embedding_dim', type=int, default=128, help="Dimension of input embedding")
    parser.add_argument('--hidden_dim', type=int, default=128, help="Dimension of hidden layers in Enc/Dec")
    parser.add_argument('--n_encode_layers', type=int, default=3, help="Number of layers in the encoder/critic network")
    parser.add_argument('--tanh_clipping', type=float, default=10., help="Clip the parameters to within +- this value"
                        "using tanh. Set to 0 to not perform any clipping.")

    # Training
    parser.add_argument('--num_workers', type=int, default=8, help="Number of parallel workers loading data batches")
    parser.add_argument('--epochs', type=int, default=100, help="The number of epochs to train")
    parser.add_argument('--initial_epoch', type=int, default=0, help="Initial epoch (relevant for learning rate decay)")
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1.0, help="Learning rate decay per epoch")
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help="Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)")
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help="Exponential moving average baseline decay (default 0.8)")
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help="Set to decrease memory usage by checkpointing encoder")
    parser.add_argument('--shrink_size', type=int, default=None,
                        help="Shrink the batch size if at least this many instances in the batch are finished to save"
                             "memory (default None means no shrinking)")

    # Baseline
    parser.add_argument('--baseline', default=None,
                        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help="Significance in the t-test for updating rollout baseline")
    parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                        help="Number of epochs to warmup the baseline, default None means 1 for rollout (exponential"
                             "used for warmup phase), 0 otherwise. Can only be used with rollout baseline.")

    # Misc
    parser.add_argument('--use_cuda', type=str2bool, default=True, help="True to use CUDA")
    parser.add_argument('--eval_only', type=str2bool, default=False, help="Set this value to only evaluate model")
    parser.add_argument('--log_step', type=int, default=50, help="Log info every log_step steps")
    parser.add_argument('--output_dir', default='outputs', help="Directory to write output models to")
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help="Save checkpoint every n epochs (default 1), 0 to save no checkpoints")
    parser.add_argument('--load_path', help="Path to load model parameters and optimizer state from")
    parser.add_argument('--resume', help="Resume from previous checkpoint file")
    parser.add_argument('--no_tensorboard', action='store_true', help="Disable logging TensorBoard files")
    parser.add_argument('--no_progress_bar', action='store_true', help="Disable progress bar")
    opts = parser.parse_args(args)

    # Use CUDA
    opts.use_cuda = torch.cuda.is_available() and opts.use_cuda

    # Filenames
    opts.time_txt = time.strftime("%Y%m%dT%H%M%S")
    num_agents_str = "_{}agents".format(opts.num_agents) if opts.num_agents > 1 else ""
    num_depots_str = "_{}depots".format(opts.num_depots) if opts.num_depots > 1 else ""
    initial_planning = os.path.join("{}{}".format(opts.data_dist, opts.graph_size), opts.cluster)
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, initial_planning) if opts.problem == 'op'
        else "{}_{}".format(opts.problem, opts.graph_size),
        "{}_{}{}{}_{}".format(opts.model, opts.baseline, num_agents_str, num_depots_str, opts.time_txt)
    )

    # Warmup epochs for baselines
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0

    # Check options are correct
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    assert opts.num_agents > 0, "num_agents must be greater than 0"
    assert opts.num_depots > 0, "num_depots must be greater than 0"
    set_seed(opts.seed)
    return opts
