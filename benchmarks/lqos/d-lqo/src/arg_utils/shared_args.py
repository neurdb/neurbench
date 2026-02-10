import argparse
import pathlib
from utils import file_utils
from collections import namedtuple
import os
import torch
import numpy as np


def seed_everything():
    torch.manual_seed(0)
    import random
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False


def get_stats_arg_parser():
    parser = argparse.ArgumentParser(description='AIOPT')
    parser.add_argument('--data', type=str, default='STATS',
                        help='')

    parser.add_argument('--is_train', type=int, default=0, help='process for train or test')

    # ----------------------------------- Data Path Params -----------------------------------
    parser.add_argument('--whole_dataset_dir', type=str, default='./datasets/origin_datasets/imdb/imdb/',
                        help='Dir of the original datasets')
    parser.add_argument('--create_table_file_path', type=str,
                        default='./datasets/processed_datasets/imdb/create_table.sql',
                        help='Dir of the original datasets')
    parser.add_argument('--create_index_file_path', type=str,
                        default='./datasets/processed_datasets/imdb/fkindexes.sql',
                        help='Dir of the original datasets')
    parser.add_argument('--base_query_file_path', type=str,
                        default='./datasets/origin_datasets/imdb/query_merge.txt',
                        help='Dir of the original datasets')
    parser.add_argument('--used_dataset_dir', type=str, default='./datasets/origin_datasets/imdb/imdb/',
                        help='Dir of the used datasets')
    parser.add_argument('--output_path', type=str, default='./experiment',
                        help='Dir of the original datasets')

    # ----------------------------------- Featurization Params -----------------------------------
    # Histogram Feature Params
    parser.add_argument('--n_bins', type=int, default=40, help='')

    # ----------------------------------- Model Params -----------------------------------
    parser.add_argument('--model', type=str, default='ALECE', help='')
    parser.add_argument('--input_dim', type=int, default=97, help='')
    parser.add_argument('--use_float64', type=int, default=0, help='Use float64 in label')
    parser.add_argument('--latent_dim', type=int, default=256, help='dimension of latent variables.')
    parser.add_argument('--mlp_num_layers', type=int, default=6, help='number of hidden layers in a mlp')
    parser.add_argument('--mlp_hidden_dim', type=int, default=512,
                        help='number of neurons in a mlp layer.')
    parser.add_argument('--use_positional_embedding', type=int, default=0, help='')
    parser.add_argument('--use_dropout', type=int, default=0, help='')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='')
    parser.add_argument('--num_attn_heads', type=int, default=8, help='')
    parser.add_argument('--attn_head_key_dim', type=int, default=511, help='')
    parser.add_argument('--feed_forward_dim', type=int, default=2048, help='')
    parser.add_argument('--num_self_attn_layers', type=int, default=6, help='')
    parser.add_argument('--num_cross_attn_layers', type=int, default=6, help='')

    # ----------------------------------- Training Params -----------------------------------
    parser.add_argument('--gpu', type=int, default=1, help='')
    parser.add_argument('--buffer_size', type=int, default=32, help='')
    parser.add_argument('--use_loss_weights', type=int, default=1, help='handle imbalance training label?')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--shuffle_buffer_size', type=int, default=400)
    # parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for Adam optimizer.')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs.')
    parser.add_argument('--min_n_epochs', type=int, default=3, help='Minimum number of epochs.')
    parser.add_argument('--card_log_scale', type=int, default=1, help='take logarithm of the card')
    parser.add_argument('--scaling_ratio', type=float, default=20., help='log(card)/scaling_ratio')

    # ----------------------------------- workload Params -----------------------------------
    parser.add_argument('--wl_data_type', type=str, default='init', help='train or test')
    parser.add_argument('--wl_type', type=str, default='ins_heavy', help='ins_heavy or upd_heavy or dist_shift')
    parser.add_argument('--test_wl_type', type=str, default=None, help='ins_heavy or upd_heavy or dist_shift')

    # ----------------------------------- featurization Params -----------------------------------
    parser.add_argument('--bs', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--clip_size', type=int, default=50, help='Clip size')
    parser.add_argument('--embed_size', type=int, default=64, help='Embedding size')
    parser.add_argument('--pred_hid', type=int, default=128, help='Predictor hidden size')
    parser.add_argument('--ffn_dim', type=int, default=128, help='Feedforward network dimension')
    parser.add_argument('--head_size', type=int, default=12, help='Head size for multihead attention')
    parser.add_argument('--n_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--sch_decay', type=float, default=0.6, help='Scheduler decay rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--newpath', type=str, default='./results/full/cost/', help='Path to save results')
    parser.add_argument('--to_predict', type=str, default='cost', help='The metric to predict')

    # ----------------------------------- pre-training -----------------------------------
    parser.add_argument('--context_length', type=int, default=5, help='context length for pretraining')

    args = parser.parse_args()
    return args


# Define the named tuple for workload paths
WorkloadPaths = namedtuple('WorkloadPaths', [
    'whole_dataset_dir',
    "used_dataset_dir",
    'create_table_file_path',
    'create_index_file_path',
    'base_query_file_path',
    'output_path',
    'tables_info_path',
    "workload_histogram_ckpt_dir"
])


def init_system_dirs(args):
    print(f"Current working directory: {os.getcwd()}")
    file_utils.detect_and_create_dir(args.output_path)
    tables_info_path = os.path.join(args.output_path, "ori_table_info.json")
    workload_histogram_ckpt_dir = os.path.join(args.output_path, 'histogram_ckpt')
    result = WorkloadPaths(
        args.whole_dataset_dir,
        args.used_dataset_dir,
        args.create_table_file_path,
        args.create_index_file_path,
        args.base_query_file_path,
        args.output_path,
        tables_info_path,
        workload_histogram_ckpt_dir
    )

    return False, result


if __name__ == '__main__':
    _args = get_stats_arg_parser()
    init_system_dirs(_args)
