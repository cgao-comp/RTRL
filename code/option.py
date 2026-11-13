import argparse
import sys, os

RUN_WIN = False
DATA_WORKER = 24
if sys.platform.startswith('win'):
    print(" >>>>>>>>>>>>>>>>> run windows <<<<<<<<<<<<<<<<<<<<")
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    RUN_WIN = True
    DATA_WORKER = 0
else:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from paths import ROOT_DIR, DATA_DIR, CKPT_DIR, LOG_DIR, WALK_LOG_DIR

parser = argparse.ArgumentParser(description='My Works')
parser.add_argument('--dir_data', type=str, default=DATA_DIR)
parser.add_argument('--data_set', type=str, default='bitcoinalpha')
parser.add_argument('--mode', type=str, default='RllmE')
parser.add_argument('--seed', type=int, default=73)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--split', type=str, default='test', choices=['test', 'valid'])

parser.add_argument('--drop_reverse_flag', action='store_false', default=True)
parser.add_argument('--use_processed_data', action='store_false', default=True)
parser.add_argument('--save_auc', type=float, default=0.5)
parser.add_argument('--save_auc_rf', type=float, action='store_true', default=False)
parser.add_argument('--hqb_len', type=int, default=5)
parser.add_argument('--hsb_len', type=int, default=3)
parser.add_argument('--num_tokens', type=int, default=1024)
parser.add_argument('--loss_smooth', type=float, default=0.3)

parser.add_argument('--randn_qwen', action='store_true', default=False)
parser.add_argument('--llm_keep_layers', type=int, default=3)
parser.add_argument('--llm_arch', type=str, default='qwen', choices=['qwen', 'mlp'])
parser.add_argument('--llm_dtype', type=str, default='bfloat16', choices=['bfloat16', 'float32'])
parser.add_argument('--llm_freeze_emb', action='store_true', default=False)
parser.add_argument('--no_push_history', action='store_false', dest='push_history', default=True)
parser.add_argument('--infer_use_history', action='store_true', default=False)
parser.add_argument('--history_order', type=str, default='reverse', choices=['reverse', 'forward'])
parser.add_argument('--freeze_residual_blocks', action='store_true', default=False)

parser.add_argument('--residual_w', type=float, default=0.3)
parser.add_argument('--lambda_temporal', type=float, default=1.0)
parser.add_argument('--lambda_sparse', type=float, default=0.3)
parser.add_argument('--lambda_stability', type=float, default=0.05)
parser.add_argument('--lambda_1', type=float, default=1.0)
parser.add_argument('--lambda_2', type=float, default=1.0)
parser.add_argument('--lambda_3', type=float, default=1.0)

parser.add_argument('--llm_model', type=str, default='Qwen3-0_6B')
parser.add_argument('--llm_dim', type=int, default=1024)

parser.add_argument('--n_neighbors', type=int, default=20, help='Maximum number of connected edge per node')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--n_nodes', type=int, default=2000)
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--num_data_workers', type=int, default=DATA_WORKER)
parser.add_argument('--train_ratio', type=float, default=0.7)

parser.add_argument('--input_dim', type=int, default=172)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--drop_out', type=float, default=None)
parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--learning_rate_drop_step', type=int, default=20)
parser.add_argument('--learning_rate_gamma', type=float, default=0.005)

args = parser.parse_args()
