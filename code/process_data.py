import os, argparse, shutil
from datetime import datetime
import random
import pandas as pd
import numpy as np

random.seed(73)
np.random.seed(73)

from utils import logger_config
from paths import DATA_DIR, WALK_LOG_DIR

parser = argparse.ArgumentParser(description='My Works: Process Data')
parser.add_argument('--dataset_name', type=str, default="amazon")
parser.add_argument('--split_ratio', type=int, default=70)
parser.add_argument('--walk_length', type=int, default=10)
parser.add_argument('--num_walks', type=int, default=5)
parser.add_argument('--restart_prob', type=float, default=0.15)
parser.add_argument('--append_num', type=int, default=1024)
args = parser.parse_args()

os.makedirs(WALK_LOG_DIR, exist_ok=True)
logger = logger_config(os.path.join(WALK_LOG_DIR, f"{args.dataset_name}_walk.log"),
                       logging_name=f"restart_prob: {args.restart_prob}")
logger.info("##########################################################")
logger.info(args)

all_dataset_name_list = ["wikipedia", "reddit", "mooc", "bitcoinalpha", "bitcoinotc", "amazon"]
dataset_name = args.dataset_name
split_ratio = args.split_ratio

graph_df = pd.read_csv(os.path.join(DATA_DIR, 'ml_{}.csv'.format(dataset_name)))
val_time, test_time = list(np.quantile(graph_df.ts, [split_ratio / 100, 1.0]))

sources = graph_df.u.values
destinations = graph_df.i.values
edge_idxs = graph_df.idx.values
labels = graph_df.label.values
timestamps = graph_df.ts.values

train_mask = np.where(timestamps <= val_time)[0]


def random_walk_sampling(graph_dict, start_nodes, walk_length=10, num_walks=5, restart_prob=0.15):
    visited_nodes = set(start_nodes)

    for start_node in start_nodes:
        for _ in range(num_walks):
            current_node = start_node
            for step in range(walk_length):
                if random.random() < restart_prob:
                    current_node = start_node
                else:
                    if current_node in graph_dict and graph_dict[current_node]:
                        current_node = random.choice(graph_dict[current_node])
                    else:
                        break

                visited_nodes.add(current_node)

    return visited_nodes


def build_adjacency_list(sources, destinations, num_lines):
    graph_dict = {}
    for i in range(num_lines):
        u, v = sources[i], destinations[i]

        if u not in graph_dict:
            graph_dict[u] = []
        if v not in graph_dict:
            graph_dict[v] = []

        graph_dict[u].append(v)
        graph_dict[v].append(u)

    return graph_dict

if __name__ == '__main__':
    train_sources = sources[train_mask]
    train_destinations = destinations[train_mask]
    train_edge_idxs = edge_idxs[train_mask]
    train_labels = labels[train_mask]
    train_timestamps = timestamps[train_mask]

    num_lines = len(train_labels)

    positive_nodes = set()
    for i in range(num_lines):
        if train_labels[i] == 1.0:
            positive_nodes.add(train_sources[i])
            positive_nodes.add(train_destinations[i])

    logger.info(f"Found {len(positive_nodes)} positive nodes")

    graph_dict = build_adjacency_list(train_sources, train_destinations, num_lines)

    visited_nodes = random_walk_sampling(
        graph_dict,
        positive_nodes,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        restart_prob=args.restart_prob
    )

    logger.info(f"Total visited nodes after random walk: {len(visited_nodes)}")

    selected_indices = set()
    for i in range(num_lines):
        if (train_sources[i] in visited_nodes and
                train_destinations[i] in visited_nodes):
            selected_indices.add(i)
    for i in range(num_lines - args.append_num, num_lines, 1):
        selected_indices.add(i)

    selected_indices = sorted(list(selected_indices))

    selected_data = {
        'u': train_sources[selected_indices],
        'i': train_destinations[selected_indices],
        'idx': train_edge_idxs[selected_indices],
        'label': train_labels[selected_indices],
        'ts': train_timestamps[selected_indices]
    }

    selected_df = pd.DataFrame(selected_data)

    save_root_dir = DATA_DIR
    os.makedirs(f"{save_root_dir}/processed/", exist_ok=True)
    file_path = f'{save_root_dir}/processed/ml_{dataset_name}_processed_train.csv'
    if os.path.exists(file_path):
        target_dir = f"{save_root_dir}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        processed_dir = os.path.join(target_dir, f"processed_{timestamp}")
        os.makedirs(processed_dir, exist_ok=True)

        for move_old_file in all_dataset_name_list:
            new_filename = f"ml_{move_old_file}_processed_train.csv"
            target_path = os.path.join(processed_dir, new_filename)
            old_file_path = f'{save_root_dir}/processed/ml_{move_old_file}_processed_train.csv'

            if os.path.exists(old_file_path):
                try:
                    shutil.move(old_file_path, target_path)
                    logger.info(f"old file moved: {file_path} -> {target_path}")
                except Exception as e:
                    logger.info(f"failed to move old file: {e}")

    selected_df.to_csv(file_path, index=False)
    logger.info(f"Selected data saved to {file_path}")
    logger.info(f"Original training set size: {num_lines}")
    logger.info(f"Selected training set size: {len(selected_indices)}")
    logger.info(f"Number of positive nodes: {len(positive_nodes)}")
    logger.info(f"Total visited nodes: {len(visited_nodes)}")
    logger.info(f"================================ END ================================")
