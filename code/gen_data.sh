#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

python process_data.py --dataset_name wikipedia    --split_ratio 70 --walk_length 3 --num_walks 3
python process_data.py --dataset_name reddit       --split_ratio 70 --walk_length 3 --num_walks 3
python process_data.py --dataset_name mooc         --split_ratio 70 --walk_length 1 --num_walks 2
python process_data.py --dataset_name bitcoinalpha --split_ratio 70 --walk_length 2 --num_walks 3
python process_data.py --dataset_name bitcoinotc   --split_ratio 70 --walk_length 2 --num_walks 3
python process_data.py --dataset_name amazon       --split_ratio 70 --walk_length 3 --num_walks 3
