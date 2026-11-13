import os
from copy import deepcopy

import torch
import torch.utils.data
import numpy as np
import pandas as pd

from utils import get_neighbor_finder


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


class DygDataset(torch.utils.data.Dataset):
    def __init__(self, config, split_flag, split_list=None):
        if split_list is None:
            split_list = [0.7, 0.15, 0.15]
        self.config = config
        self.data_name = self.config.data_set
        self.dir_data = self.config.dir_data
        dataset_name = '{}/ml_{}'.format(self.dir_data, self.data_name)

        self.full_data, cur_data, self.positive_eids, self.edge_features, self.node_features = \
            self.get_data(dataset_name, split_flag, split_list)
        self.ngh_finder = get_neighbor_finder(cur_data, uniform=False)
        self.index_start = self.positive_eids[0]

    def get_data(self, dataset_name, split_flag, split_list):
        graph_df = pd.read_csv('{}.csv'.format(dataset_name))
        edge_features = np.load('{}.npy'.format(dataset_name))
        node_features = np.load('{}_node.npy'.format(dataset_name))
        val_time, test_time = list(np.quantile(graph_df.ts, [split_list[0], 1.0 - split_list[2]]))

        sources = graph_df.u.values
        destinations = graph_df.i.values
        edge_idxs = graph_df.idx.values
        labels = graph_df.label.values
        timestamps = graph_df.ts.values

        num_lines = len(sources)

        train_mask = np.where(timestamps <= val_time)[0]
        test_mask = np.where(timestamps > test_time)[0]
        val_mask = np.where(np.logical_and(timestamps <= test_time, timestamps > val_time))[0]

        full_data_src = Data(sources, destinations, timestamps, edge_idxs, labels)
        self.num_nodes = full_data_src.n_unique_nodes
        full_data = deepcopy(full_data_src)

        if split_flag == 'train':
            cur_data = deepcopy(Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                                     edge_idxs[train_mask], labels[train_mask]))
            positive_eids = train_mask

            if self.config.use_processed_data:
                df_path = f'{self.dir_data}/processed/ml_{self.data_name}_processed_train.csv'
                if not os.path.exists(df_path):
                    raise RuntimeError(f"data not processed: {df_path}")

                graph_df_p = pd.read_csv(df_path)

                sources_p = graph_df_p.u.values
                destinations_p = graph_df_p.i.values
                edge_idxs_p = graph_df_p.idx.values
                labels_p = graph_df_p.label.values
                timestamps_p = graph_df_p.ts.values

                num_lines_p = len(sources_p)

                full_data = Data(sources_p, destinations_p, timestamps_p, edge_idxs_p, labels_p)
                positive_eids = [i for i in range(num_lines_p)]
        elif split_flag == 'valid':
            cur_data = full_data
            positive_eids = val_mask
        elif split_flag == 'test':
            cur_data = full_data
            positive_eids = test_mask
        else:
            cur_data = full_data
            positive_eids = np.where(timestamps >= 0)

        if edge_features.shape[0] - 1 == num_lines:
            edge_features = edge_features
        elif edge_features.shape[0] == num_lines:
            edge_features = np.concatenate([np.zeros([1, edge_features.shape[1]]), edge_features], axis=0)
        else:
            raise RuntimeError("edge features do not match the edge list")

        return full_data, cur_data, positive_eids, edge_features, node_features

    def edge_padding(self, neigh_edge, neigh_time, edge_feat, src_neigh_idx, source_node):
        neigh_edge = np.concatenate((neigh_edge, np.tile(source_node.reshape(-1, 1), (1, 2))), axis=0)
        neigh_time = np.concatenate((neigh_time, np.zeros([1], dtype=neigh_time.dtype)), axis=0)
        edge_feat = np.concatenate((edge_feat, np.zeros([1, edge_feat.shape[1]], dtype=edge_feat.dtype)),
                                   axis=0)
        src_neigh_idx = np.concatenate((src_neigh_idx, np.zeros([1], dtype=src_neigh_idx.dtype)), axis=0)
        return neigh_edge, neigh_time, edge_feat, src_neigh_idx

    def __getitem__(self, item):
        item += self.index_start

        source_node = self.full_data.sources[item]
        dest_node = self.full_data.destinations[item]
        current_time = self.full_data.timestamps[item]
        label = self.full_data.labels[item]

        src_neigh_edge, src_neigh_time, src_neigh_idx = self.ngh_finder.get_temporal_neighbor_all(source_node,
                                                                                                  current_time,
                                                                                                  self.config.n_layer,
                                                                                                  self.config.n_neighbors
                                                                                                  )
        src_edge_feature = self.edge_features[src_neigh_idx].astype(np.float32)
        src_edge_to_time = current_time - src_neigh_time
        src_center_node_idx = np.reshape(source_node, [-1])
        if src_neigh_edge.shape[0] == 0:
            src_neigh_edge, src_edge_to_time, src_edge_feature, src_neigh_idx = self.edge_padding(
                src_neigh_edge, src_edge_to_time, src_edge_feature, src_neigh_idx, src_center_node_idx)
        dst_neigh_edge, dst_neigh_time, dst_neigh_idx = self.ngh_finder.get_temporal_neighbor_all(dest_node,
                                                                                                  current_time,
                                                                                                  self.config.n_layer,
                                                                                                  self.config.n_neighbors
                                                                                                  )

        dst_edge_feature = self.edge_features[dst_neigh_idx].astype(np.float32)
        dst_edge_to_time = current_time - dst_neigh_time
        dst_center_node_idx = np.reshape(dest_node, [-1])
        if dst_neigh_edge.shape[0] == 0:
            dst_neigh_edge, dst_edge_to_time, dst_edge_feature, dst_neigh_idx = self.edge_padding(
                dst_neigh_edge, dst_edge_to_time, dst_edge_feature, dst_neigh_idx, dst_center_node_idx)

        label = np.reshape(label, [-1])
        current_time = np.reshape(current_time, [-1])

        return {
            'src_center_node_idx': src_center_node_idx,
            'src_neigh_edge': torch.from_numpy(src_neigh_edge),
            'src_edge_feature': torch.from_numpy(src_edge_feature),
            'src_edge_to_time': torch.from_numpy(src_edge_to_time.astype(np.float32)),

            'dst_center_node_idx': dst_center_node_idx,
            'dst_neigh_edge': torch.from_numpy(dst_neigh_edge),
            'dst_edge_feature': torch.from_numpy(dst_edge_feature),
            'dst_edge_to_time': torch.from_numpy(dst_edge_to_time.astype(np.float32)),

            'current_time': torch.from_numpy(current_time),
            'label': torch.from_numpy(label)
        }

    def __len__(self):
        return len(self.positive_eids)


class Collate:
    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size

    def dyg_collate_fn(self, batch):
        src_edge_feat = torch.cat([b['src_edge_feature'] for b in batch], dim=0)
        src_edge_to_time = torch.cat([b['src_edge_to_time'] for b in batch], dim=0)
        ori_src_center_node_idx = np.concatenate([b['src_center_node_idx'] for b in batch], axis=0)
        ori_src_neigh_edge = torch.cat([b['src_neigh_edge'] for b in batch], dim=0)
        dst_edge_feat = torch.cat([b['dst_edge_feature'] for b in batch], dim=0)
        dst_edge_to_time = torch.cat([b['dst_edge_to_time'] for b in batch], dim=0)
        ori_dst_center_node_idx = np.concatenate([b['dst_center_node_idx'] for b in batch], axis=0)
        ori_dst_neigh_edge = torch.cat([b['dst_neigh_edge'] for b in batch], dim=0)

        label = torch.cat([b['label'] for b in batch], dim=0)
        current_time = torch.cat([b['current_time'] for b in batch], dim=0)

        return {
            'src_edge_feat': src_edge_feat,
            'src_edge_to_time': src_edge_to_time,
            'ori_src_center_node_idx': torch.from_numpy(ori_src_center_node_idx),
            'ori_src_neigh_edge': ori_src_neigh_edge,

            'dst_edge_feat': dst_edge_feat,
            'dst_edge_to_time': dst_edge_to_time,
            'ori_dst_center_node_idx': torch.from_numpy(ori_dst_center_node_idx),
            'ori_dst_neigh_edge': ori_dst_neigh_edge,

            'current_time': current_time,
            'labels': label
        }
