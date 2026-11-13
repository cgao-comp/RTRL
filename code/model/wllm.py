import torch

from modules.utils import Feat_Process_Layer, MergeLayer_output
from modules.embedding_module import GraphAttentionEmbedding
from modules.time_encoding import TimeEncode_NoGrad

from model.memory_net import GlobalMemory
from model.llm_wb import LLMModel
from model.history_memory import HistoryBank
from model.residual_his import ResidualHis


class wllm(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config

        self.nodes_dim = self.cfg.input_dim
        self.dataset_name = self.cfg.data_set.lower()
        self.model_llm = self.cfg.llm_model
        self.model_llm_dim = self.cfg.llm_dim
        if self.dataset_name == "mooc":
            self.edge_dim = 4
        else:
            self.edge_dim = self.cfg.input_dim
        self.dims = self.cfg.hidden_dim

        self.n_heads = self.cfg.n_heads
        self.dropout = self.cfg.drop_out
        self.n_layers = self.cfg.n_layer
        self.n_nodes = self.cfg.n_nodes + 1
        self.device = self.cfg.device
        self.batch_size = self.cfg.batch_size
        self.residual_w = self.cfg.residual_w

        self.llm = LLMModel(device=self.device, input_dim=self.dims, out_dim=self.dims,
                            dataset_name=self.dataset_name, llm_model_name=self.model_llm,
                            llm_dim=self.model_llm_dim, config=config).to(self.device)

        self.time_encoder = TimeEncode_NoGrad(dimension=self.dims)
        self.embedding_module = GraphAttentionEmbedding(time_encoder=self.time_encoder,
                                                        n_layers=self.n_layers,
                                                        node_features_dims=self.dims,
                                                        edge_features_dims=self.dims,
                                                        time_features_dim=self.dims,
                                                        hidden_dim=self.dims,
                                                        n_heads=self.n_heads, dropout=self.dropout,
                                                        config=self.cfg)

        self.node_preocess_fn = Feat_Process_Layer(self.dims, self.dims)
        self.edge_preocess_fn = Feat_Process_Layer(self.edge_dim, self.dims)

        self.memory_pool = GlobalMemory(self.n_nodes, self.dims, self.device)
        self.affinity_llm_score_src = MergeLayer_output(self.dims * 2, self.dims, drop_out=0.1)

        self.history_queue_length = self.batch_size * self.cfg.hqb_len
        self.history_sample_b = self.cfg.hsb_len
        self.history_sample_size = self.batch_size * self.history_sample_b
        self.history_queue = HistoryBank(max_length=self.history_queue_length,
                                         device=self.device,
                                         order=self.cfg.history_order)
        self.residual_history = ResidualHis(self.history_sample_b, self.dims, self.cfg, device=self.device)

    def train_forward(self, batch_sample, device):
        src_org_edge_feat = batch_sample['src_edge_feat'].to(device)
        src_edge_to_time = batch_sample['src_edge_to_time'].to(device)
        ori_src_center_node_idx = batch_sample['ori_src_center_node_idx'].to(device)
        ori_src_neigh_edge = batch_sample['ori_src_neigh_edge'].to(device)

        dst_org_edge_feat = batch_sample['dst_edge_feat'].to(device)
        dst_edge_to_time = batch_sample['dst_edge_to_time'].to(device)
        ori_dst_center_node_idx = batch_sample['ori_dst_center_node_idx'].to(device)
        ori_dst_neigh_edge = batch_sample['ori_dst_neigh_edge'].to(device)

        current_time = batch_sample['current_time'].to(device)

        node_memory = self.memory_pool.memory_gru.data
        cur_batch_size = current_time.shape[0]
        if self.cfg.push_history:
            self.history_queue.push(ori_src_center_node_idx)
        his_memory = node_memory[self.history_queue.sample_sequential(self.history_sample_b * cur_batch_size)]
        src_node_embedding = self.compute_temporal_embeddings(ori_src_neigh_edge,
                                                              src_edge_to_time,
                                                              src_org_edge_feat,
                                                              node_memory)
        src_root_embedding = src_node_embedding[ori_src_center_node_idx, :]
        dst_node_embedding = self.compute_temporal_embeddings(ori_dst_neigh_edge,
                                                              dst_edge_to_time,
                                                              dst_org_edge_feat,
                                                              node_memory)
        dst_root_embedding = dst_node_embedding[ori_dst_center_node_idx, :]
        src_memory = self.memory_pool.get_updated_memory(ori_src_center_node_idx,
                                                         src_root_embedding)
        dst_memory = self.memory_pool.get_updated_memory(ori_dst_center_node_idx,
                                                         dst_root_embedding)

        llm_in = [src_memory.unsqueeze(1)]
        for i in range(1, self.history_sample_b + 1):
            llm_in.append(his_memory[cur_batch_size * (i - 1): cur_batch_size * i].unsqueeze(1))
        (llm_out) = self.llm(torch.concat(llm_in, dim=1),
                             ori_src_center_node_idx, ori_dst_center_node_idx, current_time)
        llm_out_src = llm_out[:, 0, :].squeeze(1)
        llm_out_his = llm_out[:, 1:, :]
        llm_out_residual = self.residual_history(llm_out_src, llm_out_his)
        src_embedding = src_root_embedding + self.residual_w * llm_out_residual
        logits = self.affinity_llm_score_src(torch.cat([src_embedding, llm_out_src], dim=-1))

        prediction_dict = {}
        prediction_dict['logits'] = logits.squeeze(1)
        prediction_dict['llm_out_src'] = llm_out_src.unsqueeze(dim=1)
        prediction_dict['llm_out_his'] = llm_out_his
        prediction_dict['llm_out_residual'] = llm_out_residual

        self.memory_pool.detach_all()
        return prediction_dict

    @torch.no_grad()
    def infer_forward(self, batch_sample, device):
        src_org_edge_feat = batch_sample['src_edge_feat'].to(device)
        src_edge_to_time = batch_sample['src_edge_to_time'].to(device)
        ori_src_center_node_idx = batch_sample['ori_src_center_node_idx'].to(device)
        ori_src_neigh_edge = batch_sample['ori_src_neigh_edge'].to(device)

        dst_org_edge_feat = batch_sample['dst_edge_feat'].to(device)
        dst_edge_to_time = batch_sample['dst_edge_to_time'].to(device)
        ori_dst_center_node_idx = batch_sample['ori_dst_center_node_idx'].to(device)
        ori_dst_neigh_edge = batch_sample['ori_dst_neigh_edge'].to(device)

        current_time = batch_sample['current_time'].to(device)
        node_memory = self.memory_pool.memory_gru.data
        cur_batch_size = current_time.shape[0]
        if self.cfg.infer_use_history:
            his_memory = node_memory[self.history_queue.sample_sequential(self.history_sample_b * cur_batch_size)]
        src_node_embedding = self.compute_temporal_embeddings(ori_src_neigh_edge,
                                                              src_edge_to_time,
                                                              src_org_edge_feat,
                                                              node_memory)
        src_root_embedding = src_node_embedding[ori_src_center_node_idx, :]
        dst_node_embedding = self.compute_temporal_embeddings(ori_dst_neigh_edge,
                                                              dst_edge_to_time,
                                                              dst_org_edge_feat,
                                                              node_memory)
        dst_root_embedding = dst_node_embedding[ori_dst_center_node_idx, :]
        src_memory = self.memory_pool.get_updated_memory(ori_src_center_node_idx,
                                                         src_root_embedding)

        dst_memory = self.memory_pool.get_updated_memory(ori_dst_center_node_idx,
                                                         dst_root_embedding)

        llm_in = [src_memory.unsqueeze(1)]
        if self.cfg.infer_use_history:
            for i in range(1, self.history_sample_b + 1):
                llm_in.append(his_memory[cur_batch_size * (i - 1): cur_batch_size * i].unsqueeze(1))
        (llm_out) = self.llm(torch.concat(llm_in, dim=1),
                             ori_src_center_node_idx, ori_dst_center_node_idx, current_time)
        llm_out_src = llm_out[:, 0, :].squeeze(1)

        if self.cfg.infer_use_history:
            llm_out_residual = self.residual_history(llm_out_src, llm_out[:, 1:, :])
        else:
            llm_out_residual = self.residual_history.infer(llm_out_src)
        src_embedding = src_root_embedding + self.residual_w * llm_out_residual
        logits = self.affinity_llm_score_src(torch.cat([src_embedding, llm_out_src], dim=-1))

        prediction_dict = {}
        prediction_dict['logits'] = logits.squeeze(1)

        self.memory_pool.detach_all()
        return prediction_dict

    def compute_temporal_embeddings(self, neigh_edge, edge_to_time, edge_feat, node_feat):
        node_feat = self.node_preocess_fn(node_feat)
        edge_feat = self.edge_preocess_fn(edge_feat)

        node_embedding = self.embedding_module.compute_embedding(neigh_edge, edge_to_time,
                                                                 edge_feat, node_feat)

        return node_embedding
