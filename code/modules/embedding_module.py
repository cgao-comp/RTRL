import torch
from torch import nn

from modules.temporal_attention import TemporalAttentionLayer


class GraphAttentionEmbedding(nn.Module):
    def __init__(self, time_encoder, n_layers, node_features_dims, edge_features_dims,
                 time_features_dim, hidden_dim, n_heads=2, dropout=None, config=None):
        super().__init__()

        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = node_features_dims
        self.n_edge_features = edge_features_dims
        self.n_time_features = time_features_dim
        self.dropout = dropout
        self.embedding_dimension = hidden_dim
        self.config = config

        self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
            n_node_features=node_features_dims,
            n_edge_features=edge_features_dims,
            time_dim=time_features_dim,
            n_head=n_heads,
            dropout=dropout,
            output_dimension=hidden_dim,
            config=self.config)
            for _ in range(n_layers)])

    def compute_embedding(self, neigh_edge, edge_to_time, edge_feat, node_feat, edge_mask=None, sample_ratio=None):
        n_layers = self.n_layers
        assert (n_layers >= 0)

        temp_node_feat = node_feat
        src_time_embeddings = self.time_encoder(torch.zeros_like(edge_to_time))
        edge_time_embeddings = self.time_encoder(edge_to_time)

        for layer in range(n_layers):
            temp_node_feat = self.aggregate(n_layers, temp_node_feat,
                                            neigh_edge,
                                            edge_feat,
                                            src_time_embeddings,
                                            edge_time_embeddings)

        out = temp_node_feat
        return out

    def aggregate(self, n_layer, node_features, edge_index,
                  edge_feature,
                  src_time_features, edge_time_embeddings):
        attention_model = self.attention_models[n_layer - 1]

        source_embedding = attention_model(node_features,
                                           edge_index,
                                           edge_feature,
                                           src_time_features,
                                           edge_time_embeddings)

        return source_embedding
