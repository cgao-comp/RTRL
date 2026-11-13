from collections import deque

import torch
from torch import nn
import torch.nn.functional as F


class ResidualHis(nn.Module):
    def __init__(self, his_len: int, dim: int, config, device: str = "cuda:0"):
        super(ResidualHis, self).__init__()

        self.his_len = his_len
        self.device = device

        self.blocks = [] if config.freeze_residual_blocks else nn.ModuleList()
        for _ in range(his_len):
            self.blocks.append(nn.Sequential(
                nn.Linear(dim, dim, bias=False),
                nn.GroupNorm(2, dim)
            ).to(device))

        self.loss_fn = ResidualLoss(his_len=his_len,
                                    lambda_temporal=config.lambda_temporal,
                                    lambda_sparse=config.lambda_sparse,
                                    lambda_stability=config.lambda_stability,
                                    device=device)

    def forward(self, x, x_his):
        assert x_his.shape[1] == self.his_len

        residual_mat = x_his[:, self.his_len - 1, :]
        for i in range(self.his_len - 2, -1, -1):
            residual_mat = residual_mat + self.blocks[i](x_his[:, i, :])
        residual_mat = residual_mat + self.blocks[self.his_len - 1](x)

        residual_mat = F.relu(residual_mat)
        return residual_mat

    @torch.no_grad()
    def infer(self, x):
        residual_mat = self.blocks[self.his_len - 1](x)
        residual_mat = F.relu(residual_mat)
        return residual_mat


class ResidualLoss(nn.Module):
    def __init__(self, his_len: int, lambda_temporal: float = 1.0,
                 lambda_sparse: float = 0.1, lambda_stability: float = 0.05, device: str = "cuda:0"):
        super(ResidualLoss, self).__init__()
        self.device = device
        self.his_len = his_len
        self.lambda_temporal = lambda_temporal
        self.lambda_sparse = lambda_sparse
        self.lambda_stability = lambda_stability

        self.temporal_weights = torch.tensor([0.5 ** i for i in range(his_len)],
                                             dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)
        self.residual_his_list = deque(maxlen=his_len)

    def forward(self, residual, x, x_his):
        batch_size, dim = residual.shape

        temporal_loss = self._temporal_consistency_loss(residual, x_his)

        sparse_loss = self._sparsity_loss(residual)

        stability_loss = 0.0
        push_residual = self.push_residual(residual)
        residual_his = torch.stack(list(self.residual_his_list), dim=1).to(self.device)
        if residual_his is not None and residual_his.shape[1] >= 2:
            stability_loss = self._stability_loss(push_residual, residual_his)

        total_loss = (temporal_loss * self.lambda_temporal +
                      sparse_loss * self.lambda_sparse +
                      stability_loss * self.lambda_stability)

        return {
            'total_loss': total_loss,
            'temporal_loss': temporal_loss,
            'sparse_loss': sparse_loss,
            'stability_loss': stability_loss
        }

    def _temporal_consistency_loss(self, residual, x_his):
        weighted_history = (x_his * self.temporal_weights.to(x_his.device)).sum(dim=1)

        cosine_sim = F.cosine_similarity(residual, weighted_history, dim=-1)
        temporal_loss = 1.0 - cosine_sim.mean()

        return temporal_loss

    def _sparsity_loss(self, residual):
        sparse_loss = residual.abs().mean()

        residual_normalized = F.softmax(residual.abs(), dim=-1)
        entropy = -(residual_normalized * torch.log(residual_normalized + 1e-10)).sum(dim=-1).mean()

        return sparse_loss + 0.1 * entropy

    def _stability_loss(self, current_residual, residual_his):
        recent_residual = residual_his[:, -1, :]

        residual_diff = current_residual - recent_residual

        residual_norm = recent_residual.norm(dim=-1, keepdim=True) + 1e-8
        normalized_diff = residual_diff / residual_norm

        stability_loss = F.smooth_l1_loss(normalized_diff,
                                          torch.zeros_like(normalized_diff),
                                          reduction='mean')

        return stability_loss

    def _reconstruction_loss(self, residual, x, x_his):
        return 0.0

    def push_residual(self, residual):
        push_residual = residual
        if len(self.residual_his_list) != 0:
            last_bs, _ = self.residual_his_list[-1].shape
            cur_bs, _ = push_residual.shape

            if cur_bs < last_bs:
                pad_rows = last_bs - cur_bs
                push_residual = F.pad(
                    push_residual,
                    pad=(0, 0, 0, pad_rows),
                    mode="constant",
                    value=0.0
                )
        self.residual_his_list.append(push_residual.detach())
        return push_residual


class MultiScaleResidualLoss(nn.Module):
    def __init__(self, his_lens=[1, 3, 5], weights=[0.5, 0.3, 0.2], device: str = "cuda:0"):
        super(MultiScaleResidualLoss, self).__init__()
        self.losses = nn.ModuleList([
            ResidualLoss(his_len, device) for his_len in his_lens
        ])
        self.weights = weights

    def forward(self, residuals_list, x, x_his_list):
        total_loss = 0.0
        loss_dict = {}

        for i, (residual, x_his) in enumerate(zip(residuals_list, x_his_list)):
            loss_dict_i = self.losses[i](residual, x, x_his)
            scale_loss = loss_dict_i['total_loss'] * self.weights[i]
            total_loss += scale_loss

            for key, value in loss_dict_i.items():
                loss_dict[f'scale{i}_{key}'] = value

        loss_dict['total_loss'] = total_loss
        return loss_dict
