import torch


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h) + x2


class MergeLayer_output(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3=1024, dim4=1, drop_out=0.2):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1, dim3, bias=False)
        self.fc2 = torch.nn.Linear(dim3, dim3, bias=False)
        self.fc3 = torch.nn.Linear(dim3, dim2, bias=False)
        self.fc4 = torch.nn.Linear(dim2, dim4, bias=False)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop_out)

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        h = self.dropout(self.act(self.fc3(h)))
        h = self.fc4(h)
        return h


class Feat_Process_Layer(torch.nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1, dim2)
        self.fc2 = torch.nn.Linear(dim2, dim2)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        h = self.act(self.fc1(x))
        return self.fc2(h)


def Drop_Batch_Data(data, drop_ratio, batch_dim=0, random_seed=None):
    if not isinstance(data, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if drop_ratio < 0 or drop_ratio >= 1:
        raise ValueError("drop_ratio must be in [0, 1)")
    if random_seed is not None:
        torch.manual_seed(random_seed)

    n_samples = data.size(batch_dim)
    n_keep = int(n_samples * (1 - drop_ratio))
    if n_keep == n_samples:
        return data.clone()

    n_keep = max(1, n_keep)
    keep_indices = torch.randperm(n_samples)[:n_keep].to(data.device)
    result = torch.index_select(data, batch_dim, keep_indices)

    return result
