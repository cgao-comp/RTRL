import torch
from torch import nn
from torch_scatter import scatter


class GlobalMemory(nn.Module):
    def __init__(self, n_nodes: int, hid_dim: int, device: str):
        super().__init__()

        self.n_nodes = n_nodes
        self.device = device
        self.dims = hid_dim

        self.init_memory()
        self.memory_updater = nn.GRUCell(input_size=self.dims,
                                         hidden_size=self.dims)

        self.message_updater = nn.Sequential(
            nn.Linear(self.dims * 2, self.dims),
            nn.ReLU(),
            nn.Linear(self.dims, self.dims),
        )

    def get_updated_memory(self, nodes, node_embedding, update_memory=True):
        node_memory = self.memory_gru[nodes]
        curr_message = torch.cat([node_memory,
                                  node_embedding],
                                 dim=1)

        (unique_sources, idx) = torch.unique(nodes, return_inverse=True)
        message = scatter(curr_message, idx, reduce='mean', dim=0)
        message = self.message_updater(message)
        updated_message = self.memory_updater(message)

        all_memory = self.memory_gru.data.clone()
        all_memory[unique_sources] = updated_message
        if update_memory:
            self.memory_gru[unique_sources] = updated_message
        return all_memory[nodes]

    def detach_all(self):
        self.memory_gru.detach_()

    def init_memory(self):
        self.memory_gru = nn.Parameter(torch.zeros((self.n_nodes, self.dims)).to(self.device),
                                       requires_grad=False)

    def add_memory_nodes(self, add_nodes: int):
        current_nodes, dims = self.memory_gru.shape
        new_memory_gru = torch.zeros((current_nodes + add_nodes, dims),
                                     device=self.device)
        new_memory_gru[:current_nodes] = self.memory_gru.data
        self.memory_gru = nn.Parameter(new_memory_gru, requires_grad=False)

        self.n_nodes = current_nodes + add_nodes
