import torch
import numpy as np


class TimeEncode(torch.nn.Module):
    def __init__(self, dimension):
        super(TimeEncode, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 1.5, dimension)))
                                           .float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        t = torch.log(t + 1)
        t = t.unsqueeze(dim=1)

        output = torch.cos(self.w(t))

        return output


class TimeEncode_NoGrad(torch.nn.Module):
    def __init__(self, dimension):
        super(TimeEncode_NoGrad, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .float().reshape(dimension, -1), requires_grad=False)
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float(), requires_grad=False)

    @torch.no_grad()
    def forward(self, t):
        t = torch.log(t + 1)
        t = t.unsqueeze(dim=1)

        output = torch.cos(self.w(t))

        return output


class No_TimeEncode(torch.nn.Module):
    def __init__(self, dimension):
        super(No_TimeEncode, self).__init__()

        self.dimension = dimension

    @torch.no_grad()
    def forward(self, t):
        batch_size = t.shape[0]

        padded_tensor = torch.zeros(batch_size, self.dimension,
                                    dtype=t.dtype,
                                    device=t.device)

        padded_tensor[:, 0] = t

        return padded_tensor
