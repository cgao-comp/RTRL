import torch
from typing import Union


class HistoryBank:
    def __init__(
            self,
            max_length: int,
            dtype: torch.dtype = torch.long,
            device: Union[str, torch.device] = "cpu",
            requires_grad: bool = False,
            order: str = "forward"
    ):
        self.max_length = max_length
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad
        self.order = order

        self.bank = torch.zeros(max_length, dtype=dtype,
                                requires_grad=requires_grad).to(self.device)
        self.size = 0
        self.pointer = 0

    def push(self, tensor: torch.Tensor, label: torch.Tensor = None):
        if tensor.dim() != 1:
            raise ValueError(f"only 1-D tensors are supported, got {tensor.dim()} dimensions")
        if label is not None:
            tensor = tensor[label < 1.0]

        tensor = tensor.to(self.device, dtype=self.dtype)
        n = tensor.size(0)

        if n == 0:
            return

        if self.size + n <= self.max_length:
            if self.pointer + n <= self.max_length:
                self.bank[self.pointer:self.pointer + n] = tensor
                self.pointer += n
            else:
                first_part = self.max_length - self.pointer
                self.bank[self.pointer:] = tensor[:first_part]
                self.bank[:n - first_part] = tensor[first_part:]
                self.pointer = n - first_part
            self.size += n
        else:
            self._push_fifo(tensor)

    def _push_fifo(self, tensor: torch.Tensor):
        n = tensor.size(0)

        if n >= self.max_length:
            self.bank[:] = tensor[-self.max_length:]
            self.size = self.max_length
            self.pointer = 0
            return

        if self.pointer + n <= self.max_length:
            self.bank[self.pointer:self.pointer + n] = tensor
            self.pointer += n
        else:
            first_part = self.max_length - self.pointer
            self.bank[self.pointer:] = tensor[:first_part]
            self.bank[:n - first_part] = tensor[first_part:]
            self.pointer = n - first_part

        if self.size < self.max_length:
            self.size = min(self.size + n, self.max_length)

    def sample(self, n: int) -> torch.Tensor:
        indices = torch.randint(0, self.max_length, (n,), device=self.device)
        return self.bank[indices]

    def sample_sequential(self, n: int) -> torch.Tensor:
        start = self.pointer

        if self.order == "forward":
            indices = (start + torch.arange(n, device=self.device)) % self.max_length
        else:
            indices = (start - torch.arange(n, device=self.device)) % self.max_length

        return self.bank[indices]

    def clear(self):
        self.bank.zero_()
        self.size = 0
        self.pointer = 0
