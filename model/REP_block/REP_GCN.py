import torch
from torch import nn, einsum
from .transforms import *


class REP_GCN0(nn.Module):
    def __init__(self, heads=3, num_point=25,
                 deploy=False, nonlinear=None):
        super(REP_GCN0, self).__init__()
        self.deploy = deploy
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        self.heads = heads
        self.num_point = num_point
        if deploy:
            self.A = nn.Parameter(torch.zeros(self.num_point, self.num_point))
        else:
            self.PA = torch.nn.Parameter(torch.Tensor(torch.rand(self.heads, self.num_point, self.num_point) * 0.2 + 0.9))

    def get_equivalent_kernel_bias(self):
        return torch.sum(self.PA, 0)

    def switch_to_deploy(self):
        if hasattr(self, 'A'):
            return

        A = self.get_equivalent_kernel_bias()
        self.A = nn.Parameter(torch.zeros(self.num_point, self.num_point))
        self.A.data = A
        for para in self.parameters():
            para.detach_()
        self.__delattr__('PA')

    def forward(self, inputs):
        if hasattr(self, 'A'):
            return self.nonlinear(einsum('i v, b c t v -> b c t i', self.A, inputs))
        y = None
        for i in range(self.heads):
            out = einsum('i v, b c t v -> b c t i', self.PA[i], inputs)
            y = y+out if y is not None else out
        return self.nonlinear(y)
