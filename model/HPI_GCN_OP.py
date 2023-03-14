import torch
from torch import nn, einsum
from .REP_block import REPconv_bn, REPTcn_Block, REPTcn_Block9
import math
from einops import rearrange
import numpy as np


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class channel_gcn6(nn.Module):
    def __init__(self, in_channels, out_channels, A, heads=3):
        super().__init__()
        self.heads = heads
        self.to_v = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.to_out = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.PA = nn.Parameter(torch.ones(self.heads, 25, 25))
        self.SA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x):
        out = self.to_v(x)
        out = rearrange(out, 'b (a h) t v -> b a h t v', a=self.heads)
        out = einsum('a i v, b a h t v -> b a h t i', self.PA, out)
        out = rearrange(out, 'b a h t i -> b (a h) t i')
        out = self.to_out(out)
        return out

class gcn_block(nn.Module):
    def __init__(self, A, in_channels=64, out_channels=128, residual=False, deploy=False, heads=3):
        super(gcn_block, self).__init__()

        self.space_att = channel_gcn6(in_channels=in_channels, out_channels=out_channels, A=A, heads=heads)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = REPconv_bn(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                       deploy=deploy, nonlinear=None, single_init=True)

    def forward(self, x):
        out = self.norm(self.space_att(x)) + self.residual(x)
        return self.relu(out)



class basic_block(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, deploy=False):
        super(basic_block, self).__init__()

        self.gcn = gcn_block(A=A, in_channels=in_channels, out_channels=out_channels,
                             residual=residual, deploy=deploy, heads=8)

        # self.tcn1 = REPTcn_Block(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=stride,
        #                          padding=5//2, groups=1, deploy=deploy)  # Rep_TCN
        self.tcn1 = REPTcn_Block9(in_channels=out_channels, out_channels=out_channels, kernel_size=9, stride=stride,
                                 padding=9//2, groups=1, deploy=deploy)  # Rep_TCN_K9

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = REPconv_bn(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                       deploy=deploy, nonlinear=None, single_init=True)

    def forward(self, x):
        out = self.gcn(x)
        out = self.bn(self.tcn1(out)) + self.residual(x)
        out = self.relu(out)
        return out


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 deploy=False, dropout=0., Is_joint=True, K=0):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.Is_joint = Is_joint
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l2 = basic_block(base_channel, base_channel, A, deploy=deploy)
        self.l3 = basic_block(base_channel, base_channel, A, deploy=deploy)
        self.l4 = basic_block(base_channel, base_channel, A, deploy=deploy)
        self.l5 = basic_block(base_channel, base_channel * 2, A, stride=2, deploy=deploy)
        self.l6 = basic_block(base_channel * 2, base_channel * 2, A, deploy=deploy)
        self.l7 = basic_block(base_channel * 2, base_channel * 2, A, deploy=deploy)
        self.l8 = basic_block(base_channel * 2, base_channel * 4, A, stride=2, deploy=deploy)
        self.l9 = basic_block(base_channel * 4, base_channel * 4, A, deploy=deploy)
        self.l10 = basic_block(base_channel * 4, base_channel * 4, A, deploy=deploy)

        self.K = K
        if K != 0:
            I = np.eye(25)
            self.A = torch.from_numpy(I - np.linalg.matrix_power(self.graph.A_outward_binary, K)).type(torch.float32)

        if self.Is_joint:
            self.input_map = nn.Sequential(
                nn.Conv2d(in_channels, base_channel // 2, 1),
                nn.BatchNorm2d(base_channel // 2),
                nn.LeakyReLU(0.1),
            )
            self.diff_map1 = nn.Sequential(
                nn.Conv2d(in_channels, base_channel // 8, 1),
                nn.BatchNorm2d(base_channel // 8),
                nn.LeakyReLU(0.1),
            )
            self.diff_map2 = nn.Sequential(
                nn.Conv2d(in_channels, base_channel // 8, 1),
                nn.BatchNorm2d(base_channel // 8),
                nn.LeakyReLU(0.1),
            )
            self.diff_map3 = nn.Sequential(
                nn.Conv2d(in_channels, base_channel // 8, 1),
                nn.BatchNorm2d(base_channel // 8),
                nn.LeakyReLU(0.1),
            )
            self.diff_map4 = nn.Sequential(
                nn.Conv2d(in_channels, base_channel // 8, 1),
                nn.BatchNorm2d(base_channel // 8),
                nn.LeakyReLU(0.1),
            )
        else:
            self.l1 = REPconv_bn(in_channels, base_channel, kernel_size=1, stride=1, padding=0, deploy=deploy,
                                 nonlinear=None, single_init=True)

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if dropout:
            self.drop_out = nn.Dropout(dropout)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        if self.K != 0:
            x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
            x = self.A.to(x.device).expand(N * M * T, -1, -1) @ x
            x = rearrange(x, '(n m t) v c -> n c t v m', m=M, t=T)

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        if self.Is_joint:
            dif1 = x[:, :, 1:] - x[:, :, 0:-1]
            dif1 = torch.cat([dif1.new(N * M, C, 1, V).zero_(), dif1], dim=-2)
            dif2 = x[:, :, 2:] - x[:, :, 0:-2]
            dif2 = torch.cat([dif2.new(N * M, C, 2, V).zero_(), dif2], dim=-2)
            dif3 = x[:, :, :-1] - x[:, :, 1:]
            dif3 = torch.cat([dif3, dif3.new(N * M, C, 1, V).zero_()], dim=-2)
            dif4 = x[:, :, :-2] - x[:, :, 2:]
            dif4 = torch.cat([dif4, dif4.new(N * M, C, 2, V).zero_()], dim=-2)
            x = torch.cat((self.input_map(x), self.diff_map1(dif1), self.diff_map2(dif2), self.diff_map3(dif3),
                           self.diff_map4(dif4)), dim=1)
        else:
            x = self.l1(x)

        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)



