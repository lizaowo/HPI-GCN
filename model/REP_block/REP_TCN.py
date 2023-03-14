import torch
import torch.nn as nn
from .transforms import *


def conv_bn(in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0,0), dilation=1, groups=1,
                   padding_mode='zeros'):
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, dilation=dilation, groups=groups,
                           bias=False, padding_mode=padding_mode)
    bn_layer = nn.BatchNorm2d(num_features=out_channels, affine=True)
    se = nn.Sequential()
    se.add_module('conv', conv_layer)
    se.add_module('bn', bn_layer)
    return se


class IdentityBasedConv1x1(nn.Conv2d):

    def __init__(self, channels, groups=1):
        super(IdentityBasedConv1x1, self).__init__(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)

        assert channels % groups == 0
        input_dim = channels // groups
        id_value = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = torch.from_numpy(id_value).type_as(self.weight)
        nn.init.zeros_(self.weight)

    def forward(self, input):
        kernel = self.weight + self.id_tensor.to(self.weight.device)
        result = F.conv2d(input, kernel, None, stride=1, padding=0, dilation=self.dilation, groups=self.groups)
        return result

    def get_actual_kernel(self):
        return self.weight + self.id_tensor.to(self.weight.device)


class BNAndPadLayer(nn.Module):
    def __init__(self,
                 pad_pixels,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(self.bn.running_var + self.bn.eps)
            else:
                pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [0, 0, self.pad_pixels, self.pad_pixels])
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            # output[:, :, :, 0:self.pad_pixels] = pad_values
            # output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class REPconv_bn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 deploy=False, nonlinear=None, single_init=False):
        super(REPconv_bn, self).__init__()
        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        if deploy:
            self.conv_bn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                         stride=(stride, 1), padding=(padding, 0), dilation=dilation, groups=groups, bias=True)
        else:
            self.TCN_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                      stride=(stride, 1), padding=(padding, 0), dilation=dilation, groups=groups)
            if single_init:
                self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transII_fusebn(self.TCN_origin.conv.weight, self.TCN_origin.bn)
        return k_origin, b_origin

    def switch_to_deploy(self):
        if hasattr(self, 'conv_bn'):
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn = nn.Conv2d(in_channels=self.TCN_origin.conv.in_channels, out_channels=self.TCN_origin.conv.out_channels,
                                     kernel_size=self.TCN_origin.conv.kernel_size, stride=self.TCN_origin.conv.stride,
                                     padding=self.TCN_origin.conv.padding, dilation=self.TCN_origin.conv.dilation, groups=self.TCN_origin.conv.groups, bias=True)
        self.conv_bn.weight.data = kernel
        self.conv_bn.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('TCN_origin')

    def forward(self, inputs):
        if hasattr(self, 'conv_bn'):
            return self.nonlinear(self.conv_bn(inputs))

        out = self.TCN_origin(inputs)
        return self.nonlinear(out)

    def single_init(self):
        if hasattr(self, "TCN_origin"):
            torch.nn.init.constant_(self.TCN_origin.bn.weight, 1.0)


class REPTcn_Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 deploy=False, nonlinear=None, single_init=False):
        super(REPTcn_Block, self).__init__()
        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        if deploy:
            self.TCN_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                         stride=(stride, 1), padding=(padding, 0), dilation=dilation, groups=groups, bias=True)

        else:
            self.TCN_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                      stride=(stride, 1), padding=(padding, 0), dilation=dilation, groups=groups)

            self.TCN_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=(stride, 1),
                                   padding=0, groups=groups)

            self.TCN_avg = nn.Sequential()
            self.TCN_avg.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                              stride=1, padding=0, groups=groups, bias=False))
            self.TCN_avg.add_module('bn', BNAndPadLayer(pad_pixels=1, num_features=out_channels))
            self.TCN_avg.add_module('avg', nn.AvgPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=0))
            self.TCN_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

            self.TCN_1x1_kxk = nn.Sequential()
            self.TCN_1x1_kxk.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                            kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.TCN_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=2, num_features=out_channels, affine=True))
            self.TCN_1x1_kxk.add_module('conv2', nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                                            kernel_size=(5, 1), stride=(stride, 1), padding=0, groups=groups, bias=False))
            self.TCN_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

            #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
            if single_init:
                #   Initialize the bn.weight of TCN_origin as 1 and others as 0. This is not the default setting.
                self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transII_fusebn(self.TCN_origin.conv.weight, self.TCN_origin.bn)

        if hasattr(self, 'TCN_1x1'):
            k_1x1, b_1x1 = transII_fusebn(self.TCN_1x1.conv.weight, self.TCN_1x1.bn)
            k_1x1 = transV_multiscale_T(k_1x1, self.kernel_size)  # 把kernal放大
        else:
            k_1x1, b_1x1 = 0, 0

        k_1x1_kxk_first = self.TCN_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transII_fusebn(k_1x1_kxk_first, self.TCN_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transII_fusebn(self.TCN_1x1_kxk.conv2.weight, self.TCN_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second, b_1x1_kxk_second, groups=self.groups)
        k_1x1_kxk_merged = transV_multiscale_T(k_1x1_kxk_merged, self.kernel_size)

        k_avg = transI_avg_T(self.out_channels, 3, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transII_fusebn(k_avg.to(self.TCN_avg.avgbn.weight.device), self.TCN_avg.avgbn)
        if hasattr(self.TCN_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transII_fusebn(self.TCN_avg.conv.weight, self.TCN_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second, b_1x1_avg_second, groups=self.groups)
            k_1x1_avg_merged = transV_multiscale_T(k_1x1_avg_merged, self.kernel_size)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second
            k_1x1_avg_merged = transV_multiscale_T(k_1x1_avg_merged, self.kernel_size)
        return transIV_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'TCN_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.TCN_reparam = nn.Conv2d(in_channels=self.TCN_origin.conv.in_channels, out_channels=self.TCN_origin.conv.out_channels,
                                     kernel_size=self.TCN_origin.conv.kernel_size, stride=self.TCN_origin.conv.stride,
                                     padding=self.TCN_origin.conv.padding, dilation=self.TCN_origin.conv.dilation, groups=self.TCN_origin.conv.groups, bias=True)
        self.TCN_reparam.weight.data = kernel
        self.TCN_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('TCN_origin')
        self.__delattr__('TCN_avg')
        if hasattr(self, 'TCN_1x1'):
            self.__delattr__('TCN_1x1')
        self.__delattr__('TCN_1x1_kxk')

    def forward(self, inputs):

        if hasattr(self, 'TCN_reparam'):
            return self.nonlinear(self.TCN_reparam(inputs))

        out = self.TCN_origin(inputs)
        if hasattr(self, 'TCN_1x1'):
            out += self.TCN_1x1(inputs)
        out += self.TCN_avg(inputs)
        out += self.TCN_1x1_kxk(inputs)
        return self.nonlinear(out)

    def init_gamma(self, gamma_value):
        if hasattr(self, "TCN_origin"):
            torch.nn.init.constant_(self.TCN_origin.bn.weight, gamma_value)
        if hasattr(self, "TCN_1x1"):
            torch.nn.init.constant_(self.TCN_1x1.bn.weight, gamma_value)
        if hasattr(self, "TCN_avg"):
            torch.nn.init.constant_(self.TCN_avg.avgbn.weight, gamma_value)
        if hasattr(self, "TCN_1x1_kxk"):
            torch.nn.init.constant_(self.TCN_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "TCN_origin"):
            torch.nn.init.constant_(self.TCN_origin.bn.weight, 1.0)


class REPTcn_Block4(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 deploy=False, nonlinear=None, single_init=False):
        super(REPTcn_Block4, self).__init__()
        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        if deploy:
            self.TCN_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                         stride=(stride, 1), padding=(padding, 0), dilation=dilation, groups=groups, bias=True)

        else:
            self.TCN_1x1_9x9 = nn.Sequential()
            self.TCN_1x1_9x9.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                           kernel_size=1, stride=1, padding=0, groups=groups,
                                                           bias=False))
            self.TCN_1x1_9x9.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=out_channels, affine=True))
            self.TCN_1x1_9x9.add_module('conv2', nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                                           kernel_size=(kernel_size, 1), stride=(stride, 1), padding=0,
                                                           groups=groups, bias=False))
            self.TCN_1x1_9x9.add_module('bn2', nn.BatchNorm2d(out_channels))

            self.TCN_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                      stride=(stride, 1), padding=(padding, 0), dilation=dilation, groups=groups)

            self.TCN_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=(stride, 1),
                                   padding=0, groups=groups)

            self.TCN_avg = nn.Sequential()
            self.TCN_avg.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                              stride=1, padding=0, groups=groups, bias=False))
            self.TCN_avg.add_module('bn', BNAndPadLayer(pad_pixels=1, num_features=out_channels))
            self.TCN_avg.add_module('avg', nn.AvgPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=0))
            self.TCN_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

            self.TCN_1x1_kxk = nn.Sequential()
            self.TCN_1x1_kxk.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                            kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.TCN_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=2, num_features=out_channels, affine=True))
            self.TCN_1x1_kxk.add_module('conv2', nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                                            kernel_size=(5, 1), stride=(stride, 1), padding=0, groups=groups, bias=False))
            self.TCN_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

            #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
            if single_init:
                #   Initialize the bn.weight of TCN_origin as 1 and others as 0. This is not the default setting.
                self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transII_fusebn(self.TCN_origin.conv.weight, self.TCN_origin.bn)

        k_1x1_9x9_first = self.TCN_1x1_9x9.conv1.weight
        k_1x1_9x9_first, b_1x1_9x9_first = transII_fusebn(k_1x1_9x9_first, self.TCN_1x1_9x9.bn1)
        k_1x1_9x9_second, b_1x1_9x9_second = transII_fusebn(self.TCN_1x1_9x9.conv2.weight, self.TCN_1x1_9x9.bn2)
        k_1x1_9x9_merged, b_1x1_9x9_merged = transIII_1x1_kxk(k_1x1_9x9_first, b_1x1_9x9_first, k_1x1_9x9_second, b_1x1_9x9_second, groups=self.groups)
        k_1x1_9x9_merged = transV_multiscale_T(k_1x1_9x9_merged, self.kernel_size)

        if hasattr(self, 'TCN_1x1'):
            k_1x1, b_1x1 = transII_fusebn(self.TCN_1x1.conv.weight, self.TCN_1x1.bn)
            k_1x1 = transV_multiscale_T(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        k_1x1_kxk_first = self.TCN_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transII_fusebn(k_1x1_kxk_first, self.TCN_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transII_fusebn(self.TCN_1x1_kxk.conv2.weight, self.TCN_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second, b_1x1_kxk_second, groups=self.groups)
        k_1x1_kxk_merged = transV_multiscale_T(k_1x1_kxk_merged, self.kernel_size)

        k_avg = transI_avg_T(self.out_channels, 3, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transII_fusebn(k_avg.to(self.TCN_avg.avgbn.weight.device), self.TCN_avg.avgbn)
        if hasattr(self.TCN_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transII_fusebn(self.TCN_avg.conv.weight, self.TCN_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second, b_1x1_avg_second, groups=self.groups)
            k_1x1_avg_merged = transV_multiscale_T(k_1x1_avg_merged, self.kernel_size)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second
            k_1x1_avg_merged = transV_multiscale_T(k_1x1_avg_merged, self.kernel_size)

        return transIV_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged, k_1x1_9x9_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged, b_1x1_9x9_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'TCN_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.TCN_reparam = nn.Conv2d(in_channels=self.TCN_origin.conv.in_channels, out_channels=self.TCN_origin.conv.out_channels,
                                     kernel_size=self.TCN_origin.conv.kernel_size, stride=self.TCN_origin.conv.stride,
                                     padding=self.TCN_origin.conv.padding, dilation=self.TCN_origin.conv.dilation, groups=self.TCN_origin.conv.groups, bias=True)
        self.TCN_reparam.weight.data = kernel
        self.TCN_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('TCN_origin')
        self.__delattr__('TCN_avg')
        if hasattr(self, 'TCN_1x1'):
            self.__delattr__('TCN_1x1')
        self.__delattr__('TCN_1x1_kxk')
        self.__delattr__('TCN_1x1_9x9')

    def forward(self, inputs):

        if hasattr(self, 'TCN_reparam'):
            return self.nonlinear(self.TCN_reparam(inputs))

        out = self.TCN_origin(inputs)
        if hasattr(self, 'TCN_1x1'):
            out += self.TCN_1x1(inputs)
        out += self.TCN_avg(inputs)
        out += self.TCN_1x1_kxk(inputs)
        out += self.TCN_1x1_9x9(inputs)
        return self.nonlinear(out)

    def init_gamma(self, gamma_value):
        if hasattr(self, "TCN_origin"):
            torch.nn.init.constant_(self.TCN_origin.bn.weight, gamma_value)
        if hasattr(self, "TCN_1x1"):
            torch.nn.init.constant_(self.TCN_1x1.bn.weight, gamma_value)
        if hasattr(self, "TCN_avg"):
            torch.nn.init.constant_(self.TCN_avg.avgbn.weight, gamma_value)
        if hasattr(self, "TCN_1x1_kxk"):
            torch.nn.init.constant_(self.TCN_1x1_kxk.bn2.weight, gamma_value)
        if hasattr(self, "TCN_1x1_9x9"):
            torch.nn.init.constant_(self.TCN_1x1_9x9.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "TCN_origin"):
            torch.nn.init.constant_(self.TCN_origin.bn.weight, 1.0)


class REPTcn_Block9(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 deploy=False, nonlinear=None, single_init=False):
        super(REPTcn_Block9, self).__init__()
        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        if deploy:
            self.TCN_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                         stride=(stride, 1), padding=(padding, 0), dilation=dilation, groups=groups, bias=True)

        else:
            self.TCN_1x1_7x7 = nn.Sequential()
            self.TCN_1x1_7x7.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                           kernel_size=1, stride=1, padding=0, groups=groups,
                                                           bias=False))
            self.TCN_1x1_7x7.add_module('bn1', BNAndPadLayer(pad_pixels=3, num_features=out_channels, affine=True))
            self.TCN_1x1_7x7.add_module('conv2', nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                                           kernel_size=(7, 1), stride=(stride, 1), padding=0,
                                                           groups=groups, bias=False))
            self.TCN_1x1_7x7.add_module('bn2', nn.BatchNorm2d(out_channels))

            self.TCN_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                      stride=(stride, 1), padding=(padding, 0), dilation=dilation, groups=groups)

            self.TCN_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=(stride, 1),
                                   padding=0, groups=groups)

            self.TCN_avg = nn.Sequential()
            self.TCN_avg.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                              stride=1, padding=0, groups=groups, bias=False))
            self.TCN_avg.add_module('bn', BNAndPadLayer(pad_pixels=1, num_features=out_channels))
            self.TCN_avg.add_module('avg', nn.AvgPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=0))
            self.TCN_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))

            self.TCN_1x1_kxk = nn.Sequential()
            self.TCN_1x1_kxk.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                            kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.TCN_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=2, num_features=out_channels, affine=True))
            self.TCN_1x1_kxk.add_module('conv2', nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                                            kernel_size=(5, 1), stride=(stride, 1), padding=0, groups=groups, bias=False))
            self.TCN_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

            #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
            if single_init:
                #   Initialize the bn.weight of TCN_origin as 1 and others as 0. This is not the default setting.
                self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transII_fusebn(self.TCN_origin.conv.weight, self.TCN_origin.bn)

        k_1x1_9x9_first = self.TCN_1x1_7x7.conv1.weight
        k_1x1_9x9_first, b_1x1_9x9_first = transII_fusebn(k_1x1_9x9_first, self.TCN_1x1_7x7.bn1)
        k_1x1_9x9_second, b_1x1_9x9_second = transII_fusebn(self.TCN_1x1_7x7.conv2.weight, self.TCN_1x1_7x7.bn2)
        k_1x1_9x9_merged, b_1x1_9x9_merged = transIII_1x1_kxk(k_1x1_9x9_first, b_1x1_9x9_first, k_1x1_9x9_second, b_1x1_9x9_second, groups=self.groups)
        k_1x1_9x9_merged = transV_multiscale_T(k_1x1_9x9_merged, self.kernel_size)

        if hasattr(self, 'TCN_1x1'):
            k_1x1, b_1x1 = transII_fusebn(self.TCN_1x1.conv.weight, self.TCN_1x1.bn)
            k_1x1 = transV_multiscale_T(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        k_1x1_kxk_first = self.TCN_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transII_fusebn(k_1x1_kxk_first, self.TCN_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transII_fusebn(self.TCN_1x1_kxk.conv2.weight, self.TCN_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second, b_1x1_kxk_second, groups=self.groups)
        k_1x1_kxk_merged = transV_multiscale_T(k_1x1_kxk_merged, self.kernel_size)

        k_avg = transI_avg_T(self.out_channels, 3, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transII_fusebn(k_avg.to(self.TCN_avg.avgbn.weight.device), self.TCN_avg.avgbn)
        if hasattr(self.TCN_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transII_fusebn(self.TCN_avg.conv.weight, self.TCN_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second, b_1x1_avg_second, groups=self.groups)
            k_1x1_avg_merged = transV_multiscale_T(k_1x1_avg_merged, self.kernel_size)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second
            k_1x1_avg_merged = transV_multiscale_T(k_1x1_avg_merged, self.kernel_size)

        return transIV_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged, k_1x1_9x9_merged),
                                 (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged, b_1x1_9x9_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'TCN_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.TCN_reparam = nn.Conv2d(in_channels=self.TCN_origin.conv.in_channels, out_channels=self.TCN_origin.conv.out_channels,
                                     kernel_size=self.TCN_origin.conv.kernel_size, stride=self.TCN_origin.conv.stride,
                                     padding=self.TCN_origin.conv.padding, dilation=self.TCN_origin.conv.dilation, groups=self.TCN_origin.conv.groups, bias=True)
        self.TCN_reparam.weight.data = kernel
        self.TCN_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('TCN_origin')
        self.__delattr__('TCN_avg')
        if hasattr(self, 'TCN_1x1'):
            self.__delattr__('TCN_1x1')
        self.__delattr__('TCN_1x1_kxk')
        self.__delattr__('TCN_1x1_7x7')

    def forward(self, inputs):

        if hasattr(self, 'TCN_reparam'):
            return self.nonlinear(self.TCN_reparam(inputs))

        out = self.TCN_origin(inputs)
        if hasattr(self, 'TCN_1x1'):
            out += self.TCN_1x1(inputs)
        out += self.TCN_avg(inputs)
        out += self.TCN_1x1_kxk(inputs)
        out += self.TCN_1x1_7x7(inputs)
        return self.nonlinear(out)

    def init_gamma(self, gamma_value):
        if hasattr(self, "TCN_origin"):
            torch.nn.init.constant_(self.TCN_origin.bn.weight, gamma_value)
        if hasattr(self, "TCN_1x1"):
            torch.nn.init.constant_(self.TCN_1x1.bn.weight, gamma_value)
        if hasattr(self, "TCN_avg"):
            torch.nn.init.constant_(self.TCN_avg.avgbn.weight, gamma_value)
        if hasattr(self, "TCN_1x1_kxk"):
            torch.nn.init.constant_(self.TCN_1x1_kxk.bn2.weight, gamma_value)
        if hasattr(self, "TCN_1x1_7x7"):
            torch.nn.init.constant_(self.TCN_1x1_7x7.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "TCN_origin"):
            torch.nn.init.constant_(self.TCN_origin.bn.weight, 1.0)