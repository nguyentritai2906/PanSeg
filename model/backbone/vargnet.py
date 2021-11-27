import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from utils.utils import get_act


class SEBlock(nn.Module):
    """Docstring for SEBlock. """
    def __init__(self, in_channels, out_channels, activation_function):
        super(SEBlock, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 'same')
        self.activation_function = get_act(activation_function)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pool1 = self.pool1(x)
        conv1 = self.conv1(pool1)
        activation1 = self.activation_function(conv1)
        conv2 = self.conv2(activation1)
        activation2 = self.activation_function(conv2)
        output = torch.mul(x, activation2)
        return output


class SeparableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 stride,
                 factor=1,
                 bias=False,
                 dilation_rate=1,
                 activation_function_out=True):
        super(SeparableConv2d, self).__init__()
        self.padding = padding
        self.activation_function_out = activation_function_out
        self.depthwise = nn.Conv2d(in_channels=in_channels,
                                   out_channels=int(in_channels * factor),
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   bias=bias,
                                   groups=int(in_channels / 8),
                                   dilation=dilation_rate)
        self.batchnorm_depthwise_out = nn.BatchNorm2d(num_features=int(
            in_channels * factor),
                                                      eps=2e-5,
                                                      momentum=0.9)
        self.activation_function = get_act('lrelu')
        self.pointwise = nn.Conv2d(in_channels=int(in_channels * factor),
                                   out_channels=out_channels,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   bias=bias,
                                   groups=1)
        self.batchnorm_pointwise_out = nn.BatchNorm2d(
            num_features=out_channels, eps=2e-5, momentum=0.9)

    def forward(self, x):
        if self.padding is not None:
            x = F.pad(x, self.padding)
        output = self.depthwise(x)
        output = self.batchnorm_depthwise_out(output)
        output = self.activation_function(output)
        output = self.pointwise(output)
        output = self.batchnorm_pointwise_out(output)
        if self.activation_function_out:
            output = self.activation_function(output)
        return output


class VarGNetBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels_1,
                 out_channels_2,
                 factor=2,
                 multiplier=1,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 dilation_rate=1,
                 with_dilation=False,
                 dimention_match=True):
        super(VarGNetBlock, self).__init__()

        in_channels = int(in_channels * multiplier)
        out_channels_1 = int(out_channels_1 * multiplier)
        out_channels_2 = int(out_channels_2 * multiplier)

        w, h = (((kernel_size[0] - 1) * dilation_rate + 1) // 2,
                ((kernel_size[1] - 1) * dilation_rate + 1) // 2)
        padding = (w, w, h, h, 0, 0, 0, 0)

        if with_dilation:
            assert stride == (1, 1)

        self.use_seblock = True
        self.dimention_match = dimention_match

        self.shortcut = SeparableConv2d(in_channels=in_channels,
                                        out_channels=out_channels_2,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        stride=stride,
                                        factor=factor,
                                        bias=False,
                                        dilation_rate=dilation_rate,
                                        activation_function_out=False)

        self.sep1 = SeparableConv2d(in_channels=in_channels,
                                    out_channels=out_channels_1,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    factor=factor,
                                    bias=False,
                                    dilation_rate=dilation_rate)

        self.sep2 = SeparableConv2d(in_channels=out_channels_1,
                                    out_channels=out_channels_2,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=(1, 1),
                                    factor=factor,
                                    bias=False,
                                    dilation_rate=dilation_rate)
        self.seblock = SEBlock(in_channels=out_channels_2,
                               out_channels=out_channels_2,
                               activation_function='lrelu')
        self.activation_function = get_act('lrelu')

    def forward(self, x):
        if self.dimention_match:
            shortcut = x
        else:
            shortcut = self.shortcut(x)
        sep1 = self.sep1(x)
        sep2 = self.sep2(sep1)
        if self.use_seblock:
            sep2 = self.seblock(sep2)
        output = torch.add(sep2, shortcut)
        output = self.activation_function(output)
        return output


class VarGNetBranchMergeBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels_1,
                 out_channels_2,
                 factor=2,
                 multiplier=1,
                 kernel_size=(3, 3),
                 stride=(2, 2),
                 dilation_rate=1,
                 with_dilation=False,
                 dimention_match=False):
        super(VarGNetBranchMergeBlock, self).__init__()

        in_channels = int(in_channels * multiplier)
        out_channels_1 = int(out_channels_1 * multiplier)
        out_channels_2 = int(out_channels_2 * multiplier)

        w, h = (((kernel_size[0] - 1) * dilation_rate + 1) // 2,
                ((kernel_size[1] - 1) * dilation_rate + 1) // 2)
        padding = (w, w, h, h, 0, 0, 0, 0)

        if with_dilation:
            stride = (1, 1)

        self.dimention_match = dimention_match

        self.shortcut = SeparableConv2d(in_channels=in_channels,
                                        out_channels=out_channels_2,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        stride=stride,
                                        factor=factor,
                                        bias=False,
                                        dilation_rate=dilation_rate,
                                        activation_function_out=False)

        self.sep1 = SeparableConv2d(in_channels=in_channels,
                                    out_channels=out_channels_1,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    factor=factor,
                                    bias=False,
                                    dilation_rate=dilation_rate,
                                    activation_function_out=False)

        self.sep2 = SeparableConv2d(in_channels=out_channels_1,
                                    out_channels=out_channels_2,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=(1, 1),
                                    factor=factor,
                                    bias=False,
                                    dilation_rate=dilation_rate,
                                    activation_function_out=False)
        self.seblock = SEBlock(in_channels=out_channels_2,
                               out_channels=out_channels_2,
                               activation_function='lrelu')
        self.activation_function = get_act('lrelu')
        pass

    def forward(self, x):
        if self.dimention_match:
            shortcut = x
        else:
            shortcut = self.shortcut(x)
        sep1_branch1 = self.sep1(x)
        sep1_branch2 = self.sep1(x)
        sep1 = torch.add(sep1_branch1, sep1_branch2)
        sep1 = self.activation_function(sep1)
        sep2 = self.sep2(sep1)
        output = torch.add(sep2, shortcut)
        output = self.activation_function(output)
        return output


class AddVarGNetConvBlock(nn.Module):
    def __init__(self,
                 units,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=(2, 2),
                 multiplier=1,
                 factor=2,
                 dilation_rate=1,
                 with_dilation=False):

        super(AddVarGNetConvBlock, self).__init__()

        self.units = units

        self.branch_merge = VarGNetBranchMergeBlock(
            in_channels=in_channels,
            out_channels_1=out_channels,
            out_channels_2=out_channels,
            factor=factor,
            multiplier=multiplier,
            kernel_size=kernel_size,
            stride=stride,
            dilation_rate=dilation_rate,
            with_dilation=with_dilation,
            dimention_match=False)

        self.vargnet_block = VarGNetBlock(in_channels=out_channels,
                                          out_channels_1=out_channels,
                                          out_channels_2=out_channels,
                                          factor=factor,
                                          multiplier=multiplier,
                                          kernel_size=kernel_size,
                                          stride=(1, 1),
                                          dilation_rate=dilation_rate,
                                          with_dilation=with_dilation,
                                          dimention_match=True)

    def forward(self, x):
        output = self.branch_merge(x)
        for _ in range(self.units - 1):
            output = self.vargnet_block(output)
        return output


class AddHeadBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 multiplier=1,
                 kernel_size=(3, 3),
                 stride=(2, 2),
                 padding=(1, 1, 1, 1, 0, 0, 0, 0),
                 head_pooling=False):

        super(AddHeadBlock, self).__init__()
        self.fix_gamma = False
        self.use_global_stats = False
        self.workspace = 512
        self.act_type = 'lrelu'
        self.padding = padding
        self.head_pooling = head_pooling

        channels = int(out_channels * multiplier)
        batchnorm_momentum = 0.9
        epsilon = 2e-5

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              bias=False,
                              groups=1)

        self.batch_norm = nn.BatchNorm2d(channels,
                                         eps=epsilon,
                                         momentum=batchnorm_momentum,
                                         affine=False)

        self.activation_function = get_act('lrelu')

        self.maxpool2d = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.vargnet_block = VarGNetBlock(
            in_channels=channels,  # 50
            out_channels_1=out_channels,
            out_channels_2=out_channels,
            factor=1,
            multiplier=multiplier,
            kernel_size=kernel_size,
            stride=(2, 2),
            dilation_rate=1,
            with_dilation=False,
            dimention_match=False)

    def forward(self, x):
        x = F.pad(x, self.padding)
        conv = self.conv(x)
        bn = self.batch_norm(conv)
        act = self.activation_function(bn)
        if self.head_pooling:
            act = F.pad(act, self.padding)
            head_data = self.maxpool2d(act)
        else:
            head_data = self.vargnet_block(act)
        return head_data


class AddEmbBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_size, bias=False):

        super(AddEmbBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fix_gamma = False
        self.use_global_stats = False
        self.workspace = 512
        self.act_type = 'lrelu'
        self.group_base = 8

        bn_mom = 0.9
        bn_eps = 2e-5

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               bias=bias)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels / self.group_base,
                               kernel_size=(7, 7),
                               stride=(1, 1),
                               bias=bias)

        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels // 2,
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               bias=bias)

        self.bn1 = nn.BatchNorm2d(num_features=out_channels,
                                  affine=False,
                                  eps=bn_eps,
                                  momentum=bn_mom)

        self.bn2 = nn.BatchNorm2d(num_features=out_channels / self.group_base,
                                  affine=False,
                                  eps=bn_eps,
                                  momentum=bn_mom)

        self.bn3 = nn.BatchNorm2d(num_features=out_channels // 2,
                                  affine=False,
                                  eps=bn_eps,
                                  momentum=bn_mom)

        self.bn4 = nn.BatchNorm2d(num_features=emb_size,
                                  affine=False,
                                  eps=bn_eps,
                                  momentum=bn_mom)

        self.fc = nn.Linear(in_features=out_channels // 2,
                            out_features=emb_size)

        self.activation_function = get_act('lrelu')

    def forward(self, x):

        if self.in_channels != self.out_channels:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.activation_function(x)

        convx_depthwise = self.conv2(x)
        convx_depthwise = self.bn2(convx_depthwise)

        convx_pointwise = self.conv3(convx_depthwise)
        convx_pointwise = self.bn3(convx_pointwise)
        convx_pointwise = self.activation_function(convx_pointwise)

        emb_feature = self.fc(convx_pointwise)
        emb_feature = self.bn4(emb_feature)

        return emb_feature


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            SeparableConv2d(in_channels,
                            mid_channels,
                            kernel_size=3,
                            padding=None,
                            stride=1),
            SeparableConv2d(in_channels,
                            mid_channels,
                            kernel_size=3,
                            padding=None,
                            stride=1))
        self.residual = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        doubleconv = self.double_conv(x)
        x = self.residual(x)
        output = torch.add(x, doubleconv)
        return output


class VarGFacenet(nn.Module):
    def __init__(self, in_channels, multiplier, emb_size, factor,
                 head_pooling):

        super(VarGFacenet, self).__init__()
        self.in_channels = in_channels

        num_stage = 3
        stage_list = [2, 3, 4]
        units = [3, 7, 4]
        filter_list = [32, 64, 128, 256]
        out_channels = 1024
        dilate_list = [1, 1, 1]
        with_dilate_list = [False, False, False]

        self.add_head_block = AddHeadBlock(in_channels=in_channels,
                                           out_channels=filter_list[0],
                                           multiplier=multiplier,
                                           head_pooling=head_pooling,
                                           kernel_size=(3, 3),
                                           stride=(1, 1),
                                           padding=[1, 1, 1, 1, 0, 0, 0, 0])

        add_vargnet_conv_block = []
        for i in range(num_stage):
            add_vargnet_conv_block.append(
                AddVarGNetConvBlock(units=units[i],
                                    in_channels=filter_list[i],
                                    out_channels=filter_list[i + 1],
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    multiplier=multiplier,
                                    factor=factor,
                                    dilation_rate=dilate_list[i],
                                    with_dilation=with_dilate_list[i]))
        self.add_vargnet_conv_block = nn.Sequential(*add_vargnet_conv_block)

        self.emb_feature = AddEmbBlock(in_channels=filter_list[3],
                                       out_channels=out_channels,
                                       emb_size=emb_size,
                                       bias=False)

    def forward(self, x):
        body = self.add_head_block(x)
        body = self.add_vargnet_conv_block(body)
        emb_feature = self.emb_feature(body)
        emb_feature = torch.squeeze(emb_feature)
        return emb_feature


class VarGNet_v2(nn.Module):
    def __init__(self,
                 in_channels,
                 multiplier=1,
                 factor=2,
                 head_pooling=False):

        super(VarGNet_v2, self).__init__()

        num_stage = 3
        units = [3, 7, 4]
        filter_list = [32, 64, 128, 256, 1024]
        out_channels = 1024
        dilate_list = [1, 1, 1]
        with_dilate_list = [False, False, False]

        self.add_head_block = AddHeadBlock(in_channels=in_channels,
                                           out_channels=filter_list[0],
                                           multiplier=multiplier,
                                           head_pooling=head_pooling,
                                           kernel_size=(3, 3),
                                           stride=(2, 2),
                                           padding=[1, 1, 1, 1, 0, 0, 0, 0])

        add_vargnet_conv_block = []
        for i in range(num_stage):
            add_vargnet_conv_block.append(
                AddVarGNetConvBlock(units=units[i],
                                    in_channels=filter_list[i],
                                    out_channels=filter_list[i + 1],
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    multiplier=multiplier,
                                    factor=factor,
                                    dilation_rate=dilate_list[i],
                                    with_dilation=with_dilate_list[i]))
        self.add_vargnet_conv_block = nn.Sequential(*add_vargnet_conv_block)

        self.conv = nn.Conv2d(in_channels=filter_list[3],
                              out_channels=int(out_channels * multiplier),
                              kernel_size=(1, 1),
                              stride=(1, 1),
                              bias=False,
                              groups=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        body = self.add_head_block(x)
        body = self.add_vargnet_conv_block(body)
        conv5 = self.conv(body)
        global_pool = self.global_pool(conv5)
        return global_pool


class VarGNet_FPS(nn.Module):
    def __init__(self,
                 in_channels,
                 multiplier=1,
                 factor=2,
                 head_pooling=False):

        super(VarGNet_FPS, self).__init__()

        units = [3, 7, 4, 1]
        filter_list = [16, 32, 64, 128, 256, 1024]
        dilate_list = [1, 1, 1, 1]
        with_dilate_list = [False, False, False, False]

        self.add_head_block = AddHeadBlock(in_channels=in_channels,
                                           out_channels=filter_list[0],
                                           multiplier=multiplier,
                                           head_pooling=head_pooling,
                                           kernel_size=(3, 3),
                                           stride=(2, 2),
                                           padding=[1, 1, 1, 1, 0, 0, 0, 0])

        self.down1 = AddVarGNetConvBlock(units=units[0],
                                         in_channels=filter_list[0],
                                         out_channels=filter_list[1],
                                         kernel_size=(3, 3),
                                         stride=(2, 2),
                                         multiplier=multiplier,
                                         factor=factor,
                                         dilation_rate=dilate_list[0],
                                         with_dilation=with_dilate_list[0])

        self.down2 = AddVarGNetConvBlock(units=units[1],
                                         in_channels=filter_list[1],
                                         out_channels=filter_list[2],
                                         kernel_size=(3, 3),
                                         stride=(2, 2),
                                         multiplier=multiplier,
                                         factor=factor,
                                         dilation_rate=dilate_list[1],
                                         with_dilation=with_dilate_list[1])

        self.down3 = AddVarGNetConvBlock(units=units[2],
                                         in_channels=filter_list[2],
                                         out_channels=filter_list[3],
                                         kernel_size=(3, 3),
                                         stride=(2, 2),
                                         multiplier=multiplier,
                                         factor=factor,
                                         dilation_rate=dilate_list[2],
                                         with_dilation=with_dilate_list[2])

        # self.down4 = AddVarGNetConvBlock(units=units[3],
        #                                  in_channels=filter_list[3],
        #                                  out_channels=filter_list[4],
        #                                  kernel_size=(3, 3),
        #                                  stride=(2, 2),
        #                                  multiplier=multiplier,
        #                                  factor=factor,
        #                                  dilation_rate=dilate_list[3],
        #                                  with_dilation=with_dilate_list[3])

    def forward(self, x):
        body = self.add_head_block(x)
        down1 = self.down1(body)
        down2 = self.down2(down1)  # [1, 256, 64, 64]
        down3 = self.down3(down2)  # [1, 256, 32, 32]
        #  down4 = self.down4(down3)  # [1, 256, 16, 16]
        return down1, down2, down3


# from torchinfo import summary
# model = VarGNet_FPS(3)
# print(summary(model, (1, 3, 640, 640)))
# input = torch.rand((1, 3, 640, 640))
# output = model(input)
# print(output[0].shape, output[1].shape, output[2].shape)
