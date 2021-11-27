import math
from functools import reduce

import torch
from data.encoder import DataEncoder
from torch import nn
from torch.nn import functional as F


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels, (3, 3),
                              padding='same')
        self.up = nn.Upsample(scale_factor=2,
                              mode='bilinear',
                              align_corners=True)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.up(x)
        return x


class MergeFeatureMaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.up3 = nn.Sequential(Up(256, 256), Up(256, 128))
        self.up2 = Up(256, 128)
        self.up1 = nn.Conv2d(256, 128, (3, 3), padding='same')

    def forward(self, x1, x2, x3):
        out3 = self.up3(x3)
        out2 = self.up2(x2)
        out1 = F.relu(self.up1(x1))
        # https://stackoverflow.com/questions/61774526/add-multiple-tensors-inplace-in-pytorch#61774748
        feature_map = reduce(torch.Tensor.add_, [out1, out2, out3],
                             torch.zeros_like(out1))
        return feature_map


class BBoxHead(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(1):  # 4
            layers.append(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(
            nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x, batch_size):
        out1, out2, out3 = x
        loc_preds = []  # len([torch.Size([N, H*W*9, 4]), ...]) = 3
        cls_preds = []  # len([torch.Size([N, H*W*9, num_classes]), ...]) = 3
        for fm in [out1, out2, out3]:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(
                batch_size, -1, 4)
            # [N, 9*num_classes, H, W] -> [N, H, W, 9*num_classes] -> [N, H*W*9, num_classes]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(
                batch_size, -1, self.num_classes)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)


class AttentionHead(nn.Module):
    num_attention_mask = 50
    scale_factor = 50

    def __init__(self):
        super().__init__()
        self.encoder = DataEncoder()

    def _normalize(self, x: torch.Tensor):
        channel_min = x.min(dim=1, keepdim=True)[0].min(
            dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        channel_max = x.max(dim=1, keepdim=True)[0].max(
            dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        return (x - channel_min) / (channel_max - channel_min)

    def forward(self, loc_preds, cls_preds, fms_size, batch_size, img_size):
        decoded_boxes, decoded_labels = [], []
        for i in range(batch_size):
            boxes, labels = self.encoder.decode(loc_preds[i], cls_preds[i],
                                                tuple(img_size[1:]))
            decoded_boxes.append(boxes)  # [num_box, 4]
            decoded_labels.append(labels)  # [num_box, num_classes]

        batch_masks = []
        batch_labels = []
        for batch in range(batch_size):
            boxes = decoded_boxes[batch]
            labels = decoded_labels[batch]

            masks = []
            zero_masks = 0

            for box in boxes:
                x, y, w, h = box
                xc, yc = x + w // 2, y + h // 2
                mask = MaskGenerator(
                    torch.tensor([[xc], [yc]], dtype=torch.float32),
                    torch.diag(
                        torch.tensor([.25 * w, .25 * h], dtype=torch.float32)),
                    fms_size)()
                masks.append(mask)

            if len(masks) > self.num_attention_mask:
                batch_masks.append(torch.stack(
                    masks[:self.num_attention_mask]))
                batch_labels.append(labels[:self.num_attention_mask])
            elif len(masks) < self.num_attention_mask:
                zero_masks = self.num_attention_mask - len(masks)
                batch_masks.append(
                    torch.cat([
                        torch.stack(masks),
                        torch.zeros((zero_masks, fms_size[0], fms_size[1]))
                    ]))
                batch_labels.append(
                    torch.cat([labels, torch.zeros((zero_masks, 1))]))

        attention_masks = self._normalize(
            torch.stack(batch_masks)) * self.scale_factor
        attention_labels = torch.stack(batch_labels)

        shuffler = nn.ChannelShuffle(self.num_attention_mask // 2)
        attention_masks = shuffler(attention_masks)
        attention_labels = torch.squeeze(
            shuffler(torch.unsqueeze(attention_labels, -1)))

        return attention_masks, attention_labels


class PanopticHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.preconv = nn.Sequential(nn.Conv2d(306, 128, 3, padding='same'),
                                     nn.ReLU(), nn.BatchNorm2d(128))
        self.conv1 = nn.Sequential(nn.Conv2d(128, 128, 3, padding='same'),
                                   nn.ReLU(), nn.BatchNorm2d(128))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding='same'),
                                   nn.ReLU(), nn.BatchNorm2d(128))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 3, padding='same'),
                                   nn.ReLU(), nn.BatchNorm2d(128))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, 3, padding='same'),
                                   nn.ReLU(), nn.BatchNorm2d(128))
        self.conv5 = nn.Conv2d(128, 50 + 11 + 2, 1,
                               padding='same')  # N_att = 50, N_stuff = 11
        self.up = nn.UpsamplingBilinear2d(scale_factor=8)

    def forward(self, x):
        preconv = self.preconv(x)
        conv1 = self.conv1(preconv)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up = self.up(conv5)
        out = torch.argmax(up, dim=1)
        return out


class MaskGenerator(object):
    def __init__(self, mean: torch.Tensor, covar: torch.Tensor, size: tuple):
        self.u_ = mean  # Tensor of shape [2, 1]
        self.sig_ = covar  # Tensor of shape [2, 2]
        self.size_ = size[2:]

    def _prob(self, x):
        #  x = torch.unsqueeze(x, dim=-1) if x.dim() == 2 else x
        factor1 = (2 * math.pi)**(-self.u_.shape[0] / 2) * torch.linalg.det(
            self.sig_)**(-1 / 2)
        factor2 = torch.exp(
            (-1 / 2) * torch.einsum('ijk, jl, ilk -> ik', x - self.u_,
                                    torch.linalg.inv(self.sig_), x - self.u_))
        return factor1 * factor2

    def __call__(self):
        x = torch.round(torch.linspace(0, self.size_[0], self.size_[0]))
        y = torch.round(torch.linspace(0, self.size_[1], self.size_[1]))
        X, Y = torch.meshgrid(x, y)
        x_new = torch.stack([X, Y]).permute((1, 2, 0)).reshape(-1, 2, 1)
        prob = self._prob(x_new)
        out = prob.reshape(self.size_[0], self.size_[1])
        return out


class ConvUnit(nn.Module):
    """Conv + BN + Act"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 k,
                 s,
                 wd,
                 act=None,
                 **kwargs):
        super(ConvUnit, self).__init__(**kwargs)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=k,
                              stride=s,
                              padding='same',
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        if act is None:
            self.act_fn = nn.Identity()
        elif act == 'relu':
            self.act_fn = nn.ReLU()
        elif act == 'lrelu':
            self.act_fn = nn.LeakyReLU(0.1)
        else:
            raise NotImplementedError(
                'Activation function type {} is not recognized.'.format(act))

    def forward(self, x):
        return self.act_fn(self.bn(self.conv(x)))


class FPN(nn.Module):
    """Feature Pyramid Network"""
    def __init__(self, out_channels, wd, **kwargs):
        super(FPN, self).__init__(**kwargs)
        act = 'relu'
        if (out_channels <= 64):
            act = 'lrelu'

        self.output1 = ConvUnit(in_channels=32,
                                out_channels=out_channels,
                                k=1,
                                s=1,
                                wd=wd,
                                act=act)
        self.output2 = ConvUnit(in_channels=64,
                                out_channels=out_channels,
                                k=1,
                                s=1,
                                wd=wd,
                                act=act)
        self.output3 = ConvUnit(in_channels=128,
                                out_channels=out_channels,
                                k=1,
                                s=1,
                                wd=wd,
                                act=act)
        self.merge1 = ConvUnit(in_channels=256,
                               out_channels=out_channels,
                               k=3,
                               s=1,
                               wd=wd,
                               act=act)
        self.merge2 = ConvUnit(in_channels=256,
                               out_channels=out_channels,
                               k=3,
                               s=1,
                               wd=wd,
                               act=act)

    def forward(self, x):
        output1 = self.output1(x[0])  # [80, 80, out_ch]
        output2 = self.output2(x[1])  # [40, 40, out_ch]
        output3 = self.output3(x[2])  # [20, 20, out_ch]

        up_h, up_w = list(output2.shape[-2:])
        up3 = F.interpolate(output3, [up_h, up_w], mode='nearest')
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up_h, up_w = list(output1.shape[-2:])
        up2 = F.interpolate(output2, [up_h, up_w], mode='nearest')
        output1 = output1 + up2
        output1 = self.merge1(output1)

        return output1, output2, output3
