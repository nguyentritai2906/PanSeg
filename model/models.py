import torch
from torch import nn

from model.backbone.unet import UNet
from model.backbone.vargnet import VarGNet_FPS
from model.fpsnet import (FPN, AttentionHead, BBoxHead, MergeFeatureMaps,
                          PanopticHead)
from model.loss.focal import FocalLoss


class FPSNet(nn.Module):
    def __init__(self, num_classes):
        super(FPSNet, self).__init__()
        self.num_classes = num_classes
        self.backbone = VarGNet_FPS(3)
        self.fpn = FPN(256, 5e-4)
        self.merge_fms = MergeFeatureMaps()
        self.bbox_head = BBoxHead(num_classes)
        self.attention_head = AttentionHead()
        self.panoptic_head = PanopticHead()
        self.features = UNet(3, 80)
        self.focal_loss = FocalLoss(19)
        self.softmax_crossentropy_loss = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        backbone = self.backbone(x)
        feature_map = self.features(x)
        fpn = self.fpn(backbone)
        #  for f in fpn:
        #  print(f.shape)
        loc_preds, cls_preds = self.bbox_head(fpn, x.size(0))
        attention_masks, attention_labels = self.attention_head(
            loc_preds, cls_preds, fpn[0].shape, x.size(0), x[0].shape)
        panoptic_out = self.panoptic_head(  # [batch_size, H, W]
            torch.cat([attention_masks, feature_map], dim=1))
        results = dict(outputs=panoptic_out,
                       output_labels=attention_labels,
                       bboxes=loc_preds,
                       labels=cls_preds)
        #  print(loc_preds.shape)
        #  print(cls_preds.shape)
        if targets is None:
            return results
        else:
            return self.loss(results, targets)

    def loss(self, results, targets=None):
        # self, loc_preds, loc_targets, cls_preds, cls_targets
        focal = self.focal_loss(results['bboxes'], targets['boxes'],
                                results['labels'], targets['classes'])
        sce = self.softmax_crossentropy_loss(results['outputs'],
                                             targets['panoptic'])
        return 0.5 * focal + sce


if __name__ == "__main__":
    from torchinfo import summary
    net = FPSNet(19)
    input_data = torch.rand((2, 3, 512, 1024))
    print(summary(net, input_data=input_data, depth=2))
