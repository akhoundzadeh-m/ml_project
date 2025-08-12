import torch
import torch.nn as nn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.image_list import ImageList

class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, num_groups=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.GroupNorm(num_groups, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False),
            nn.GroupNorm(num_groups, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
        )
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.GroupNorm(num_groups, out_channels)
            ) if stride != 1 or in_channels != out_channels 
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        return self.relu(self.net(x) + identity)


class SepConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=16):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.gn(x)
        return self.relu(x)


class BackboneCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels//8,
                kernel_size=3, stride=2,
                padding=1
            ),
            nn.GroupNorm(num_groups, out_channels//8), nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            Bottleneck(
                in_channels=out_channels//8,
                mid_channels=out_channels//8,
                out_channels=out_channels//4,
                stride=2,
                num_groups=num_groups
            ),
            SepConv(
                in_channels=out_channels//4,
                out_channels=out_channels//4,
                num_groups=num_groups
            )
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels//4,
                out_channels=out_channels//2,
                kernel_size=3, stride=1,
                padding=2, dilation=2
            ),
            nn.GroupNorm(num_groups, out_channels//2), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1)
        )

        self.block4 = nn.Sequential(
            Bottleneck(
                in_channels=out_channels//2,
                mid_channels=out_channels//4,
                out_channels=out_channels,
                stride=2,
                num_groups=num_groups
            ),
            nn.Dropout2d(0.1)
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


class RoIMLPHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class RoIPredictor(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(RoIPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class CustomFasterRCNN(nn.Module):
    def __init__(self, anchor_sizes, anchor_ratios, num_classes=2):
        super().__init__()
        self.backbone = BackboneCNN(in_channels=1, out_channels=256)
        self._init_rpn(anchor_sizes, anchor_ratios)
        self._init_roi_heads(num_classes)
    
    def _init_rpn(self, anchor_sizes, anchor_ratios):
        anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
        rpn_head = RPNHead(
            in_channels=self.backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0]
        )

        self.rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator, head=rpn_head,
            fg_iou_thresh=0.7, bg_iou_thresh=0.3,
            batch_size_per_image=128, positive_fraction=0.5,
            pre_nms_top_n={"training": 800, "testing": 500},
            post_nms_top_n={"training": 600, "testing": 300},
            nms_thresh=0.7
        )

    def _init_roi_heads(self, num_classes):
        self.roi_head = RoIHeads(
            box_roi_pool=MultiScaleRoIAlign(
                featmap_names=["0"], output_size=7, sampling_ratio=2
            ),
            box_head=RoIMLPHead(
                in_channels=self.backbone.out_channels * 49, out_channels=1024
            ),
            box_predictor=RoIPredictor(
                in_channels=1024, num_classes=num_classes
            ),
            fg_iou_thresh=0.5, bg_iou_thresh=0.5,
            batch_size_per_image=256, positive_fraction=0.5,
            bbox_reg_weights=None,
            score_thresh=0.7, nms_thresh=0.3, detections_per_img=20
        )
    
    def forward(self, images, targets=None):
        if isinstance(images, torch.Tensor):
            images = [images]

        image_sizes = [(image.shape[-2], image.shape[-1]) for image in images]

        x = torch.stack(images)
        feat = self.backbone(x)
        features = {"0": feat}

        image_list = ImageList(x, image_sizes)
        proposals, rpn_losses = self.rpn(image_list, features, targets)

        detections, roi_losses = self.roi_head(features, proposals, image_sizes, targets)

        if self.training:
            losses = {}
            losses.update(rpn_losses)
            losses.update(roi_losses)
            return losses
        else:
            return detections


class CharCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128 * 12 * 12, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.fc(self.net(x))