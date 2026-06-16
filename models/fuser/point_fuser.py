import torch
import torch.nn as nn
from mmengine.registry import MODELS


class MLPMixer(nn.Sequential):

    def __init__(self, in_channels, out_channels, embed_dim=1024):
        super().__init__(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.SyncBatchNorm(embed_dim, eps=1e-3, momentum=0.01),
        )

    def forward(self, inputs):
        return super().forward(inputs)


@MODELS.register_module()
class PointFuser(nn.Module):

    def __init__(self, box_aware, input_feat_dim=None, num_points=1024):
        super().__init__()
        self.box_aware = box_aware
        self.input_feat_dim = input_feat_dim
        self.num_points = num_points
        in_channels = (input_feat_dim if input_feat_dim is not None else 2)
        if box_aware:
            in_channels += 1
        self.fuse = nn.Sequential(
            MLPMixer(in_channels, 64, embed_dim=num_points),
            MLPMixer(64, 128, embed_dim=num_points),
            MLPMixer(128, 256, embed_dim=num_points),
            nn.Linear(256, 1),
            nn.SyncBatchNorm(num_points, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Flatten(),
        )
        if box_aware:
            self.wlh_mlp = nn.Sequential(
                nn.Linear(3, 128),
                nn.SyncBatchNorm(128, eps=1e-3, momentum=0.01),
                nn.ReLU(True),
                nn.Linear(128, num_points)
            )

    def _pack_by_batch(self, feats, batch_idx):
        batch_size = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0
        packed = feats.new_zeros((batch_size, self.num_points, feats.size(-1)))
        for batch_id in range(batch_size):
            sample_feats = feats[batch_idx == batch_id]
            num_points = min(sample_feats.size(0), self.num_points)
            if num_points:
                packed[batch_id, :num_points] = sample_feats[:num_points]
        return packed

    def forward(self, stack_feats, batch_idx=None, wlh=None):
        if wlh is None and batch_idx is not None and batch_idx.dtype != torch.long:
            wlh = batch_idx
            batch_idx = None

        if batch_idx is None:
            prev_feats, this_feats = torch.split(
                stack_feats, stack_feats.size(0) // 2, 0)
            cat_feats = torch.cat([prev_feats, this_feats], 2)
        else:
            cat_feats = self._pack_by_batch(stack_feats, batch_idx)

        if self.box_aware and wlh is not None:
            if wlh.ndim == 3 and wlh.size(0) == 1:
                wlh = wlh.squeeze(0)
            wlh = self.wlh_mlp(wlh).unsqueeze(-1)
            cat_feats = torch.cat([cat_feats, wlh], 2)
        return self.fuse(cat_feats)
