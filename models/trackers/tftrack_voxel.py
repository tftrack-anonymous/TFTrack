import numpy as np
import torch
from mmengine.model import BaseModel
from mmengine.registry import MODELS

from datasets import points_utils
from datasets.metrics import estimateAccuracy, estimateOverlap


@MODELS.register_module()
class TFTrackVoxel(BaseModel):

    def __init__(self,
                 backbone=None,
                 fuser=None,
                 head=None,
                 cfg=None):
        super().__init__()
        self.config = cfg
        self.backbone = MODELS.build(backbone)
        self.fuse = MODELS.build(fuser)
        self.head = MODELS.build(head)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self,
                inputs,
                data_samples=None,
                mode: str = 'predict',
                **kwargs):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        if mode == 'predict':
            return self.predict(inputs)
        raise RuntimeError(f'Invalid mode "{mode}". '
                           'Only supports loss and predict mode')

    def get_feats(self, inputs):
        stack_feats = self.backbone(self._prepare_points(inputs['this_points']))
        cat_feats = self.fuse(stack_feats)
        if self.config.box_aware:
            wlh = inputs['wlh']
            wlh = torch.stack(wlh) if isinstance(wlh, list) else wlh.unsqueeze(0)
            return self.head(cat_feats, wlh)
        return self.head(cat_feats)

    @torch.no_grad()
    def inference(self, inputs):
        results = self.get_feats(inputs)
        coors = results['coors'][0]
        if self.config.use_rot:
            rot = results['rotation'][0]
            return coors, rot
        return coors

    def loss(self, inputs, data_samples):
        results = self.get_feats(inputs)
        return self.head.loss(results, data_samples)

    def predict(self, inputs):
        ious = []
        distances = []
        results_bbs = []
        last_coors = np.array([0., 0.])

        for frame_id, frame in enumerate(inputs):
            this_bb = frame['3d_bbox']
            if frame_id == 0:
                results_bbs.append(this_bb)
            else:
                data_dict, ref_bb, flag = self.build_input_dict(
                    inputs, frame_id, results_bbs)
                if flag:
                    if self.config.use_rot:
                        coors, rot = self.inference(data_dict)
                        rot = float(rot)
                    else:
                        coors = self.inference(data_dict)
                        rot = 0.
                    coors_x, coors_y, coors_z = map(float, coors[:3])
                    last_coors = np.array([coors_x, coors_y])
                    candidate_box = points_utils.getOffsetBB(
                        ref_bb, [coors_x, coors_y, coors_z, rot],
                        degrees=True, use_z=True, limit_box=False)
                else:
                    candidate_box = points_utils.getOffsetBB(
                        ref_bb, [last_coors[0], last_coors[1], 0, 0],
                        degrees=True, use_z=True, limit_box=False)
                results_bbs.append(candidate_box)

            this_overlap = estimateOverlap(
                this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])
            this_accuracy = estimateAccuracy(
                this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])
            ious.append(this_overlap)
            distances.append(this_accuracy)

        return ious, distances

    def _empty_point_dim(self):
        if self.backbone.__class__.__name__ == 'PillarNet':
            return 4
        return 3

    def _prepare_points(self, points):
        if self.backbone.__class__.__name__ == 'PillarNet':
            prepared = []
            for point in points:
                if point.size(1) < 4:
                    pad = point.new_zeros((point.size(0), 4 - point.size(1)))
                    point = torch.cat([point, pad], dim=1)
                prepared.append(point[:, :4])
            return prepared
        return [point[:, :3] for point in points]

    def build_input_dict(self, sequence, frame_id, results_bbs):
        assert frame_id > 0, "no need to construct an input_dict at frame 0"

        this_frame = sequence[frame_id]
        ref_box = results_bbs[-1]
        this_frame_pc = points_utils.crop_pc_in_range(
            this_frame['pc'], ref_box, self.config.point_cloud_range)
        this_points = this_frame_pc.points.T

        if self.config.post_processing:
            flag = not (this_points.shape[0] < 25 and frame_id < 15)
        else:
            flag = True

        if this_points.shape[0] < 1:
            this_points = np.zeros(
                (1, self._empty_point_dim()), dtype='float32')

        data_dict = {
            'this_points': [
                torch.as_tensor(
                    this_points, dtype=torch.float32, device=self.device)],
            'wlh': torch.as_tensor(
                ref_box.wlh, dtype=torch.float32, device=self.device),
        }
        return data_dict, ref_box, flag
