import numpy as np
import torch
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from nuscenes.utils import geometry_utils

from datasets import points_utils
from datasets.metrics import estimateAccuracy, estimateOverlap


@MODELS.register_module()
class TFTrackPoint(BaseModel):

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
        if 'coord' in inputs:
            model_inputs = {
                key: value
                for key, value in inputs.items()
                if key in {'coord', 'feat', 'batch', 'grid_size'}
            }
            stack_feats = self.backbone(model_inputs)
            feats = stack_feats['feat']
            batch_idx = stack_feats['batch']
            wlh = inputs.get('wlh')
            cat_feats = self.fuse(
                feats,
                batch_idx,
                wlh if self.config.box_aware else None)
        else:
            points = [
                point[:, :3] for point in inputs['prev_points'] + inputs['this_points']
            ]
            stack_points = torch.stack(points)
            stack_feats = self.backbone(stack_points)
            wlh = inputs.get('wlh')
            if self.config.box_aware:
                wlh = torch.stack(wlh) if isinstance(wlh, list) else wlh.unsqueeze(0)
                cat_feats = self.fuse(stack_feats, wlh)
            else:
                cat_feats = self.fuse(stack_feats)

        return self.head(cat_feats)

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

    def _single_stream_inputs(self, this_points, ref_box):
        if this_points.shape[1] < 4:
            intensity = np.zeros((this_points.shape[0], 1), dtype=this_points.dtype)
            feats = np.concatenate([this_points, intensity], axis=1)
        else:
            feats = this_points[:, :4]

        device = self.device
        coords = torch.as_tensor(this_points[:, :3], dtype=torch.float32, device=device)
        feats = torch.as_tensor(feats, dtype=torch.float32, device=device)
        return {
            'coord': coords,
            'feat': feats,
            'batch': torch.zeros(coords.shape[0], dtype=torch.long, device=device),
            'grid_size': torch.as_tensor(
                getattr(self.config, 'grid_size', [0.075, 0.075, 0.15]),
                dtype=torch.float32,
                device=device),
            'wlh': torch.as_tensor(ref_box.wlh, dtype=torch.float32, device=device),
        }

    def build_input_dict(self, sequence, frame_id, results_bbs):
        assert frame_id > 0, "no need to construct an input_dict at frame 0"

        prev_frame = sequence[frame_id - 1]
        this_frame = sequence[frame_id]
        ref_box = results_bbs[-1]

        prev_frame_pc = points_utils.crop_pc_in_range(
            prev_frame['pc'], ref_box, self.config.point_cloud_range)
        this_frame_pc = points_utils.crop_pc_in_range(
            this_frame['pc'], ref_box, self.config.point_cloud_range)

        prev_points = prev_frame_pc.points.T
        this_points = this_frame_pc.points.T

        if self.config.post_processing:
            ref_bb = points_utils.transform_box(ref_box, ref_box)
            prev_idx = geometry_utils.points_in_box(ref_bb, prev_points.T, 1.25)
            flag = not (
                sum(prev_idx) < 3 and this_points.shape[0] < 25 and frame_id < 15)
        else:
            flag = True

        prev_points, _ = points_utils.regularize_pc(prev_points, 1024)
        this_points, _ = points_utils.regularize_pc(this_points, 1024)

        if getattr(self.config, 'single_stream', False):
            return self._single_stream_inputs(this_points, ref_box), ref_box, flag

        device = self.device
        data_dict = {
            'prev_points': [
                torch.as_tensor(prev_points, dtype=torch.float32, device=device)],
            'this_points': [
                torch.as_tensor(this_points, dtype=torch.float32, device=device)],
            'wlh': torch.as_tensor(ref_box.wlh, dtype=torch.float32, device=device),
        }
        return data_dict, ref_box, flag
