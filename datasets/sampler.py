import numpy as np
import torch
from nuscenes.utils import geometry_utils
from torch.utils.data import Dataset
from . import points_utils
from mmengine.registry import DATASETS
from mmengine.registry import FUNCTIONS
from nuscenes.nuscenes import NuScenes


class KalmanFiltering:
    def __init__(self, bnd=[1, 1, 10]):
        self.bnd = bnd
        self.reset()

    def sample(self, n=10):
        return np.random.multivariate_normal(self.mean, self.cov, size=n)

    def addData(self, data, score):
        score = score.clip(min=1e-5)  # prevent sum=0 in case of bad scores
        self.data = np.concatenate((self.data, data))
        self.score = np.concatenate((self.score, score))
        self.mean = np.average(self.data, weights=self.score, axis=0)
        self.cov = np.cov(self.data.T, ddof=0, aweights=self.score)

    def reset(self):
        self.mean = np.zeros(len(self.bnd))
        self.cov = np.diag(self.bnd)
        if len(self.bnd) == 2:
            self.data = np.array([[], []]).T
        else:
            self.data = np.array([[], [], []]).T
        self.score = np.array([])
        


@DATASETS.register_module()
class TrainSampler(torch.utils.data.Dataset):

    def __init__(self, dataset=None, cfg=None):
        super().__init__()
        self.config = cfg
        self.dataset = DATASETS.build(dataset)
        self.num_candidates = cfg.num_candidates
        num_frames_total = 0
        self.tracklet_start_ids = [num_frames_total]
        for i in range(self.dataset.get_num_tracklets()):
            num_frames_total += self.dataset.get_num_frames_tracklet(i)
            self.tracklet_start_ids.append(num_frames_total)

        
    @staticmethod
    def processing(data, config):
        prev_frame = data['prev_frame']
        this_frame = data['this_frame']
        prev_pc, prev_box = prev_frame['pc'], prev_frame['3d_bbox']
        this_pc, this_box = this_frame['pc'], this_frame['3d_bbox']

        prev_frame_pc = points_utils.crop_pc_in_range(prev_pc, prev_box, config.point_cloud_range)
        this_frame_pc = points_utils.crop_pc_in_range(this_pc, prev_box, config.point_cloud_range)

        prev_box_local = points_utils.transform_box(prev_box, prev_box)
        this_box_local = points_utils.transform_box(this_box, prev_box)


        prev_points = prev_frame_pc.points.T
        this_points = this_frame_pc.points.T
        
        if config.regular_pc:
            prev_points, _ = points_utils.regularize_pc(prev_points, 1024)
            this_points, _ = points_utils.regularize_pc(this_points, 1024)
        else:
            if prev_points.shape[0] < 1:
                prev_points = np.zeros((1, 3), dtype='float32')
            if this_points.shape[0] < 1:
                this_points = np.zeros((1, 3), dtype='float32')

        if config.flip:
            prev_points, prev_box_local, this_points, this_box_local = \
                points_utils.flip_augmentation(prev_points, prev_box_local, this_points, this_box_local)

        theta = this_box_local.orientation.degrees * this_box_local.orientation.axis[-1]
        box_label = this_box_local.center

        inputs = {
            'prev_points': torch.as_tensor(prev_points, dtype=torch.float32),
            'this_points': torch.as_tensor(this_points, dtype=torch.float32),
            'wlh': torch.as_tensor(prev_box.wlh, dtype=torch.float32),
        }
        data_samples = {
            'box_label': torch.as_tensor(box_label, dtype=torch.float32),
            'theta': torch.as_tensor([0.2 * theta], dtype=torch.float32),
        }

        return {
            'inputs': inputs,
            'data_samples': data_samples,
        }

    def get_anno_index(self, index):
        return index // self.num_candidates

    def get_candidate_index(self, index):
        return index % self.num_candidates

    def __len__(self):
        return self.dataset.get_num_frames_total() * self.num_candidates

    def __getitem__(self, index):
        anno_id = self.get_anno_index(index)
        candidate_id = self.get_candidate_index(index)
        try:
            for i in range(0, self.dataset.get_num_tracklets()):
                if self.tracklet_start_ids[i] <= anno_id < self.tracklet_start_ids[i + 1]:
                    tracklet_id = i
                    this_frame_id = anno_id - self.tracklet_start_ids[i]
                    prev_frame_id = max(this_frame_id - 1, 0)
                    frame_ids = (prev_frame_id, this_frame_id)
            prev_frame, this_frame = self.dataset.get_frames(tracklet_id, frame_ids=frame_ids)
            data = {
                "prev_frame": prev_frame,
                "this_frame": this_frame,
                "candidate_id": candidate_id}
            return self.processing(data, self.config)
        except AssertionError:
            return self[torch.randint(0, len(self), size=(1,)).item()]


@DATASETS.register_module()
class TestSampler(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = DATASETS.build(dataset)

    def __len__(self):
        return self.dataset.get_num_tracklets()

    def __getitem__(self, index):
        tracklet_annos = self.dataset.tracklet_anno_list[index]
        frame_ids = list(range(len(tracklet_annos)))
        return self.dataset.get_frames(index, frame_ids)


@FUNCTIONS.register_module()
def my_collate_fn(samples):
    coord_list, feat_list, batch_list, wlh_list = [], [], [], []

    for idx, sample in enumerate(samples):
        coords = sample['inputs']['coord']
        feats = sample['inputs']['feat']
        coord_list.append(coords)
        feat_list.append(feats)
        batch_list.append(torch.full((coords.shape[0],), idx, dtype=torch.long))
        wlh_list.append(sample['inputs']['wlh'])

    return {
        'inputs': {
            'coord': torch.cat(coord_list, dim=0),
            'feat': torch.cat(feat_list, dim=0),
            'batch': torch.cat(batch_list, dim=0),
            'grid_size': samples[0]['inputs']['grid_size'],
            'wlh': torch.stack(wlh_list),
        },
        'data_samples': [sample['data_samples'] for sample in samples],
    }
