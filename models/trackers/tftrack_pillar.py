from mmengine.registry import MODELS

from .tftrack_voxel import TFTrackVoxel


@MODELS.register_module()
class TFTrackPillar(TFTrackVoxel):
    """Pillar tracker variant.

    PillarNet shares the same tracking flow as the voxel variant, but keeping a
    separate registered model name makes pillar configs and checkpoints easier
    to read.
    """
    pass
