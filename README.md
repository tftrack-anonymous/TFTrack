# TFTrack

TFTrack collects three single-object LiDAR tracking variants in one release tree:

- `configs/voxel`: voxel backbone variant.
- `configs/ptv3`: PointTransformerV3 point-backbone variant.
- `configs/pillar`: PillarNet backbone variant.

The shared tracker registry exposes `TFTrackVoxel`, `TFTrackPoint`, and `TFTrackPillar`; the backbone registry exposes `VoxelNet`, `PointNet`, `PointTransformerV3`, and `PillarNet` when their dependencies are installed.

## Layout

```text
configs/
  voxel/      # voxel configs for KITTI, NuScenes, Waymo
  point/      # original PointNet point configs
  ptv3/       # PointTransformerV3 configs
  pillar/     # PillarNet configs
datasets/     # KITTI, NuScenes, Waymo loaders and samplers
models/       # backbones, fusers, heads, trackers
ops/          # PillarNet CUDA op sources
docs/         # install, data, training, testing notes
```

## Install

Create the environment with the versions in `docs/INSTALL.md`, then install the extra PTV3/Pillar dependencies listed in `requirements.txt`.

For the pillar CUDA ops:

```shell
cd ops/pillar_ops
python setup.py develop
```

Edit each config's `data_dir` before training or testing.

## Train

```shell
python train.py --config configs/voxel/kitti/car.py
python train.py --config configs/ptv3/nuscenes/car.py
python train.py --config configs/pillar/kitti/car.py
```

## Test

```shell
python test.py --config configs/voxel/kitti/car.py --load_from path/to/checkpoint.pth
python test.py --config configs/ptv3/nuscenes/car.py --load_from path/to/checkpoint.pth
python test.py --config configs/pillar/kitti/car.py --load_from path/to/checkpoint.pth
```

Use `dist_train.sh` and `dist_test.sh` for distributed runs after setting the config and checkpoint arguments.
