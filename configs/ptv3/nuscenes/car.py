_base_ = '../../default_runtime.py'
data_dir = 'data/nuscenes'
category_name = 'Car'
batch_size = 128
point_cloud_range = [-4.8, -4.8, -1.5, 4.8, 4.8, 1.5]
box_aware = True
use_rot = False


model = dict(
    type='TFTrackPoint',
    backbone=dict(
        type='PointTransformerV3',
        in_channels=4,               # since you had points_features=3
        order=('z', 'z-trans'),      # or any desired order(s)
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=False,  # requires flash_attn installed
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        # PDNorm settings omitted unless you use them
    ),
    fuser=dict(
        type='PointFuser',
        box_aware=box_aware,
        input_feat_dim=64,
    ),
    head=dict(
        type='PointHead',
        q_distribution='laplace',  # ['laplace', 'gaussian']
        use_rot=use_rot
    ),
    cfg=dict(
        point_cloud_range=point_cloud_range,
        box_aware=box_aware,
        post_processing=False,
        use_rot=use_rot,
        single_stream=True,
        grid_size=[0.075, 0.075, 0.15],
    )
)

train_dataset = dict(
    type='TrainSampler',
    dataset=dict(
        type='NuScenesDataset',
        path=data_dir,
        split='train_track',
        category_name=category_name,
        preloading=False,
        preload_offset=10,
    ),
    cfg=dict(
        num_candidates=4,
        target_thr=None,
        search_thr=5,
        point_cloud_range=point_cloud_range,
        regular_pc=True,
        flip=True,
        single_stream=True,
        grid_size=[0.075, 0.075, 0.15],
    )
)

test_dataset = dict(
    type='TestSampler',
    dataset=dict(
        type='NuScenesDataset',
        path=data_dir,
        split='val',
        category_name=category_name,
        preloading=False
    ),
)

train_dataloader = dict(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='my_collate_fn')
    )

val_dataloader = dict(
    dataset=test_dataset,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=lambda x: x,
)

test_dataloader = dict(
    dataset=test_dataset,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=lambda x: x,
)
