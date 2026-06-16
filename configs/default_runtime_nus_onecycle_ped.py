default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1,
                    save_best='precision', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = dict(type='Visualizer',
                  vis_backends=[dict(type='LocalVisBackend'),
                                dict(type='TensorboardVisBackend')])

epoch_num = 150
lr = 1.5*0.0001 #0.0001

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.00001), # betas=(0.9, 0.999),
    clip_grad=dict(max_norm=35, norm_type=2),
    constructor='DefaultOptimWrapperConstructor',
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.)
)

param_scheduler = [
    dict(type='LinearLR', #lr warmup
        start_factor=0.00001,
        by_epoch=False,
        begin = 0,
        end = 50),
    dict(type='CosineAnnealingLR', # 主学习率调度器
         T_max=50,
         eta_min=lr*0.01,
        #  begin=0,
        #  end=10,
         by_epoch=True,
         convert_to_iter_based=True),  
]
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=1)
val_cfg = dict()
test_cfg = dict()

custom_imports = dict(
    imports=['models', 'datasets'],
    allow_failed_imports=False
)