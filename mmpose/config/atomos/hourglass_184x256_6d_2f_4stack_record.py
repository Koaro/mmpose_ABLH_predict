_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/coco.py'
]
evaluation = dict(interval=10, metric='mAP', save_best='AP')

optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
log_config = dict(
    interval=50, 
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
channel_cfg = dict(
    num_output_channels=1,
    dataset_joints=1,
    dataset_channel=[
        [0],
    ],
    inference_channel=[
        0
    ])

# model settings
model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='HourglassNet',
        num_stacks=4,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapMultiStageHead',
        in_channels=256,
        out_channels=channel_cfg['num_output_channels'],
        num_stages=1,
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[128, 128],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file=''
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = '../../datasets/atomos'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    val_dataloader=dict(samples_per_gpu=8),
    test_dataloader=dict(samples_per_gpu=8),
    train=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/record/train_LIDAR_6day.json',
        img_prefix=f'{data_root}/2-feature/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/record/val_LIDAR.json',
        img_prefix=f'{data_root}/2-feature/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/record/val_LIDAR.json',
        img_prefix=f'{data_root}/2-feature/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
