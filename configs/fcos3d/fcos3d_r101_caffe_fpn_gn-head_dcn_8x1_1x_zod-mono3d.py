_base_ = [
    '../_base_/datasets/zod-frames-mono3d.py', '../_base_/models/fcos3d.py',
    '../_base_/schedules/mmdet_schedule_1x.py', '../_base_/default_runtime.py'
]
class_names = [
    'Vehicle', 'VulnerableVehicle', 'Pedestrian', 'TrafficSign',
    'TrafficSignal'
]
# model settings
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    # default fcos3d head but without velocity and attributes
    bbox_head=dict(
        num_classes=len(class_names),
        pred_attrs=False,
        pred_velo=False,
        group_reg_dims=(2, 1, 3, 1),  # offset, depth, size, rot
        reg_branch=(
            (256, ),  # offset
            (256, ),  # depth
            (256, ),  # size
            (256, ),  # rot
        ),
        bbox_coder=dict(type='FCOS3DBBoxCoder', code_size=7),
        bbox_code_size=7,
    ),
    # Default training config but remove weights for velocity
    train_cfg=dict(
        code_weight=[1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0], debug=True),
)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        # with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='ResizeMono3D', img_scale=(1920, 1056), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d', 'gt_labels_3d',
            'centers2d', 'depths'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='ResizeMono3D', img_scale=(1920, 1056), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img']),
        ])
]
_mmdir = 'data/zod/mmdet3d'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        times=1,
        dataset=dict(
            pipeline=train_pipeline,
            classes=class_names,
            ann_file=f'{_mmdir}/zod_infos_train_mono3d.coco.json',
            use_png=True,
            anonymization_mode='original')),  # Train is a wrapped dataset
    val=dict(
        pipeline=test_pipeline,
        classes=class_names,
        ann_file=f'{_mmdir}/zod_infos_val_mono3d.coco.json',
        use_png=True,
        anonymization_mode='original'),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    lr=0.001, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
total_epochs = 12
evaluation = dict(interval=2)

log_config = dict(
    # interval=10,
    # interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='MMDet3DWandbHook',
            init_kwargs=dict(
                project='mmdet-test',
                name='fcos3d_mono_zod',
            ),
            log_checkpoint=False,
            log_checkpoint_metadata=False,
            num_eval_images=10,
            bbox_score_thr=0.1,
            visualize_img=True,
            visualize_3d=True,
        )
    ])
