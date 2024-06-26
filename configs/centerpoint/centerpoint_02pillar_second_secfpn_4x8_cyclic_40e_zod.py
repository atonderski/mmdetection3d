_base_ = [
    '../_base_/datasets/zod-frames-3d.py',
    '../_base_/models/centerpoint_02pillar_second_secfpn_zod.py',
    '../_base_/schedules/cyclic_40e.py',
    '../_base_/default_runtime.py',
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-76.8, 0, -5.0, 76.8, 256.0, 3.0]

# For ZOD we usually do 3-class detection
class_names = ['Vehicle', 'VulnerableVehicle', 'Pedestrian']

dataset_type = 'ZodFramesDataset'
data_root = 'data/zod/'
file_client_args = dict(backend='disk')

# db_sampler = dict(
#     data_root=data_root,
#     info_path=data_root + 'nuscenes_dbinfos_train.pkl',
#     rate=1.0,
#     bbox_code_size=7,
#     prepare=dict(
#         filter_by_difficulty=[-1],
#         filter_by_min_points=dict(
#             car=5,
#             truck=5,
#             bus=5,
#             trailer=5,
#             construction_vehicle=5,
#             traffic_cone=5,
#             barrier=5,
#             motorcycle=5,
#             bicycle=5,
#             pedestrian=5)),
#     classes=class_names,
#     sample_groups=dict(
#         car=2,
#         truck=3,
#         construction_vehicle=7,
#         bus=4,
#         trailer=6,
#         barrier=2,
#         motorcycle=6,
#         bicycle=6,
#         pedestrian=2,
#         traffic_cone=2),
#     points_loader=dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=5,
#         use_dim=[0, 1, 2, 3, 4],
#         file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='ZodLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='ZodLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='ZodLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadImageFromFile'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=10,
    workers_per_gpu=16,
    train=dict(
        times=1,
        dataset=dict(
            ann_file=data_root + 'mmdet3d/zod_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names)),
    val=dict(
        pipeline=test_pipeline,
        classes=class_names,
        ann_file=data_root + 'mmdet3d/zod_infos_val.pkl'),
    test=dict(
        pipeline=test_pipeline,
        classes=class_names,
        ann_file=data_root + 'mmdet3d/zod_infos_val.pkl'))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='MMDet3DWandbHook',
            init_kwargs=dict(
                project='mmdet-test',
                name='centerpoint_zod',
            ),
            log_checkpoint=False,
            log_checkpoint_metadata=False,
            num_eval_images=1,
            bbox_score_thr=0.1,
            visualize_3d=True,
            visualize_img=True,
            max_points=20000,
        )
    ])

evaluation = dict(interval=10, pipeline=eval_pipeline)
