_base_ = [
    'centerpoint_02pillar_second_secfpn_4x8_cyclic_40e_zod.py',
]

point_cloud_range = [-76.8, 0, -5.0, 76.8, 256.0, 3.0]
class_names = ['Vehicle', 'VulnerableVehicle', 'Pedestrian']
data_root = 'data/zod/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='ZodLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='ZodLoadPointsFromMultiSweeps',
        use_dim=[0, 1, 2, 3, 4],
        time_dim=4,
        sweeps_num=2,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
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
    dict(
        type='ZodLoadPointsFromMultiSweeps',
        use_dim=[0, 1, 2, 3, 4],
        time_dim=4,
        sweeps_num=2,
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
eval_pipeline = [
    dict(
        type='ZodLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='ZodLoadPointsFromMultiSweeps',
        use_dim=[0, 1, 2, 3, 4],
        time_dim=4,
        sweeps_num=2,
        file_client_args=file_client_args),
    dict(type='LoadImageFromFile'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

evaluation = dict(pipeline=eval_pipeline)
