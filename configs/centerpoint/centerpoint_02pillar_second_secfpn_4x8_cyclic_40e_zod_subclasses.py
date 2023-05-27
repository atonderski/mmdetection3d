_base_ = [
    'centerpoint_02pillar_second_secfpn_4x8_cyclic_40e_zod.py',
]

# Here we use subclasses of the ZOD dataset
classes_veh = [
    'Vehicle_Car',
    'Vehicle_Van',
    'Vehicle_Truck',
    'Vehicle_Bus',
    'Vehicle_Trailer',
    'Vehicle_TramTrain',
    'Vehicle_HeavyEquip',
    'Vehicle_Emergency',
    'Vehicle_Other',
]
classes_vuln_veh = [
    'VulnerableVehicle_Bicycle',
    'VulnerableVehicle_Motorcycle',
    'VulnerableVehicle_Stroller',
    'VulnerableVehicle_Wheelchair',
    'VulnerableVehicle_PersonalTransporter',
    'VulnerableVehicle_NoRider',
    'VulnerableVehicle_Other',
]
classes_dyn_rest = [
    'Pedestrian',
    'Animal',
]
classes_stat = [
    'PoleObject', 'TrafficBeacon', 'TrafficGuide', 'DynamicBarrier',
    'TrafficSign_Front', 'TrafficSign_Back', 'TrafficSignal_Front',
    'TrafficSignal_Back'
]

class_names = classes_veh + classes_vuln_veh + classes_dyn_rest + classes_stat

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-76.8, 0, -5.0, 76.8, 256.0, 3.0]
data_root = 'data/zod/'
file_client_args = dict(backend='disk')

model = dict(
    pts_bbox_head=dict(tasks=[
        dict(num_class=len(classes_veh), class_names=classes_veh),
        dict(num_class=len(classes_vuln_veh), class_names=classes_vuln_veh),
        dict(num_class=len(classes_dyn_rest), class_names=classes_dyn_rest),
        dict(num_class=len(classes_stat), class_names=classes_stat),
    ]))

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
    train=dict(
        dataset=dict(
            pipeline=train_pipeline,
            classes=class_names,
            merge_subclasses=False)),
    val=dict(
        pipeline=test_pipeline, classes=class_names, merge_subclasses=False),
    test=dict(
        pipeline=test_pipeline, classes=class_names, merge_subclasses=False),
)

evaluation = dict(pipeline=eval_pipeline)
