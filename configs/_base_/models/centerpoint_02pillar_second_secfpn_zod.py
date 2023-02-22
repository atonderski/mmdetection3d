_base_ = './centerpoint_02pillar_second_secfpn_nus.py'

voxel_size = [0.2, 0.2, 8]
model = dict(
    pts_voxel_layer=dict(
        max_num_points=20, voxel_size=voxel_size, max_voxels=(30000, 40000)),
    pts_bbox_head=dict(
        tasks=[
            dict(num_class=1, class_names=['Vehicle']),
            dict(num_class=1, class_names=['VulnerableVehicle']),
            dict(num_class=1, class_names=['Pedestrian']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), _delete_=True),
        bbox_coder=dict(
            post_center_range=[-61.2, -10.0, -10.0, 112.4, 61.2, 10.0],
            voxel_size=voxel_size[:2],
            code_size=7),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-61.2, -10.0, -10.0, 61.2, 112.4, 10.0],
            max_per_img=500,
            pc_range=[-51.2, 0],
            voxel_size=voxel_size[:2],
        )))
