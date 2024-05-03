_base_ = './centerpoint_02pillar_second_secfpn_nus.py'

voxel_size = [0.2, 0.2, 8]
bev_range = [-76.8, 0, -5.0, 76.8, 256.0, 3.0]
post_center_range = [-86.8, -10.0, -10.0, 86.8, 266.0, 10.0]  # bev_range + pad
# Compute bev-grid resolution
grid_x = int((bev_range[3] - bev_range[0]) / voxel_size[0])
grid_y = int((bev_range[4] - bev_range[1]) / voxel_size[1])

model = dict(
    pts_voxel_layer=dict(
        max_num_points=20,
        voxel_size=voxel_size,
        max_voxels=(50000, 70000),
        point_cloud_range=bev_range,
    ),
    pts_voxel_encoder=dict(point_cloud_range=bev_range),
    pts_middle_encoder=dict(
        output_shape=(grid_y, grid_x)),  # depends on bev_range and voxel_size
    pts_bbox_head=dict(
        tasks=[
            dict(num_class=1, class_names=['Vehicle']),
            dict(num_class=1, class_names=['VulnerableVehicle']),
            dict(num_class=1, class_names=['Pedestrian']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), _delete_=True),
        bbox_coder=dict(
            pc_range=bev_range[:-2],
            post_center_range=post_center_range,
            voxel_size=voxel_size[:2],
            code_size=7,
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            point_cloud_range=bev_range,
            grid_size=[grid_x, grid_y, 1
                       ],  # depends on bev_range and voxel_size
        )),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=post_center_range,
            max_per_img=500,
            pc_range=[-76.8, 0],
            voxel_size=voxel_size[:2],
        )))
