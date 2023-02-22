# Copyright (c) OpenMMLab. All rights reserved.
import os
from os import path as osp
from typing import List, Union

import mmcv
import numpy as np
from pyquaternion import Quaternion
from zod import ZodFrames
from zod.anno.object import OBJECT_CLASSES as ALL_CLASSES
from zod.constants import AnnotationProject, Anonymization, Camera, Lidar
from zod.data_classes.box import Box3D
from zod.data_classes.calibration import CameraCalibration, LidarCalibration
from zod.data_classes.geometry import Pose
from zod.data_classes.oxts import EgoMotion
from zod.data_classes.sensor import SensorFrame

from mmdet3d.core.bbox.structures.utils import points_cam2img


def create_zod_infos(root_path,
                     out_dir,
                     info_prefix,
                     version='full',
                     max_sweeps=10):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        ourt_dir (str): Path of the output directory.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    if version == 'single':
        print('Will only use a single (the second) frame from the mini set.')
        zod = ZodFrames(root_path, 'mini')
        train_scenes = zod.get_split('val')[1:2]
        val_scenes = zod.get_split('val')[1:2]
    else:
        zod = ZodFrames(root_path, version)
        train_scenes = zod.get_split('train')
        val_scenes = zod.get_split('val')
    train_infos = _fill_infos(zod, train_scenes, max_sweeps=max_sweeps)
    val_infos = _fill_infos(zod, val_scenes, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    print('train sample: {}, val sample: {}'.format(
        len(train_infos), len(val_infos)))
    data = dict(infos=train_infos, metadata=metadata)
    info_path = osp.join(out_dir, '{}_infos_train.pkl'.format(info_prefix))
    mmcv.dump(data, info_path)
    data['infos'] = val_infos
    info_val_path = osp.join(out_dir, '{}_infos_val.pkl'.format(info_prefix))
    mmcv.dump(data, info_val_path)


def _fill_infos(zod: ZodFrames, frames: List[str], max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        zod (:obj:`ZodFrames`): Dataset class.
        frames (list[str]): IDs of the training/validation frames.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        list[dict]: Information that will be saved to the info file.
    """
    infos = []
    for frame_id in mmcv.track_iter_progress(frames):
        frame = zod[frame_id]
        lidar_frame = frame.get_lidar_frames()[0]
        mmcv.check_file_exist(lidar_frame.filepath)
        lidar_calib = frame.calibration.lidars[Lidar.VELODYNE]
        core_lidar2ego = lidar_calib.extrinsics
        # TODO: Should timestamp be from lidar or camera?
        core_timestamp = lidar_frame.time.timestamp()
        core_ego_pose = Pose(frame.ego_motion.get_poses(core_timestamp))
        info = {
            'lidar_path': lidar_frame.filepath,
            'frame_id': frame_id,
            'sweeps': [],
            'cams': dict(),
            'lidar2ego_translation': core_lidar2ego.translation,
            'lidar2ego_rotation': core_lidar2ego.rotation,
            'ego2global_translation': core_ego_pose.translation,
            'ego2global_rotation': core_ego_pose.rotation,
            'timestamp': core_timestamp,
        }

        cameras = [
            Camera.FRONT,
        ]
        for cam in cameras:
            cam_calib = frame.calibration.cameras[cam]
            cam_info = obtain_sensor2lidar(
                frame.get_camera_frame(Anonymization.BLUR),
                cam_calib,
                core_ego_pose,
                core_lidar2ego,
                frame.ego_motion,
                cam.value,  # TODO: Check what this is used for
            )
            cam_info.update(
                cam_intrinsic=cam_calib.intrinsics,
                cam_distortion=cam_calib.distortion,
                cam_undistortion=cam_calib.undistortion,
                proj_model='kannala',
            )
            info['cams'].update({cam.value: cam_info})

        # obtain sweeps for a single key-frame
        info['sweeps'] = [
            obtain_sensor2lidar(lf, lidar_calib, core_ego_pose, core_lidar2ego,
                                frame.ego_motion, 'lidar')
            for lf in frame.get_lidar_frames(num_before=max_sweeps)[:-1]
        ]

        # obtain annotation
        annos = frame.get_annotation(AnnotationProject.OBJECT_DETECTION)
        locs = np.array([
            b.box3d.center if b.box3d else [-1, -1, -1] for b in annos
        ]).reshape(-1, 3)
        dims = np.array([
            b.box3d.size if b.box3d else [-1, -1, -1] for b in annos
        ]).reshape(-1, 3)
        rots = np.array([
            b.box3d.orientation.yaw_pitch_roll[0] if b.box3d else -1
            for b in annos
        ]).reshape(-1, 1)
        gt_boxes = np.concatenate([locs, dims, rots], axis=1)
        # TODO: maybe check number of lidar points
        # valid_flag = np.array(
        #     [
        #         (anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
        #         for anno in annotations
        #     ],
        #     dtype=bool,
        # ).reshape(-1)
        valid_flag = np.ones(gt_boxes.shape[0], dtype=bool)
        has_3d = np.array([a.box3d is not None for a in annos], dtype=bool)
        is_ignore = np.array([a.should_ignore_object() for a in annos],
                             dtype=bool)

        names = [b.name for b in annos]
        names = np.array(names)

        assert len(gt_boxes) == len(annos), f'{len(gt_boxes)}, {len(annos)}'
        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names
        info['gt_boxes_2d'] = np.array([b.box2d.xyxy
                                        for b in annos]).reshape(-1, 4)
        # info["num_lidar_pts"] = np.array(
        #   [a["num_lidar_pts"] for a in annotations]
        # )
        info['valid_flag'] = valid_flag
        info['has_3d'] = has_3d
        info['is_ignore'] = is_ignore

        infos.append(info)

    return infos


def obtain_sensor2lidar(
    sensor_frame: SensorFrame,
    sensor_calib: Union[CameraCalibration, LidarCalibration],
    core_ego_pose: Pose,
    core_lidar2ego: Pose,
    ego_motion: EgoMotion,
    sensor_type: str,
) -> dict:
    """Obtain the info with RT matric from general sensor to core (top)
    LiDAR."""
    ego_pose = Pose(ego_motion.get_poses(sensor_frame.time.timestamp()))
    sensor2ego = sensor_calib.extrinsics
    sweep = {
        'data_path': sensor_frame.filepath,
        'type': sensor_type,
        'sensor2ego_translation': sensor2ego.translation,
        'sensor2ego_rotation': sensor2ego.rotation,
        'ego2global_translation': ego_pose.translation,
        'ego2global_rotation': ego_pose.rotation,
        'timestamp': sensor_frame.time.timestamp(),
    }
    # transforms for sweep frame
    s2e_t = sensor_calib.extrinsics.translation
    e2g_t_s = ego_pose.translation
    s2e_r_mat = sensor_calib.extrinsics.rotation_matrix
    e2g_r_s_mat = ego_pose.rotation_matrix

    # transforms for core frame
    e2g_r_mat = core_ego_pose.rotation_matrix
    e2g_t = core_ego_pose.translation
    l2e_r_mat = core_lidar2ego.rotation_matrix
    l2e_t = core_lidar2ego.translation

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    R = (s2e_r_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (s2e_t @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) +
        l2e_t @ np.linalg.inv(l2e_r_mat).T)
    # NOTE: this is a bit strange, but sensor2lidar_rotation is a
    # rotation matrix (other rotations are quaternions)
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool, optional): Whether to export mono3d annotation.
            Default: True.
    """
    version = version if version != 'single' else 'mini'
    zod = ZodFrames(root_path, version=version)

    # get bbox annotations for camera
    camera_types = [Camera.FRONT]
    infos = mmcv.load(info_path)['infos']

    # info_2d_list = []
    cat2Ids = [
        dict(id=i, name=cat_name) for i, cat_name in enumerate(ALL_CLASSES)
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(infos):
        frame = zod[info['frame_id']]
        for cam in camera_types:
            image_id = f"camera_{cam.value}_{info['frame_id']}"
            cam_info = info['cams'][cam.value]
            coco_infos = get_2d_boxes(
                image_id,
                info,
                cam_info,
                mono3d=mono3d,
            )
            # alternative if this is slow:
            width = frame.get_camera_frame().width
            height = frame.get_camera_frame().height
            coco_2d_dict['images'].append(
                dict(
                    file_name=os.path.relpath(cam_info['data_path'],
                                              root_path),
                    id=image_id,
                    frame_id=info['frame_id'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'].elements,
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'].elements,
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    cam_distortion=cam_info['cam_distortion'],
                    cam_undistortion=cam_info['cam_undistortion'],
                    proj_model=cam_info['proj_model'],
                    width=width,
                    height=height,
                ))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(image_id, info, cam_info, mono3d=True):
    """Get the 2D annotation records for a given frame.

    Args:
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """
    records = []
    cam2lidar = Pose.from_translation_rotation(
        cam_info['sensor2lidar_translation'],
        cam_info['sensor2lidar_rotation'])
    assert len(info['gt_boxes']) == len(info['gt_names']) == len(
        info['gt_boxes_2d'])
    for box, box2d, name, has_3d, is_ignore in zip(
            info['gt_boxes'],
            info['gt_boxes_2d'],
            info['gt_names'],
            info['has_3d'],
            info['is_ignore'],
    ):
        # Generate dictionary record to be included in the .json file.
        x1, y1, x2, y2 = box2d
        record = {
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'category_name': name,
            'category_id': ALL_CLASSES.index(name),
            'area': (y2 - y1) * (x2 - x1),
            'file_name': cam_info['data_path'],
            'image_id': image_id,
            'iscrowd': is_ignore,
            'has_3d': has_3d,
        }

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (record is not None) and has_3d:
            # box is (x, y, z, l, w, h, yaw) in lidar coordinates
            # convert to camera coordinates
            box = box.copy()
            box = Box3D(
                box[:3],
                box[3:6],
                Quaternion(axis=(0, 0, 1), radians=box[6]),
                frame=Lidar.VELODYNE,
            )
            box._transform_inv(cam2lidar, Camera.FRONT)

            loc = box.center.tolist()
            dim = box.size[[
                0, 2, 1
            ]].tolist()  # mmdet dimensions in cam coords are (l, h, w)
            rot = [-box.orientation.yaw_pitch_roll[0]
                   ]  # rotation about gravity axis
            record['bbox_cam3d'] = loc + dim + rot

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d,
                cam_info['cam_intrinsic'],
                with_depth=True,
                meta={
                    'distortion': cam_info['cam_distortion'],
                    'proj_model': cam_info['proj_model'],
                },
            )
            record['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if record['center2d'][2] <= 0:
                print(f"depth < 0: {record['center2d'][2]}")
                continue

        records.append(record)

    return records
