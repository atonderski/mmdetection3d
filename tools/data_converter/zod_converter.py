# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

from mmdet3d.core.bbox import points_cam2img
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets.zen_dataset import ZenDataset
from mmdet3d.zod_tmp import (CameraCalibration, EgoPose, FrameInformation,
                             LidarCalibration, OXTSData, SensorFrame,
                             ZenseactOpenDataset)

nus_categories = (
    "car",
    "bicycle",
    "motorcycle",
    "pedestrian",
)


def create_nuscenes_infos(
    root_path, info_prefix, version="mini", max_sweeps=10, use_blur=True
):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
        use_blur (bool, optional): Whether to use blurred images.
            Default: True.
    """
    if version != "mini":
        raise NotImplemented
    train_scenes = ["00000" + str(i) for i in range(9)]
    val_scenes = ["000009"]
    zod = ZenseactOpenDataset(root_path)
    train_infos = _fill_infos(
        zod, train_scenes, max_sweeps=max_sweeps, use_blur=use_blur
    )
    val_infos = _fill_infos(zod, val_scenes, max_sweeps=max_sweeps, use_blur=use_blur)

    metadata = dict(version=version)
    print("train sample: {}, val sample: {}".format(len(train_infos), len(val_infos)))
    data = dict(infos=train_infos, metadata=metadata)
    info_path = osp.join(root_path, "{}_infos_train.pkl".format(info_prefix))
    mmcv.dump(data, info_path)
    data["infos"] = val_infos
    info_val_path = osp.join(root_path, "{}_infos_val.pkl".format(info_prefix))
    mmcv.dump(data, info_val_path)


def _fill_infos(
    zod: ZenseactOpenDataset, frames: List[str], max_sweeps=10, use_blur=True
):
    """Generate the train/val infos from the raw data.

    Args:
        zod (:obj:`ZenseactOpenDataset`): Dataset class in the ZenseactOpenDataset.
        frames (list[str]): IDs of the training/validation frames.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        list[dict]: Information of training/validation set that will be saved to the info file.
    """
    infos = []
    for frame_id in mmcv.track_iter_progress(frames):
        frame_info = zod.get_frame_info(frame_id)

        lidar_path = frame_info.lidar_frame[zod.VELODYNE].filepath
        mmcv.check_file_exist(lidar_path)
        calib = zod.read_calibration(frame_info.calibration_path)
        oxts = zod.read_oxts(frame_info.oxts_path)
        core_lidar_calib = calib.lidars[zod.VELODYNE]
        core_ego_pose = oxts.get_ego_pose(frame_info.timestamp)

        info = {
            "lidar_path": lidar_path,
            "sweeps": [],
            "cams": dict(),
            "lidar2ego_translation": core_lidar_calib.translation,
            "lidar2ego_rotation": core_lidar_calib.rotation,
            "ego2global_translation": core_ego_pose.translation,
            "ego2global_rotation": core_ego_pose.rotation,
            "timestamp": frame_info.timestamp,
        }

        cameras = [
            zod.CAMERA_FRONT,
        ]
        # suffix = "_blur" if use_blur else "_dnat"
        suffix = ""
        for cam in cameras:
            cam_calib = calib.cameras[cam]
            cam_info = obtain_sensor2lidar(
                frame_info.camera_frame[cam + suffix],
                calib.cameras[cam],
                core_ego_pose,
                core_lidar_calib,
                oxts,
                cam,
            )
            cam_info.update(cam_intrinsic=cam_calib.intrinsics)
            info["cams"].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        info["sweeps"] = [
            obtain_sensor2lidar(
                frame, core_lidar_calib, core_ego_pose, core_lidar_calib, oxts, "lidar"
            )
            for frame in frame_info.previous_lidar_frames[zod.VELODYNE][:max_sweeps]
        ]

        # obtain annotation
        annos = zod.get_dynamic_objects(frame_id)
        locs = np.array([b.pos for b in annos]).reshape(-1, 3)
        dims = np.array([b.lwh for b in annos]).reshape(-1, 3)
        rots = np.array([b.rot.yaw_pitch_roll[0] for b in annos]).reshape(-1, 1)
        gt_boxes = np.concatenate([locs, dims, rots], axis=1)
        # valid_flag = np.array(
        #     [
        #         (anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
        #         for anno in annotations
        #     ],
        #     dtype=bool,
        # ).reshape(-1)

        names = [b.name for b in annos]
        for i in range(len(names)):
            if names[i] in ZenDataset.NameMapping:
                names[i] = ZenDataset.NameMapping[names[i]]
        names = np.array(names)

        assert len(gt_boxes) == len(annos), f"{len(gt_boxes)}, {len(annos)}"
        info["gt_boxes"] = gt_boxes
        info["gt_names"] = names
        info["gt_boxes_2d"] = np.array([b.box2d for b in annos]).reshape(-1, 4)
        # info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
        # info["valid_flag"] = valid_flag

        infos.append(info)

    return infos


def obtain_sensor2lidar(
    sensor_frame: SensorFrame,
    sensor_calib: Union[CameraCalibration, LidarCalibration],
    core_ego_pose: EgoPose,
    core_lidar_calib: LidarCalibration,
    oxts: OXTSData,
    sensor_type: str,
) -> dict:
    """Obtain the info with RT matric from general sensor to core (top) LiDAR."""
    ego_pose = oxts.get_ego_pose(sensor_frame.timestamp)
    sweep = {
        "data_path": sensor_frame.filepath,
        "type": sensor_type,
        "sensor2ego_translation": sensor_calib.translation,
        "sensor2ego_rotation": sensor_calib.rotation,
        "ego2global_translation": ego_pose.translation,
        "ego2global_rotation": ego_pose.rotation,
        "timestamp": sensor_frame.timestamp,
    }
    # transforms for sweep frame
    l2e_t_s = sensor_calib.translation
    e2g_t_s = ego_pose.translation
    l2e_r_s_mat = sensor_calib.rotation_matrix
    e2g_r_s_mat = ego_pose.rotation_matrix

    # transforms for core frame
    e2g_r_mat = core_ego_pose.rotation_matrix
    e2g_t = core_ego_pose.translation
    l2e_r_mat = core_lidar_calib.rotation_matrix
    l2e_t = core_lidar_calib.translation

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep
