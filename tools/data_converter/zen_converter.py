# Copyright (c) OpenMMLab. All rights reserved.
import os
from os import path as osp
from typing import List, Union

import mmcv
import numpy as np
from pyquaternion import Quaternion

from agp.zod.frames.zod_frames import ZodFrames
from agp.zod.utils.constants import CAMERA_FRONT, LIDAR_VELODYNE
from agp.zod.utils.dataclasses import (CameraCalibration, LidarCalibration,
                                       OXTSData, Pose, SensorFrame)
from agp.zod.utils.objects import Box3D
from mmdet3d.core.bbox.structures.utils import points_cam2img
from mmdet3d.datasets import ZenDataset


def create_zen_infos(
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
    zod = ZodFrames(root_path, version)
    train_scenes = zod.get_split("train")
    val_scenes = zod.get_split("val")
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


def _fill_infos(zod: ZodFrames, frames: List[str], max_sweeps=10, use_blur=True):
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
        frame_info = zod[frame_id]

        lidar_path = frame_info.lidar_frame[LIDAR_VELODYNE].filepath
        mmcv.check_file_exist(lidar_path)
        calib = zod.read_calibration(frame_id)
        oxts = zod.read_oxts(frame_id)
        lidar_calib = calib.lidars[LIDAR_VELODYNE]
        core_lidar2ego = lidar_calib.extrinsics
        core_ego_pose = oxts.get_ego_pose(frame_info.timestamp)

        info = {
            "lidar_path": lidar_path,
            "frame_id": frame_id,
            "sweeps": [],
            "cams": dict(),
            "lidar2ego_translation": core_lidar2ego.translation,
            "lidar2ego_rotation": core_lidar2ego.rotation,
            "ego2global_translation": core_ego_pose.translation,
            "ego2global_rotation": core_ego_pose.rotation,
            "timestamp": frame_info.timestamp.timestamp(),
        }

        cameras = [
            CAMERA_FRONT,
        ]
        suffix = "_blur" if use_blur else "_dnat"
        for cam in cameras:
            cam_calib = calib.cameras[cam]
            cam_info = obtain_sensor2lidar(
                frame_info.camera_frame[cam + suffix],
                calib.cameras[cam],
                core_ego_pose,
                core_lidar2ego,
                oxts,
                cam,
            )
            # TEMPORARY HACK - REMOVE THIS
            cam_info["data_path"] = cam_info["data_path"].replace(suffix, "_original")
            cam_info.update(cam_intrinsic=cam_calib.intrinsics, cam_distortion=cam_calib.distortion, proj_model="kannala")
            info["cams"].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        info["sweeps"] = [
            obtain_sensor2lidar(
                frame, lidar_calib, core_ego_pose, core_lidar2ego, oxts, "lidar"
            )
            for frame in frame_info.previous_lidar_frames[LIDAR_VELODYNE][:max_sweeps]
        ]

        # obtain annotation
        annos = zod.read_dynamic_objects(frame_id)
        locs = np.array([b.box3d.center for b in annos]).reshape(-1, 3)
        dims = np.array([b.box3d.size for b in annos]).reshape(-1, 3)
        rots = np.array([b.box3d.orientation.yaw_pitch_roll[0] for b in annos]).reshape(-1, 1)
        gt_boxes = np.concatenate([locs, dims, rots], axis=1)
        # valid_flag = np.array(
        #     [
        #         (anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
        #         for anno in annotations
        #     ],
        #     dtype=bool,
        # ).reshape(-1)
        # TEMPORARY: valid always true
        valid_flag = np.ones(gt_boxes.shape[0], dtype=bool)

        names = [b.name for b in annos]
        for i in range(len(names)):
            if names[i] in ZenDataset.NameMapping:
                names[i] = ZenDataset.NameMapping[names[i]]
            else:
                print(names[i])
        names = np.array(names)

        assert len(gt_boxes) == len(annos), f"{len(gt_boxes)}, {len(annos)}"
        info["gt_boxes"] = gt_boxes
        info["gt_names"] = names
        info["gt_boxes_2d"] = np.array([b.box2d for b in annos]).reshape(-1, 4)
        # info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
        info["valid_flag"] = valid_flag

        infos.append(info)

    return infos


def obtain_sensor2lidar(
    sensor_frame: SensorFrame,
    sensor_calib: Union[CameraCalibration, LidarCalibration],
    core_ego_pose: Pose,
    core_lidar2ego: Pose,
    oxts: OXTSData,
    sensor_type: str,
) -> dict:
    """Obtain the info with RT matric from general sensor to core (top) LiDAR."""
    ego_pose = oxts.get_ego_pose(sensor_frame.timestamp)
    sensor2ego = sensor_calib.extrinsics
    sweep = {
        "data_path": sensor_frame.filepath,
        "type": sensor_type,
        "sensor2ego_translation": sensor2ego.translation,
        "sensor2ego_rotation": sensor2ego.rotation,
        "ego2global_translation": ego_pose.translation,
        "ego2global_rotation": ego_pose.rotation,
        "timestamp": sensor_frame.timestamp.timestamp(),
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
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (s2e_t @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    # NOTE: this is a bit strange, but sensor2lidar_rotation is a rotation matrix (other rotations are quaternions)
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
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
    # get bbox annotations for camera
    camera_types = [CAMERA_FRONT]
    infos = mmcv.load(info_path)["infos"]
    zod = ZodFrames(root_path, version=version)
    # info_2d_list = []
    cat2Ids = [
        dict(id=ZenDataset.CLASSES.index(cat_name), name=cat_name)
        for cat_name in ZenDataset.CLASSES
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(infos):
        for cam in camera_types:
            image_id = f"{cam}_{info['frame_id']}"
            cam_info = info["cams"][cam]
            coco_infos = get_2d_boxes(
                image_id,
                info,
                cam_info,
                mono3d=mono3d,
            )
            # alternative if this is slow:
            width, height = zod.read_calibration(info["frame_id"]).cameras[cam].image_dimensions
            # (height, width, _) = mmcv.imread(cam_info["data_path"]).shape
            coco_2d_dict["images"].append(
                dict(
                    file_name=os.path.relpath(cam_info["data_path"], root_path),
                    id=image_id,
                    frame_id=info["frame_id"],
                    cam2ego_rotation=cam_info["sensor2ego_rotation"].elements,
                    cam2ego_translation=cam_info["sensor2ego_translation"],
                    ego2global_rotation=info["ego2global_rotation"].elements,
                    ego2global_translation=info["ego2global_translation"],
                    cam_intrinsic=cam_info["cam_intrinsic"],
                    cam_distortion=cam_info["cam_distortion"],
                    proj_model=cam_info["proj_model"],
                    width=width,
                    height=height,
                )
            )
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info["segmentation"] = []
                coco_info["id"] = coco_ann_id
                coco_2d_dict["annotations"].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f"{info_path[:-4]}_mono3d"
    else:
        json_prefix = f"{info_path[:-4]}"
    mmcv.dump(coco_2d_dict, f"{json_prefix}.coco.json")


def get_2d_boxes(image_id, info, cam_info, mono3d=True):
    """Get the 2D annotation records for a given frame.

    Args:
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """
    records = []
    cam2lidar = Pose.from_translation_rotation(cam_info["sensor2lidar_translation"], cam_info["sensor2lidar_rotation"])
    assert len(info["gt_boxes"]) == len(info["gt_names"]) == len(info["gt_boxes_2d"])
    for box, box2d, name in zip(info["gt_boxes"], info["gt_boxes_2d"], info["gt_names"]):
        # Generate dictionary record to be included in the .json file.
        x1, y1, x2, y2 = box2d
        record = {
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "category_name": name,
            "category_id": ZenDataset.CLASSES.index(name),
            "area": (y2 - y1) * (x2 - x1),
            "file_name": cam_info["data_path"],
            "image_id": image_id,
            "iscrowd": 0,
        }

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (record is not None):
            # box is (x, y, z, l, w, h, yaw) in lidar coordinates
            # convert to camera coordinates
            box = box.copy()
            box = Box3D(box[:3], box[3:6], Quaternion(axis=(0, 0, 1), radians=box[6]), frame=LIDAR_VELODYNE)
            box._transform_inv(cam2lidar, CAMERA_FRONT)

            loc = box.center.tolist()
            dim = box.size[[0,2,1]].tolist()  # mmdet dimensions in cam coords are (l, h, w)
            rot = [-box.orientation.yaw_pitch_roll[0]]  # rotation about gravity axis
            record["bbox_cam3d"] = loc + dim + rot

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, cam_info["cam_intrinsic"], with_depth=True, dist_coeffs=cam_info['cam_distortion'], proj_model=cam_info['proj_model']
            )
            record["center2d"] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if record["center2d"][2] <= 0:
                print(f"depth < 0: {record['center2d'][2]}")
                continue

        records.append(record)

    return records
