import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Union

import h5py
import numpy as np
from dataclass_wizard import JSONSerializable
from pyquaternion import Quaternion
from tqdm import tqdm

### FROM CONSTANTS ###
# sensor data
LIDAR_VELODYNE = "lidar_velodyne"
CAMERA_FRONT = "camera_front"
BLUR = "blur"
DNAT = "dnat"
CAMERA_FRONT_BLUR = f"{CAMERA_FRONT}_{BLUR}"
CAMERA_FRONT_DNAT = f"{CAMERA_FRONT}_{DNAT}"
CALIBRATION = "calibration"
OXTS = "oxts"

# dataset paths
SINGLE_FRAMES = "single_frames"

### END FROM CONSTANTS ###


@dataclass
class Pose:
    """A general class describing some pose."""
    transform: np.ndarray

    @property
    def translation(self) -> np.ndarray:
        """Return the translation (array)."""
        return self.transform[:3, 3]

    @property
    def rotation(self) -> Quaternion:
        """Return the rotation as a quaternion."""
        return Quaternion(matrix=self.rotation_matrix)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Return the rotation matrix."""
        return self.transform[:3, :3]
    
    @property
    def inverse(self) -> "Pose":
        """Return the inverse of the pose."""
        return Pose(np.linalg.inv(self.transform))

@dataclass
class OXTSData:
    acceleration_x: np.ndarray
    acceleration_y: np.ndarray
    acceleration_z: np.ndarray
    angular_rate_x: np.ndarray
    angular_rate_y: np.ndarray
    angular_rate_z: np.ndarray
    ecef_x: np.ndarray
    ecef_y: np.ndarray
    ecef_z: np.ndarray
    heading: np.ndarray
    leap_seconds: np.ndarray
    pitch: np.ndarray
    pos_alt: np.ndarray
    pos_lat: np.ndarray
    pos_lon: np.ndarray
    roll: np.ndarray
    std_dev_pos_east: np.ndarray
    std_dev_pos_north: np.ndarray
    time_gps: np.ndarray
    traveled: np.ndarray
    vel_down: np.ndarray
    vel_forward: np.ndarray
    vel_lateral: np.ndarray

    def get_ego_pose(self, timestamp: datetime) -> Pose:
        return Pose(np.eye(4, 4))

    @classmethod
    def from_hdf5(cls, file: h5py.Group) -> "OXTSData":
        return cls(
            acceleration_x=np.array(file["accelerationX"]),
            acceleration_y=np.array(file["accelerationY"]),
            acceleration_z=np.array(file["accelerationZ"]),
            angular_rate_x=np.array(file["angularRateX"]),
            angular_rate_y=np.array(file["angularRateY"]),
            angular_rate_z=np.array(file["angularRateZ"]),
            ecef_x=np.array(file["ecef_x"]),
            ecef_y=np.array(file["ecef_y"]),
            ecef_z=np.array(file["ecef_z"]),
            heading=np.array(file["heading"]),
            leap_seconds=np.array(file["leapSeconds"]),
            pitch=np.array(file["pitch"]),
            pos_alt=np.array(file["posAlt"]),
            pos_lat=np.array(file["posLat"]),
            pos_lon=np.array(file["posLon"]),
            roll=np.array(file["roll"]),
            std_dev_pos_east=np.array(file["stdDevPosEast"]),
            std_dev_pos_north=np.array(file["stdDevPosNorth"]),
            time_gps=np.array(file["time_gps"]),
            traveled=np.array(file["traveled"]),
            vel_down=np.array(file["velDown"]),
            vel_forward=np.array(file["velForward"]),
            vel_lateral=np.array(file["velLateral"]),
        )


@dataclass
class LidarCalibration:
    extrinsics: Pose  # lidar pose in the ego frame



@dataclass
class CameraCalibration:
    extrinsics: Pose  # 4x4 matrix describing the camera pose in the ego frame
    intrinsics: np.ndarray  # 3x3 matrix
    distortion: np.ndarray  # 4 vector
    image_dimensions: np.ndarray  # width, height
    field_of_view: np.ndarray  # vertical, horizontal (degrees)

@dataclass
class Calibration:
    lidars: Dict[str, LidarCalibration]
    cameras: Dict[str, CameraCalibration]

    @classmethod
    def from_dict(cls, calib_dict: Dict[str, Any]):
        lidars = {
            LIDAR_VELODYNE: LidarCalibration(
                extrinsics=Pose(np.array(calib_dict["FC"]["lidar_extrinsics"]))
            ),
        }
        cameras = {
            CAMERA_FRONT: CameraCalibration(
                extrinsics=Pose(np.array(calib_dict["FC"]["extrinsics"])),
                intrinsics=np.array(calib_dict["FC"]["intrinsics"]),
                distortion=np.array(calib_dict["FC"]["distortion"]),
                image_dimensions=np.array(calib_dict["FC"]["image_dimensions"]),
                field_of_view=np.array(calib_dict["FC"]["field_of_view"]),
            ),
        }
        return cls(lidars=lidars, cameras=cameras)


@dataclass
class SensorFrame(JSONSerializable):
    """Class to store sensor information."""

    filepath: str
    timestamp: datetime


@dataclass
class FrameInformation(JSONSerializable):
    """Class to store frame information."""

    frame_id: str
    timestamp: datetime

    traffic_sign_annotation_path: Union[str, None]
    ego_road_annotation_path: Union[str, None]
    dynamic_objects_annotation_path: str
    static_objects_annotation_path: str
    lane_markings_annotation_path: str
    road_condition_annotation_path: str

    lidar_frame: Dict[str, SensorFrame]
    previous_lidar_frames: Dict[str, List[SensorFrame]]
    future_lidar_frames: Dict[str, List[SensorFrame]]
    camera_frame: Dict[str, SensorFrame]

    oxts_path: str
    calibration_path: str

    metadata_path: str

    def convert_paths_to_absolute(self, root_path: str):
        self.traffic_sign_annotation_path = (
            os.path.join(root_path, self.traffic_sign_annotation_path)
            if self.traffic_sign_annotation_path
            else None
        )
        self.ego_road_annotation_path = (
            os.path.join(root_path, self.ego_road_annotation_path)
            if self.ego_road_annotation_path
            else None
        )
        self.dynamic_objects_annotation_path = os.path.join(
            root_path, self.dynamic_objects_annotation_path
        )
        self.static_objects_annotation_path = os.path.join(
            root_path, self.static_objects_annotation_path
        )
        self.lane_markings_annotation_path = os.path.join(
            root_path, self.lane_markings_annotation_path
        )
        self.road_condition_annotation_path = os.path.join(
            root_path, self.road_condition_annotation_path
        )
        self.oxts_path = os.path.join(root_path, self.oxts_path)
        self.calibration_path = os.path.join(root_path, self.calibration_path)
        self.metadata_path = os.path.join(root_path, self.metadata_path)
        for sensor_frame in self.lidar_frame.values():
            sensor_frame.filepath = os.path.join(root_path, sensor_frame.filepath)
        for lidar_frames in self.previous_lidar_frames.values():
            for sensor_frame in lidar_frames:
                sensor_frame.filepath = os.path.join(root_path, sensor_frame.filepath)
        for lidar_frames in self.future_lidar_frames.values():
            for sensor_frame in lidar_frames:
                sensor_frame.filepath = os.path.join(root_path, sensor_frame.filepath)
        for sensor_frame in self.camera_frame.values():
            sensor_frame.filepath = os.path.join(root_path, sensor_frame.filepath)


@dataclass
class DynamicObject:
    """Class to store dynamic object information."""

    pos: np.ndarray
    lwh: np.ndarray
    rot: Quaternion
    box2d: np.ndarray  # 2d bounding box in xyxy format
    name: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create a DynamicObject from a dictionary.

        Example dict:
        {
            "geometry": {
                "coordinates": [[1610.16, 1102.42], [1615.43, 1112.06], [1610.85, 1132.92], [1604.43, 1112.06]],
                "type": "MultiPoint"
            },
            "properties": {
                "annotation_uuid": "04568424-8984-43df-8a2c-c6a9b11e71b6",
                "class": "Pedestrian",
                "emergency": false,
                "is_pulling_or_pushing": "Nothing",
                "location_3d": {"coordinates": [-20.129077433649584, 99.19406998419957, -2.009195389523851], "type": "Point"},
                "object_type": "Pedestrian",
                "occlusion_ratio": "VeryHeavy",
                "orientation_3d_qw": 0.7515051536212287,
                "orientation_3d_qx": 0.0,
                "orientation_3d_qy": 0.0,
                "orientation_3d_qz": 0.6597272194481091,
                "relative_position": "NotOnEgoRoad",
                "size_3d_height": 1.6171658742481,
                "size_3d_length": 0.591543435193717,
                "size_3d_width": 0.7233595586832583,
                "unclear": false
            }
        }

        """
        if data["properties"]["unclear"] or "location_3d" not in data["properties"]:
            return None
        box2d = np.array(data["geometry"]["coordinates"])
        box2d = np.array(
            [box2d[:, 0].min(), box2d[:, 1].min(), box2d[:, 0].max(), box2d[:, 1].max()]
        )
        pos = np.array(data["properties"]["location_3d"]["coordinates"])
        lwh = np.array(
            [
                data["properties"]["size_3d_length"],
                data["properties"]["size_3d_width"],
                data["properties"]["size_3d_height"],
            ]
        )
        rot = Quaternion(
            w=data["properties"]["orientation_3d_qw"],
            x=data["properties"]["orientation_3d_qx"],
            y=data["properties"]["orientation_3d_qy"],
            z=data["properties"]["orientation_3d_qz"],
        )
        return cls(
            pos=pos, lwh=lwh, rot=rot, box2d=box2d, name=data["properties"]["class"]
        )


class ZodFrames(object):

    SPLITS = {
        "mini": {
            "train": [str(id_).zfill(6) for id_ in range(9)],
            "val": [str(id_).zfill(6) for id_ in [9]],
        },
        "full": {
            "train": [str(id_).zfill(6) for id_ in range(90000)],
            "val": [str(id_).zfill(6) for id_ in range(90000, 100000)],
        },
    }

    def __init__(self, dataset_root: str, version: str):
        self._dataset_root = dataset_root
        self._version = version
        assert version in self.SPLITS, f"Unknown version: {version}"
        self._frames = self._load_frames()

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, frame_id: str) -> FrameInformation:
        """Get frame by id, which is zero-padded frame number."""
        return self._frames[frame_id]

    # @property
    # def frames(self) -> FrameInformation:
    #     """Get all frames."""
    #     yield (FrameInformation.from_dict(frame) for frame in self._frames.values())

    def _load_frames(self) -> Dict[str, FrameInformation]:
        """Load frames for the given version."""
        frames = {}
        all_ids = (
            self.SPLITS[self._version]["train"] + self.SPLITS[self._version]["val"]
        )
        for frame_id in tqdm(all_ids, desc="Loading frame information"):
            frame_path = os.path.join(
                self._dataset_root, SINGLE_FRAMES, frame_id, "frame_info.json"
            )
            with open(frame_path, "r") as frames_file:
                frame = json.load(frames_file)
                frames[frame_id] = FrameInformation.from_dict(frame)
                frames[frame_id].convert_paths_to_absolute(self._dataset_root)
        return frames

    def get_split(self, split: str) -> List[str]:
        """Get split by name (e.g. train / val)."""
        assert split in self.SPLITS[self._version], f"Unknown split: {split}"
        return self.SPLITS[self._version][split]

    def read_calibration(self, frame_id: str) -> Calibration:
        with open(self._frames[frame_id].calibration_path) as f:
            calib = json.load(f)
        return Calibration.from_dict(calib)

    def read_oxts(self, frame_id: str) -> OXTSData:
        """Read OXTS files from hd5 format."""
        with h5py.File(self._frames[frame_id].oxts_path, "r") as f:
            data = OXTSData.from_hdf5(f)
        return data

    def read_dynamic_objects(self, frame_id: str) -> List[DynamicObject]:
        with open(self._frames[frame_id].dynamic_objects_annotation_path) as f:
            dynamic_objects = json.load(f)
        objs = (DynamicObject.from_dict(anno) for anno in dynamic_objects)
        return [obj for obj in objs if obj is not None]
