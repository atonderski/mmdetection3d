# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict
from typing import List, Optional

import numpy as np
import pyquaternion
import torch
from mmcv.utils import print_log
from zod.constants import EVALUATION_CLASSES, Anonymization
from zod.eval.detection import DetectionBox, EvalBoxes
from zod.eval.detection import nuscenes_evaluate as zod_eval

from mmdet3d.core.bbox.structures.utils import points_cam2img
from mmdet3d.core.evaluation.kitti_utils.eval import kitti_eval
from mmdet3d.datasets.nuscenes_mono_dataset import NuScenesMonoDataset
from mmdet3d.datasets.zen_dataset import flatten_dict
from ..core.bbox import CameraInstance3DBoxes
from .builder import DATASETS

BLUR = Anonymization.BLUR.value


@DATASETS.register_module()
class ZenMonoDataset(NuScenesMonoDataset):
    r"""Monocular 3D detection on NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        data_root (str): Path of dataset root.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Camera' in this class. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        eval_version (str, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool, optional): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
        version (str, optional): Dataset version. Defaults to 'v1.0-trainval'.
    """
    CLASSES = EVALUATION_CLASSES
    CLASSES_TO_KITTI = {
        'Vehicle': 'Car',
        'VulnerableVehicle': 'Cyclist',
        'Pedestrian': 'Pedestrian',
    }

    def __init__(
        self,
        data_root,
        ann_file,
        pipeline,
        load_interval=1,
        with_velocity=False,
        eval_version='zen',
        version=None,  # TODO: see if needed
        anonymization_mode=BLUR,
        use_png=False,
        **kwargs,
    ):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            load_interval=load_interval,
            with_velocity=with_velocity,
            eval_version=None,  # Dont pass it to nuscenes
            version=None,
            **kwargs,
        )
        self.eval_version = eval_version
        self.bbox_code_size = 7
        self.anonymization_mode = anonymization_mode
        self.use_png = use_png
        # Maybe change image paths depending on settings
        if self.anonymization_mode != BLUR:
            self._rename_image_paths(
                lambda x: x.replace(BLUR, self.anonymization_mode))
        if self.use_png:
            self._rename_image_paths(lambda x: x.replace('.jpg', '.png'))

    def _rename_image_paths(self, rename_func):
        """Rename image paths.

        Args:
            rename_func (function): Function to rename image paths.
        """
        for info in self.data_infos:
            info['filename'] = rename_func(info['filename'])
            info['file_name'] = rename_func(info['file_name'])

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox annotation.

        Args:
            img_info (list[dict]): Image info.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, labels,
                gt_bboxes_3d, gt_labels_3d, centers2d, depths, bboxes_ignore,
                masks, seg_map
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_bboxes_cam3d = []
        centers2d = []
        depths = []
        for i, ann in enumerate(ann_info):
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann['iscrowd'] or not ann['has_3d']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(None)
                # 3D annotations in camera coordinates
                bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(1, -1)
                # nan_mask = np.isnan(velo_cam3d[:, 0])
                # velo_cam3d[nan_mask] = [0.0, 0.0]
                # bbox_cam3d = np.concatenate([bbox_cam3d, velo_cam3d],
                #                             axis=-1)
                gt_bboxes_cam3d.append(bbox_cam3d.squeeze())
                # 2.5D annotations in camera coordinates
                center2d = ann['center2d'][:2]
                depth = ann['center2d'][2]
                centers2d.append(center2d)
                depths.append(depth)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_cam3d:
            gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)
            centers2d = np.array(centers2d, dtype=np.float32)
            depths = np.array(depths, dtype=np.float32)
        else:
            gt_bboxes_cam3d = np.zeros((0, self.bbox_code_size),
                                       dtype=np.float32)
            centers2d = np.zeros((0, 2), dtype=np.float32)
            depths = np.zeros((0), dtype=np.float32)

        gt_bboxes_cam3d = CameraInstance3DBoxes(
            gt_bboxes_cam3d,
            box_dim=gt_bboxes_cam3d.shape[-1],
            origin=(0.5, 0.5, 0.5))
        gt_labels_3d = copy.deepcopy(gt_labels)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_bboxes_3d=gt_bboxes_cam3d,
            gt_labels_3d=gt_labels_3d,
            centers2d=centers2d,
            depths=depths,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
        )

        return ann

    def evaluate(
        self,
        results,
        metric=None,
        logger=None,
        jsonfile_prefix=None,
        # result_names=["img_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        """Evaluation in Zen protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files, including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        assert len(results[0]) == 1 and 'img_bbox' in results[0], print(
            results[0])
        results = [res['img_bbox'] for res in results]

        if self.eval_version == 'kitti':
            eval_results = self._evaluate_kitti(results, logger)
        elif self.eval_version == 'zen':
            eval_results = self._evaluate_zen(results, logger)
        else:
            raise ValueError('Unsupported eval_version: {}'.format(
                self.eval_version))
        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return eval_results

    # Zen evaluation

    def _evaluate_zen(self, results, logger) -> dict:
        """Evaluate in Zen protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
        """
        det_boxes, gt_boxes = EvalBoxes(), EvalBoxes()
        for idx, (det, info) in enumerate(zip(results, self.data_infos)):
            frame_id = info['frame_id']
            det_boxes.add_boxes(frame_id, self._det_to_zen(det, frame_id))
            gt_boxes.add_boxes(frame_id, self._gt_to_zen(idx, frame_id))

        results_dict = flatten_dict(zod_eval(gt_boxes, det_boxes))
        print_log(results_dict, logger=logger)
        return results_dict

    def _det_to_zen(self, det: dict, frame_id: str) -> List[DetectionBox]:
        dets = []
        for box3d, label, score in zip(
                det['boxes_3d'].tensor.numpy(),
                det['labels_3d'].numpy(),
                det['scores_3d'].numpy(),
        ):
            dets.append(self._obj_to_zen(frame_id, box3d, label, score))
        return [det for det in dets if det is not None]

    def _gt_to_zen(self, idx: int, frame_id: str) -> List[DetectionBox]:
        anno: dict = self.get_ann_info(idx)
        gts = []
        for box3d, label in zip(anno['gt_bboxes_3d'], anno['gt_labels_3d']):
            gts.append(self._obj_to_zen(frame_id, box3d, label))
        return [gt for gt in gts if gt is not None]

    def _obj_to_zen(self,
                    frame_id: str,
                    box3d: np.ndarray,
                    label: int,
                    score: float = -1.0) -> Optional[DetectionBox]:
        # object is in lidar frame - meaning that the rotation is
        # around the y-axis. TODO: should this say camera frame?
        # TODO: check if rotation should be negative
        rot = pyquaternion.Quaternion(axis=(0, 1, 0), radians=box3d[6:])
        # TODO: perhaps it should be like this (taken from the kitti code)
        # q1 = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        # q2 = pyquaternion.Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
        # quat = q2 * q1

        # ego translation is same as translation since world is ego-centered
        class_name = self.CLASSES[int(label)]
        if class_name not in EVALUATION_CLASSES:
            return None
        box = DetectionBox(
            sample_token=frame_id,
            translation=tuple(box3d[:3]),
            size=tuple(box3d[3:6]),
            rotation=tuple(rot.elements),
            ego_translation=tuple(box3d[:3]),
            detection_name=class_name,
            detection_score=float(score),
        )
        return box

    # KITTI evaluation

    def _evaluate_kitti(self, results, logger) -> dict:
        dt_annos = [
            self.pred_to_kitti(pred, info)
            for pred, info in zip(results, self.data_infos)
        ]
        gt_annos = [self.get_kitti_anno(idx) for idx in range(len(dt_annos))]

        current_classes = tuple(self.CLASSES_TO_KITTI[cls]
                                for cls in self.CLASSES
                                if cls in self.CLASSES_TO_KITTI)
        ap_result_str, ap_dict = kitti_eval(
            gt_annos=gt_annos,
            dt_annos=dt_annos,
            current_classes=current_classes)
        print_log(ap_result_str)
        return ap_dict

    def pred_to_kitti(self, pred: dict, info: dict):
        """Convert network predictions to kitti style detections.

        Predictions consist of a dict with:
            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
        """
        kitti_dict = defaultdict(list)

        if len(pred['scores_3d']) > 0:
            boxes_3d = pred['boxes_3d']
            image_shape = boxes_3d.tensor.new_tensor(
                (info['height'], info['width'])).numpy()
            box_corners_in_image = points_cam2img(
                boxes_3d.corners,
                proj_mat=info['cam_intrinsic'],
                meta={
                    'distortion': info['cam_distortion'],
                    'proj_model': info['proj_model']
                },
            )
            minxy = torch.min(box_corners_in_image, dim=-2)[0]
            maxxy = torch.max(box_corners_in_image, dim=-2)[0]
            boxes_2d = torch.cat([minxy, maxxy], dim=-1)

            for box2d, box3d, label, score in zip(
                    boxes_2d.numpy(),
                    boxes_3d.tensor.numpy(),
                    pred['labels_3d'].numpy(),
                    pred['scores_3d'].numpy(),
            ):
                # Post-processing - clip to image boundaries, and discard
                # check box_preds_camera
                box2d[2:] = np.minimum(box2d[2:], image_shape[::-1])
                box2d[:2] = np.maximum(box2d[:2], [0, 0])
                if (box2d[0] - box2d[2]) * (box2d[1] - box2d[3]) <= 0:
                    continue
                zen_name = self.CLASSES[int(label)]
                if zen_name not in self.CLASSES_TO_KITTI:
                    continue
                kitti_name = self.CLASSES_TO_KITTI[zen_name]
                kitti_dict['name'].append(kitti_name)
                kitti_dict['truncated'].append(0.0)
                kitti_dict['occluded'].append(0)
                kitti_dict['alpha'].append(-np.arctan2(box3d[0], box3d[2]) +
                                           box3d[6])
                kitti_dict['bbox'].append(box2d)
                kitti_dict['dimensions'].append(box3d[3:6])
                kitti_dict['location'].append(box3d[:3])
                kitti_dict['rotation_y'].append(box3d[6])
                kitti_dict['score'].append(score)

        if kitti_dict:
            return {k: np.stack(v) for k, v in kitti_dict.items()}
        else:
            return _empty_kitti_dict(with_score=True)

    def get_kitti_anno(self, idx: int):
        anno: dict = self.get_ann_info(idx)
        kitti_dict = defaultdict(list)
        for box3d, box2d, label in zip(anno['gt_bboxes_3d'], anno['bboxes'],
                                       anno['gt_labels_3d']):
            box2d_xyxy = np.array([
                box2d[0],
                box2d[1],
                box2d[0] + box2d[2],
                box2d[1] + box2d[3],
            ])
            zen_name = self.CLASSES[int(label)]
            if zen_name not in self.CLASSES_TO_KITTI:
                continue
            kitti_name = self.CLASSES_TO_KITTI[zen_name]
            alpha = np.arctan(-np.arctan2(box3d[0], box3d[2]) + box3d[6])
            kitti_dict['bbox'].append(box2d_xyxy)
            kitti_dict['location'].append(box3d[:3])
            kitti_dict['dimensions'].append(box3d[3:6])
            kitti_dict['rotation_y'].append(box3d[6])
            kitti_dict['alpha'].append(alpha)
            kitti_dict['name'].append(kitti_name)
            kitti_dict['truncated'].append(0.0)
            kitti_dict['occluded'].append(0)

        if kitti_dict:
            return {k: np.stack(v) for k, v in kitti_dict.items()}
        else:
            return _empty_kitti_dict(with_score=False)


def _empty_kitti_dict(with_score: bool):
    kitti_dict = {
        'name': np.array([]),
        'truncated': np.array([]),
        'occluded': np.array([]),
        'alpha': np.array([]),
        'bbox': np.zeros([0, 4]),
        'dimensions': np.zeros([0, 3]),
        'location': np.zeros([0, 3]),
        'rotation_y': np.array([]),
    }
    if with_score:
        kitti_dict['score'] = np.array([])
    return kitti_dict
