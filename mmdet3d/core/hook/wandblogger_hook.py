# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict

import mmcv
import numpy as np
import torch
from mmcv.runner import HOOKS

from mmdet3d.core.bbox.structures.box_3d_mode import Box3DMode
from mmdet3d.core.bbox.structures.utils import points_cam2img
from mmdet.core import MMDetWandbHook


@HOOKS.register_module()
class MMDet3DWandbHook(MMDetWandbHook):
    """Enhanced Wandb logger hook for MMDetection3D with 3d visualizations.

    Comparing with the :cls:`mmdetection.runner.MMDetWandbHook`, this hook can
    handle 3d object predictions and ground truths (instead of 2D only). These
    predictions can be visualized interactively in 3D in the W&B dashboard,
    and/or projected to 2D images.

    For more details, please refer to the parent documentation:
    :cls:`mmdetection.runner.MMDetWandbHook`
    """

    def __init__(self,
                 visualize_3d=False,
                 visualize_img=False,
                 max_points=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._visualize_3d = visualize_3d
        self._visualize_img = visualize_img
        self._max_points = max_points
        self._lidar_loader = None

        # Cached objs and infos for image visualizations
        self._img_gt_objs = {}
        self._val_img_infos = {}

    def _add_ground_truth(self, runner):
        # Randomly select the samples to be logged (with consistent seed)
        self.eval_idxs = np.arange(len(self.val_dataset))
        np.random.seed(42)
        np.random.shuffle(self.eval_idxs)
        self.eval_idxs = self.eval_idxs[:self.num_eval_images]

        # Set up the class set in W&B
        CLASSES = self.val_dataset.CLASSES
        self.class_id_to_label = {id: name for id, name in enumerate(CLASSES)}
        self.class_id_to_label[-1] = 'Ignore'
        self.class_set = self.wandb.Classes([{
            'id': id,
            'name': name
        } for id, name in self.class_id_to_label.items()])

        if self._visualize_img:
            self._add_img_ground_truth(runner)
        if self._visualize_3d is not None:
            # Note! W&B does not support 3D de-duplication
            # so we only set up the loader here
            self._setup_lidar_loader(runner)

    def _get_pred_boxes(self, results):
        """Get the predicted bounding boxes from the results."""
        boxes_3d = results['boxes_3d']
        labels = results['labels_3d']
        scores = results['scores_3d']
        # Remove bounding boxes and masks with score lower than threshold.
        if self.bbox_score_thr > 0:
            assert boxes_3d is not None
            inds = scores > self.bbox_score_thr
            boxes_3d = boxes_3d[inds]
            labels = labels[inds]
            scores = scores[inds]
        return boxes_3d, labels, scores

    def _log_predictions(self, results):
        for ndx, eval_image_index in enumerate(self.eval_idxs):
            if self._visualize_img:
                # Reference the data artifact for efficient de-duplication
                table_idxs = self.data_table_ref.get_index()
                assert len(table_idxs) == len(self.eval_idxs)
                results_img = results[eval_image_index]
                # Prioritize img detections over pts detections if we
                # are using the nested format
                if 'img_bbox' in results_img:
                    results_img = results_img['img_bbox']
                elif 'pts_bbox' in results_img:
                    results_img = results_img['pts_bbox']
                preds = self._get_pred_boxes(results_img)
                data_ref = self.data_table_ref.data[ndx]
                self._add_img_predictions(preds, data_ref, eval_image_index)
            if self._visualize_3d is not None:
                # 3D vis does not support de-duplication yet:
                # https://github.com/wandb/wandb/issues/4989
                results_3d = results[eval_image_index]
                if 'pts_bbox' in results_3d:
                    results_3d = results_3d['pts_bbox']
                elif 'img_bbox' in results_3d:
                    results_3d = results_3d['img_bbox']
                preds = self._get_pred_boxes(results_3d)
                self._add_3d_predictions(preds, eval_image_index)

    # 3D Visualizations #

    def _setup_lidar_loader(self, runner):
        # Get lidar loading pipeline
        from mmdet3d.datasets.pipelines.loading import (LoadPointsFromFile,
                                                        ZodLoadPointsFromFile)
        for t in self.val_dataset.pipeline.transforms:
            if isinstance(t, (LoadPointsFromFile, ZodLoadPointsFromFile)):
                self._lidar_loader = t
        if self._lidar_loader is None:
            runner.logger.warning(
                'LoadPointsFromFile is required to add points to 3d '
                'visualization. Only bounding boxes will be shown now.')
        return self._lidar_loader

    def _add_3d_predictions(self, preds, idx):
        content: Dict[str, Any] = {'type': 'lidar/beta'}
        if self._lidar_loader:
            data_info = self.val_dataset.get_data_info(idx)
            # Load point cloud
            points = self._lidar_loader(data_info)['points']
            points = points.tensor[:, :3].numpy()
            if self._max_points and points.shape[0] > self._max_points:
                # Randomly subsample points
                points = points[np.random.choice(
                    len(points), self._max_points, replace=False)]
            content['points'] = points
        else:
            # Need to add dummy points for W&B to render properly
            content['points'] = np.array([[-50, -50, -5], [50, 50, 5]],
                                         dtype=np.float32)
        # Process detections and ground truths
        boxes_3d, labels, scores = preds
        pred_wandb_boxes = self._get_wandb_bboxes_3d(boxes_3d, labels, scores)
        anno = self.val_dataset.get_ann_info(idx)
        gt_wandb_boxes = self._get_wandb_bboxes_3d(
            anno['gt_bboxes_3d'],
            anno['gt_labels_3d'],
        )
        content['boxes'] = np.array(pred_wandb_boxes + gt_wandb_boxes)
        self.wandb.log({'predictions_3d': self.wandb.Object3D(content)})

    def _get_wandb_bboxes_3d(self, boxes_3d, labels, scores=None):
        """Get list of structured dict for logging bounding boxes to W&B.

        Args:
            boxes_3d (list): List of 3d bounding boxes.
            labels (int): List of label ids.
            scores (optional[list]): List of scores.

        Returns:
            numpy array of bounding boxes to be logged.
        """
        # TODO: should we convert camera boxes to lidar here?
        box_data = []
        for i, (label, corners) in enumerate(zip(labels, boxes_3d.corners)):
            label = int(label)
            class_name = str(self.class_id_to_label[label])
            if scores is not None:
                score = float(scores[i])
                box_data.append({
                    'corners': corners.tolist(),
                    'label': f'{class_name}',
                    'color': (255 * score, 0, 0),  # red for pred
                })
            else:
                box_data.append({
                    'corners': corners.tolist(),
                    'label': f'{class_name}-gt',
                    'color': (0, 255, 0),  # green for gt
                })
        return box_data

    # Image Visualizations #

    def _extract_image_info(self, idx):
        """Get image info from dataset and cache it."""
        info = {}
        from mmdet3d.datasets import Custom3DDataset
        if isinstance(self.val_dataset, Custom3DDataset):
            data_info = self.val_dataset.get_data_info(idx)
            assert isinstance(data_info, dict)
            for field in [
                    'cam2img', 'distortion', 'proj_model', 'lidar2cam',
                    'lidar2img'
            ]:
                if field in data_info:
                    info[field] = data_info[field]
        else:
            # Assume that this is a mono3d dataset
            data_info = self.val_dataset.data_infos[idx]
            info = {
                'cam2img': data_info['cam_intrinsic'],
                'distortion': data_info['cam_distortion'],
                'proj_model': data_info['proj_model'],
            }
        self._val_img_infos[idx] = info
        return info

    def _add_img_ground_truth(self, runner):
        # Get image loader
        from mmdet3d.datasets.pipelines.loading import (
            LoadImageFromFile, LoadImageFromFileMono3D,
            LoadMultiViewImageFromFiles)
        img_loader = None
        for t in self.val_dataset.pipeline.transforms:
            if isinstance(t, (
                    LoadImageFromFileMono3D,
                    LoadMultiViewImageFromFiles,
                    LoadImageFromFile,
            )):
                img_loader = t
        if img_loader is None:
            self.log_evaluation = False
            runner.logger.warning(
                'Some image loader is required to add images '
                'to W&B Tables.')
            return

        for idx in self.eval_idxs:
            img_info = self._extract_image_info(idx)
            # Load the image
            if isinstance(img_loader, LoadImageFromFileMono3D):
                image = img_loader(
                    dict(
                        img_info=self.val_dataset.data_infos[idx],
                        img_prefix=self.val_dataset.img_prefix,
                    ))['img']
            else:
                image = img_loader(self.val_dataset.get_data_info(idx))['img']
                if len(image.shape) > 3:
                    # If multi-view, we only pick the first camera
                    image = image[0]
                    for field in list(img_info.keys()):
                        img_info[field] = img_info[field][0]
            image = mmcv.bgr2rgb(image)

            # Parse ground truth boxes
            data_ann = self.val_dataset.get_ann_info(idx)
            wandb_boxes = self._get_wandb_bboxes_img(
                img_info,
                data_ann['gt_bboxes_3d'],
                data_ann['gt_labels_3d'],
            )
            self._img_gt_objs[idx] = wandb_boxes.copy()

            # Log a row to the data table.
            self.data_table.add_data(
                f'img_{idx}',
                self.wandb.Image(
                    image, boxes=wandb_boxes, classes=self.class_set))

    def _add_img_predictions(self, preds, data_ref, eval_image_index):
        img_info = self._val_img_infos[eval_image_index]
        boxes_3d, labels, scores = preds
        pred_boxes = self._get_wandb_bboxes_img(
            img_info, boxes_3d, labels, scores, log_gt=False)
        # TODO: maybe stop logging predictions as an artifact
        self.eval_table.add_data(
            data_ref[0], data_ref[1],
            self.wandb.Image(
                data_ref[1], boxes=pred_boxes, classes=self.class_set))
        gts = self._img_gt_objs[eval_image_index]
        both_boxes = {**gts, **pred_boxes}
        self.wandb.log({
            'img_predictions':
            self.wandb.Image(
                data_ref[1], boxes=both_boxes, classes=self.class_set)
        })

    def _get_wandb_bboxes_img(self,
                              img_info,
                              boxes_3d,
                              labels,
                              scores=None,
                              log_gt=True):
        """Get list of structured dict for logging bounding boxes to W&B.

        Args:
            img_info (dict): Image info dict.
            boxes_3d (list): List of 3d bounding boxes.
            labels (int): List of label ids.
            scores (optional[list]): List of scores.
            log_gt (bool): Whether to log ground truth or prediction boxes.

        Returns:
            Dictionary of bounding boxes to be logged.
        """
        bboxes = _box3d_to_img(boxes_3d, img_info, scores)
        wandb_boxes = {}
        box_data = []
        for bbox, label in zip(bboxes, labels):
            label = int(label)
            if len(bbox) == 5:
                confidence = float(bbox[4])
                class_name = self.class_id_to_label[label]
                box_caption = f'{class_name} {confidence:.2f}'
            else:
                box_caption = str(self.class_id_to_label[label])

            position = dict(
                minX=int(bbox[0]),
                minY=int(bbox[1]),
                maxX=int(bbox[2]),
                maxY=int(bbox[3]))

            box_data.append({
                'position': position,
                'class_id': label,
                'box_caption': box_caption,
                'domain': 'pixel'
            })
            if len(bbox) == 5:
                box_data[-1]['scores'] = {'confidence': float(bbox[4])}

        wandb_bbox_dict = {
            'box_data': box_data,
            'class_labels': self.class_id_to_label
        }

        if log_gt:
            wandb_boxes['ground_truth'] = wandb_bbox_dict
        else:
            wandb_boxes['predictions'] = wandb_bbox_dict

        return wandb_boxes


def _box3d_to_img(boxes_3d, img_info, scores=None):
    # Move box to camera frame
    from mmdet3d.core import LiDARInstance3DBoxes
    if isinstance(boxes_3d, LiDARInstance3DBoxes):
        boxes_3d = boxes_3d.convert_to(Box3DMode.CAM, img_info['lidar2cam'])
    # Project bbox to 2d
    box_corners_in_image = points_cam2img(
        boxes_3d.corners, proj_mat=img_info['cam2img'], meta=img_info)
    minxy = torch.min(box_corners_in_image, dim=-2)[0]
    maxxy = torch.max(box_corners_in_image, dim=-2)[0]
    if scores is None:
        bboxes = torch.cat([minxy, maxxy], dim=-1)
    else:
        bboxes = torch.cat([minxy, maxxy, scores[:, None]], dim=-1)
    return bboxes
