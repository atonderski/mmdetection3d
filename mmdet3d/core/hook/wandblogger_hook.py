# Copyright (c) OpenMMLab. All rights reserved.
import json

import mmcv
import numpy as np
import torch
from mmcv.runner import HOOKS

from mmdet3d.core.bbox.structures.utils import points_cam2img
from mmdet.core import MMDetWandbHook


@HOOKS.register_module()
class MMDet3DWandbHook(MMDetWandbHook):
    """Enhanced Wandb logger hook for MMDetection3D with image visualizations.

    Comparing with the :cls:`mmdetection.runner.MMDetWandbHook`, this hook can
    handle 3d object predictions and ground truths (instead of 2D only).

    For more details, please refer to the parent documentation:
    :cls:`mmdetection.runner.MMDetWandbHook`
    """

    def __init__(self,
                 range_3d=None,
                 max_points=None,
                 project_to_img=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.range_3d = range_3d
        self.max_points = max_points
        self.project_to_img = project_to_img

    def _init_data_table(self):
        """Initialize the W&B Tables for validation data."""
        if self.project_to_img:
            columns = ['image_name', 'image']
            self.data_table_img = self.wandb.Table(columns=columns)
        if self.range_3d is not None:
            columns = ['sample_name', 'points']
            self.data_table_3d = self.wandb.Table(columns=columns)

    def _init_pred_table(self):
        """Initialize the W&B Tables for model evaluation."""
        if self.project_to_img:
            columns = ['image_name', 'ground_truth', 'prediction']
            self.eval_table_img = self.wandb.Table(columns=columns)
        if self.range_3d is not None:
            columns = ['sample_name', 'ground_truth', 'prediction']
            self.eval_table_3d = self.wandb.Table(columns=columns)

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

        if self.project_to_img:
            self._add_img_ground_truth(runner)
        if self.range_3d is not None:
            self._add_3d_ground_truth(runner)

    def _get_pred_boxes(self, results):
        """Get the predicted bounding boxes from the results."""
        boxes_3d = results['boxes_3d']
        scores = results['scores_3d'].numpy()
        labels = results['labels_3d'].numpy()
        # Remove bounding boxes and masks with score lower than threshold.
        if self.bbox_score_thr > 0:
            assert boxes_3d is not None
            inds = scores > self.bbox_score_thr
            boxes_3d = boxes_3d[inds]
            scores = scores[inds]
            labels = labels[inds]
        return boxes_3d, scores, labels

    def _log_predictions(self, results):

        for ndx, eval_image_index in enumerate(self.eval_idxs):
            # Get the result
            if self.project_to_img:
                table_idxs = self.data_table_img_ref.get_index()
                assert len(table_idxs) == len(self.eval_idxs)
                img_results = results[eval_image_index]['img_bbox']
                preds = self._get_pred_boxes(img_results)
                img_info = self.val_dataset.data_infos[eval_image_index]
                data_ref = self.data_table_img_ref.data[ndx]
                self._add_img_predictions(preds, img_info, data_ref)
            if self.range_3d is not None:
                table_idxs = self.data_table_3d_ref.get_index()
                assert len(table_idxs) == len(self.eval_idxs)
                results_3d = results[eval_image_index]
                if 'pts_bbox' in results_3d:
                    results_3d = results_3d['pts_bbox']
                preds = self._get_pred_boxes(results_3d)
                data_ref = self.data_table_3d_ref.data[ndx]
                self._add_3d_predictions(preds, data_ref)

    def _log_data_table(self):
        """Log the W&B Tables for validation data as artifact and calls
        `use_artifact` on it so that the evaluation table can use the reference
        of already uploaded images.

        This allows the data to be uploaded just once.
        """
        data_artifact = self.wandb.Artifact('val', type='dataset')
        if self.project_to_img:
            data_artifact.add(self.data_table_img, 'val_data_img')
        if self.range_3d is not None:
            data_artifact.add(self.data_table_3d, 'val_data_3d')
        self.wandb.run.use_artifact(data_artifact)
        data_artifact.wait()
        to_log = {}
        if self.project_to_img:
            self.data_table_img_ref = data_artifact.get('val_data_img')
            to_log['ground_truth_img'] = self.data_table_img_ref
        if self.range_3d is not None:
            self.data_table_3d_ref = data_artifact.get('val_data_3d')
            to_log['ground_truth_3d'] = self.data_table_3d_ref
        # self.wandb.log(to_log)

    def _log_eval_table(self, idx):
        """Log the W&B Tables for model evaluation.

        The table will be logged multiple times creating new version. Use this
        to compare models at different intervals interactively.
        """
        pred_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.id}_pred', type='evaluation')
        if self.by_epoch:
            aliases = ['latest', f'epoch_{idx}']
        else:
            aliases = ['latest', f'iter_{idx}']
        to_log = {}
        if self.project_to_img:
            pred_artifact.add(self.eval_table_img, 'eval_data_img')
            to_log['predictions_img'] = self.eval_table_img.data[0][2]
        if self.range_3d is not None:
            pred_artifact.add(self.eval_table_3d, 'eval_data_3d')
            to_log['predictions_3d'] = self.eval_table_3d.data[0][2]
        self.wandb.run.log_artifact(pred_artifact, aliases=aliases)
        # log the first row of the eval table to the wandb run
        # self.wandb.log(to_log)

    # 3D Visualizations #

    def _add_3d_ground_truth(self, runner):
        # Get lidar loading pipeline
        from mmdet3d.datasets.pipelines.loading import (LoadPointsFromFile,
                                                        ZodLoadPointsFromFile)
        lidar_loader = None
        for t in self.val_dataset.pipeline.transforms:
            if isinstance(t, (LoadPointsFromFile, ZodLoadPointsFromFile)):
                lidar_loader = t
        if lidar_loader is None:
            runner.logger.warning(
                'LoadPointsFromFile is required to add points to 3d '
                'visualization. Only bounding boxes will be shown now.')

        for idx in self.eval_idxs:
            data_info = self.val_dataset.get_data_info(idx)
            points = lidar_loader(data_info)['points'].tensor[:, :3].numpy()
            if self.max_points and points.shape[0] > self.max_points:
                # Randomly subsample points
                points = points[np.random.choice(
                    len(points), self.max_points, replace=False)]
            print(points.dtype)
            anno = self.val_dataset.get_ann_info(idx)
            boxes = self._get_wandb_bboxes_3d(
                anno['gt_bboxes_3d'],
                anno['gt_labels_3d'],
            )
            self.data_table_3d.add_data(
                f'vis_3d_{idx}',
                self.wandb.Object3D({
                    'type': 'lidar/beta',
                    'points': points,
                    'boxes': boxes,
                }),
            )
            # self.wandb.log({
            #     'gt_3d':
            #     self.wandb.Object3D({
            #         'type': 'lidar/beta',
            #         'points': points,
            #         'boxes': boxes,
            #     })
            # })

    def _add_3d_predictions(self, preds, data_ref):
        boxes_3d, labels, scores = preds
        assert scores is not None
        pred_wandb_boxes = self._get_wandb_bboxes_3d(boxes_3d, labels, scores)
        with open(data_ref[1]._path) as f:
            data = json.load(f)
        points = np.array(data['points'], dtype=np.float32)
        gt_wandb_boxes = data['boxes']
        wandb_boxes = np.array(pred_wandb_boxes.tolist() + gt_wandb_boxes)
        pred_obj = self.wandb.Object3D({
            'type': 'lidar/beta',
            'points': points,
            'boxes': wandb_boxes
        })
        self.eval_table_3d.add_data(data_ref[0], data_ref[1], pred_obj)
        self.wandb.log({'predictions_3d': pred_obj})

    def _get_wandb_bboxes_3d(self, boxes_3d, labels, scores=None):
        """Get list of structured dict for logging bounding boxes to W&B.

        Args:
            boxes_3d (list): List of 3d bounding boxes.
            labels (int): List of label ids.
            scores (optional[list]): List of scores.

        Returns:
            numpy array of bounding boxes to be logged.
        """
        box_data = []
        for i, (label, corners) in enumerate(zip(labels, boxes_3d.corners)):
            if not isinstance(label, int):
                label = int(label)
            class_name = str(self.class_id_to_label.get(label, str(label)))
            # if scores is not None:
            #     confidence = scores[i]
            #     # box_caption = f'{class_name} {confidence:.2f}'
            # else:
            #     box_caption = class_name
            if scores is not None:
                box_data.append({
                    'corners': corners.tolist(),
                    'label': f'pred-{class_name}',
                    'color': (255, 0, 0),  # red for pred
                    # 'color': (255 * float(scores[i]), 0, 0),  # red for pred
                })
            else:
                box_data.append({
                    'corners': corners.tolist(),
                    'label': f'gt-{class_name}',
                    'color': (0, 255, 0),  # green for gt
                })
        return np.array(box_data)

    # Image Visualizations #

    def _add_img_ground_truth(self, runner):
        # Get image loading pipeline
        from mmdet3d.datasets.pipelines.loading import LoadImageFromFileMono3D
        img_loader = None
        # TODO: support other image loading pipelines
        for t in self.val_dataset.pipeline.transforms:
            if isinstance(t, LoadImageFromFileMono3D):
                img_loader = t
        if img_loader is None:
            self.log_evaluation = False
            runner.logger.warning(
                'LoadImageFromFile is required to add images '
                'to W&B Tables.')
            return
        img_prefix = self.val_dataset.img_prefix

        for idx in self.eval_idxs:
            img_info = self.val_dataset.data_infos[idx]
            image_name = img_info.get('filename', f'img_{idx}')
            img_meta = img_loader(
                dict(img_info=img_info, img_prefix=img_prefix))
            image = mmcv.bgr2rgb(img_meta['img'])

            data_ann = self.val_dataset.get_ann_info(idx)
            wandb_boxes = self._get_wandb_bboxes_img(
                img_meta,
                data_ann['bboxes_3d'],
                data_ann['labels_3d'],
            )

            # Log a row to the data table.
            self.data_table_img.add_data(
                image_name,
                self.wandb.Image(
                    image, boxes=wandb_boxes, classes=self.class_set))

    def _add_img_predictions(self, preds, img_info, data_ref):
        boxes_3d, labels, scores = preds
        wandb_boxes = self._get_wandb_bboxes_img(
            img_info, boxes_3d, labels, scores, log_gt=False)
        self.eval_table_img.add_data(
            data_ref[0], data_ref[1],
            self.wandb.Image(
                data_ref[1], boxes=wandb_boxes, classes=self.class_set))

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
        # Project bbox to 2d
        box_corners_in_image = points_cam2img(
            boxes_3d.corners,
            proj_mat=img_info['cam_intrinsic'],
            meta={
                'distortion': img_info['cam_distortion'],
                'proj_model': img_info['proj_model']
            },
        )
        minxy = torch.min(box_corners_in_image, dim=-2)[0]
        maxxy = torch.max(box_corners_in_image, dim=-2)[0]
        if scores is None:
            bboxes = torch.cat([minxy, maxxy], dim=-1)
        else:
            bboxes = torch.cat([minxy, maxxy, scores[:, None]], dim=-1)

        wandb_boxes = {}
        box_data = []
        for bbox, label in zip(bboxes, labels):
            if not isinstance(label, int):
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
