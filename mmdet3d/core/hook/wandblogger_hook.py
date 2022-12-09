# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
from mmcv.runner import HOOKS

from mmdet3d.core.bbox.structures.utils import points_cam2img
from mmdet.core import MMDetWandbHook


@HOOKS.register_module()
class MMDet3DWandbHook(MMDetWandbHook):
    """Enhanced Wandb logger hook for MMDetection3D.

    Comparing with the :cls:`mmdetection.runner.MMDetWandbHook`, this hook can
    handle 3d object predictions and ground truths (instead of 2D only).

    For more details, please refer to the parent documentation:
    :cls:`mmdetection.runner.MMDetWandbHook`
    """

    def _add_ground_truth(self, runner):
        # Get image loading pipeline
        from mmdet3d.datasets.pipelines.loading import LoadImageFromFileMono3D
        img_loader = None
        for t in self.val_dataset.pipeline.transforms:
            if isinstance(t, LoadImageFromFileMono3D):
                img_loader = t

        if img_loader is None:
            self.log_evaluation = False
            runner.logger.warning(
                'LoadImageFromFile is required to add images '
                'to W&B Tables.')
            return

        # Select the images to be logged.
        self.eval_image_indexs = np.arange(len(self.val_dataset))
        # Set seed so that same validation set is logged each time.
        np.random.seed(42)
        np.random.shuffle(self.eval_image_indexs)
        self.eval_image_indexs = self.eval_image_indexs[:self.num_eval_images]

        CLASSES = self.val_dataset.CLASSES
        self.class_id_to_label = {
            id + 1: name
            for id, name in enumerate(CLASSES)
        }
        self.class_set = self.wandb.Classes([{
            'id': id,
            'name': name
        } for id, name in self.class_id_to_label.items()])

        img_prefix = self.val_dataset.img_prefix

        for idx in self.eval_image_indexs:
            img_info = self.val_dataset.data_infos[idx]
            image_name = img_info.get('filename', f'img_{idx}')
            # img_height, img_width = img_info['height'], img_info['width']

            img_meta = img_loader(
                dict(img_info=img_info, img_prefix=img_prefix))

            # Get image and convert from BGR to RGB
            image = mmcv.bgr2rgb(img_meta['img'])

            data_ann = self.val_dataset.get_ann_info(idx)
            bboxes = data_ann['bboxes']
            labels = data_ann['labels']

            # Get dict of bounding boxes to be logged.
            assert len(bboxes) == len(labels)
            wandb_boxes = self._get_wandb_bboxes(bboxes, labels, log_gt=True)

            # Log a row to the data table.
            self.data_table.add_data(
                image_name,
                self.wandb.Image(
                    image, boxes=wandb_boxes, classes=self.class_set))

    def _log_predictions(self, results):
        table_idxs = self.data_table_ref.get_index()
        assert len(table_idxs) == len(self.eval_image_indexs)

        for ndx, eval_image_index in enumerate(self.eval_image_indexs):
            # Get the result
            current_results = results[eval_image_index]['img_bbox']
            img_info = self.val_dataset.data_infos[eval_image_index]
            boxes_3d = current_results['boxes_3d']
            scores = current_results['scores_3d']
            labels = current_results['labels_3d']

            # Remove bounding boxes and masks with score lower than threshold.
            if self.bbox_score_thr > 0:
                assert boxes_3d is not None
                inds = scores > self.bbox_score_thr
                boxes_3d = boxes_3d[inds]
                scores = scores[inds]
                labels = labels[inds]

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
            boxes_2d = torch.cat([minxy, maxxy, scores[:, None]], dim=-1)

            # Get dict of bounding boxes to be logged.
            wandb_boxes = self._get_wandb_bboxes(
                boxes_2d, labels, log_gt=False)

            # Log a row to the eval table.
            self.eval_table.add_data(
                self.data_table_ref.data[ndx][0],
                self.data_table_ref.data[ndx][1],
                self.wandb.Image(
                    self.data_table_ref.data[ndx][1],
                    boxes=wandb_boxes,
                    classes=self.class_set))

    def _get_wandb_bboxes(self, bboxes, labels, log_gt=True):
        """Get list of structured dict for logging bounding boxes to W&B.

        Args:
            bboxes (list): List of bounding box coordinates in
                        (minX, minY, maxX, maxY) format.
            labels (int): List of label ids.
            log_gt (bool): Whether to log ground truth or prediction boxes.

        Returns:
            Dictionary of bounding boxes to be logged.
        """
        wandb_boxes = {}

        box_data = []
        for bbox, label in zip(bboxes, labels):
            if not isinstance(label, int):
                label = int(label)
            label = label + 1

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

    def _log_eval_table(self, idx):
        """Log the W&B Tables for model evaluation.

        The table will be logged multiple times creating new version. Use this
        to compare models at different intervals interactively.
        """
        super()._log_eval_table(idx)
        # log the first row of the eval table to the wandb run
        self.wandb.log({'predictions': self.eval_table.data[0][2]})
