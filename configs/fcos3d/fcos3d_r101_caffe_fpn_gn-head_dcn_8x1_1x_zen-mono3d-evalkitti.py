_base_ = ['fcos3d_r101_caffe_fpn_gn-head_dcn_8x1_1x_zen-mono3d.py']

data = dict(
    train=dict(eval_version='kitti'),
    val=dict(eval_version='kitti'),
    test=dict(eval_version='kitti'))


data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        times=240,
        dataset=dict(
            ann_file='data/zod/mmdet3d/zen-single_infos_train_mono3d.coco.json',
            )),  # Train is a wrapped dataset
    val=dict(
        ann_file='data/zod/mmdet3d/zen-single_infos_val_mono3d.coco.json',
    ),
)

# learning policy
lr_config = dict(warmup_iters=50)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='MMDet3DWandbHook',
            init_kwargs=dict(
                project='mmdet-test',
                name='fcos3d_mono_kitti',
            ),
            log_checkpoint=False,
            log_checkpoint_metadata=False,
            num_eval_images=10,
            bbox_score_thr=0.1,
        )])
