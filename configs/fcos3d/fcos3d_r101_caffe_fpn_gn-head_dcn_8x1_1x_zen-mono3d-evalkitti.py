_base_ = ['fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_zen-mono3d.py']

data = dict(
    train=dict(eval_version='kitti'),
    val=dict(eval_version='kitti'),
    test=dict(eval_version='kitti'))

log_config = dict(
    # interval=10,
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
