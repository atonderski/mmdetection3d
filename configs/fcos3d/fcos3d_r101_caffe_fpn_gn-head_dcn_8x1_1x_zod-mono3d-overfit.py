_base_ = ['fcos3d_r101_caffe_fpn_gn-head_dcn_8x1_1x_zod-mono3d.py']

_mmdir = 'data/zod/mmdet3d'
data = dict(
    train=dict(
        times=480,
        dataset=dict(
            ann_file=f'{_mmdir}/zod-single_infos_train_mono3d.coco.json',
        )),  # Train is a wrapped dataset
    val=dict(ann_file=f'{_mmdir}/zod-single_infos_val_mono3d.coco.json', ))

log_config = dict(interval=10, )
