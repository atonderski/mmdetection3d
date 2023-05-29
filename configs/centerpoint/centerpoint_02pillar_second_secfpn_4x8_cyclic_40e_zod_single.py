_base_ = [
    'centerpoint_02pillar_second_secfpn_4x8_cyclic_40e_zod.py',
]
data_root = 'data/zod/'
data = dict(
    train=dict(
        times=16 * 50,
        dataset=dict(ann_file=data_root +
                     'mmdet3d/zod-single_infos_train.pkl'),
    ),
    val=dict(ann_file=data_root + 'mmdet3d/zod-single_infos_val.pkl'),
    test=dict(ann_file=data_root + 'mmdet3d/zod-single_infos_val.pkl'))

log_config = dict(interval=10, )

evaluation = dict(interval=1)
