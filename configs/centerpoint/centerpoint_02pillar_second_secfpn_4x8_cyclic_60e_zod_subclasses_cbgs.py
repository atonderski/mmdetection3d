_base_ = [
    'centerpoint_02pillar_second_secfpn_4x8_cyclic_60e_zod_subclasses.py',
]

data = dict(train=dict(type='CBGSDataset'))
