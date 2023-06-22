_base_ = [
    'centerpoint_02pillar_second_secfpn_4x8_cyclic_40e_zod_subclasses.py',
]

evaluation = dict(interval=60)
runner = dict(max_epochs=60)
