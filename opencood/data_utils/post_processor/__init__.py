from opencood.data_utils.post_processor.base_postprocessor import BasePostprocessor
from opencood.data_utils.post_processor.camera_bev_postprocessor import CameraBevPostprocessor
from opencood.data_utils.post_processor.bev_postprocessor import BevPostprocessor
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor

__all__ = {
    'BevPostprocessor': BevPostprocessor,
    'BasePostprocessor': BasePostprocessor,
    'CameraBevPostprocessor': CameraBevPostprocessor,
    'VoxelPostprocessor': VoxelPostprocessor
}


def build_postprocessor(anchor_cfg, train):
    process_method_name = anchor_cfg['core_method']
    assert process_method_name in ['BevPostprocessor',
                                   'BasePostprocessor',
                                   'CameraBevPostprocessor',
                                   'VoxelPostprocessor']
    anchor_generator = __all__[process_method_name](
        anchor_params=anchor_cfg,
        train=train
    )

    return anchor_generator
