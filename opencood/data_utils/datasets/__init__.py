
from opencood.data_utils.datasets.bev_seg_datasets.camera_only.late_fusion_dataset import CamLateFusionDataset as BEVSegCamLateFusionDataset
from opencood.data_utils.datasets.bev_seg_datasets.camera_only.intermediate_fusion_dataset import CamIntermediateFusionDataset as BEVSegCamIntermediateFusionDataset
from opencood.data_utils.datasets.bev_seg_datasets.camera_only.scenario_intermediate_fusion_dataset import CamScenarioIntermediateFusionDataset as BEVSegCamScenarioIntermediateFusionDataset

from opencood.data_utils.datasets.bev_seg_datasets.camera_only_embeddings.embedding_dataset import ScenarioEmbeddingsDataset
from opencood.data_utils.datasets.bev_seg_datasets.camera_only_embeddings.embedding_dataset_no_ram import ScenarioEmbeddingsDatasetNoRAM


__all__ = {
    'BEVSegCamLateFusionDataset': BEVSegCamLateFusionDataset,
    'BEVSegCamIntermediateFusionDataset': BEVSegCamIntermediateFusionDataset,
    'BEVSegCamScenarioIntermediateFusionDataset': BEVSegCamScenarioIntermediateFusionDataset,
    'ScenarioEmbeddingsDataset': ScenarioEmbeddingsDataset,
    'ScenarioEmbeddingsDatasetNoRAM': ScenarioEmbeddingsDatasetNoRAM
}

# the final range for evaluation
GT_RANGE = [-100, -40, -5, 100, 40, 3]
# The communication range for cavs
COM_RANGE = 70

def build_dataset(dataset_cfg, visualize=False, train=True, isSim=False, validate=False, **kwargs):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"
    assert dataset_name in ['BEVSegCamLateFusionDataset',
                            'BEVSegCamIntermediateFusionDataset',
                            'BEVSegCamScenarioIntermediateFusionDataset',
                            'ScenarioEmbeddingsDataset',
                            'ScenarioEmbeddingsDatasetNoRAM'
                            ], error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train,
        validate=validate,
        isSim=isSim,
        **kwargs
    )

    return dataset