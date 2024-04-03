import numpy as np
import wandb
from opencood.tools.runner_temporal import BaseLogger
import matplotlib.pyplot as plt
from opencood.visualization.vis_utils import matplot_to_numpy


class NoLogger(BaseLogger):
    """
    No logger class. Does not log anything.
    """
    def __init__(self) -> None:
        super().__init__()
    

    def _log_metrics(self, metrics: dict, epoch: int, iteration: int, batch_len: int, **kwargs) -> None:
        pass


    def _log_visualization(self, visualization: np.ndarray or dict, epoch: int, iteration: int, batch_len: int, **kwargs) -> None:
        pass


    def finalize(self) -> None:
        pass


class WandBLogger(BaseLogger):
    def __init__(
            self, project: str, entity: str, name: str,
            config: dict, mode: str = 'online'
    ) -> None:
        """
        WandB logger class.
        :param project: project name
        :param entity: entity name
        :param name: run name
        :param config: model config dict
        :param mode: online or disabled
        """
        super().__init__()
        self.project = project
        self.entity = entity
        self.name = name
        self.config = config

        assert mode in ['online', 'disabled']
        self.mode = mode

        wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            mode=mode
        )
    

    def _log_metrics(self, metrics: dict, epoch: int, iteration: int, batch_len: int, **kwargs) -> None:
        wandb.log(metrics)


    def _log_visualization(self, visualization: np.ndarray or dict, epoch: int, iteration: int, batch_len: int, **kwargs) -> None:
        if visualization is None:
            print('Visualization is None and cannot be logged to wandb.')
        
        if 'caption_prefix' in kwargs:
            caption = kwargs['caption_prefix'] + f' Epoch {epoch} Iteration {iteration}'
        else:
            caption = f'Epoch {epoch} Iteration {iteration}'

        if isinstance(visualization, np.ndarray):
            w_img = wandb.Image(
                visualization,
                caption=caption)
            
            wandb.log({'visualizations': w_img})
        elif isinstance(visualization, plt.Figure):
            w_img = wandb.Image(
                matplot_to_numpy(visualization),
                caption=caption)
            
            wandb.log({'visualizations': w_img})
        elif isinstance(visualization, dict):
            # assume that it is a point cloud
            pc = wandb.Object3D(
                caption=caption,
                data_or_path=dict(
                    type='lidar/beta',
                    **visualization
                )
            )
            
            wandb.log({'point_cloud': pc})
        else:
            print('Visualization is not numpy array or dict and cannot be logged to wandb.')
        
        

    def finalize(self) -> None:
        wandb.finish()
