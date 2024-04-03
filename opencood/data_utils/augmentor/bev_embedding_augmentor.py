import torch
import torch.nn as nn

from opencood.data_utils.augmentor.augment_utils import signal_to_noise, zero_out, full_zero_out


class NoiseCombiner(nn.Module):
    def __init__(self, noise_module: list, config: str = 'Sequential'):
        """
        config: 'Sequential' or 'RandomSequential' or 'RandomSingle'
        """
        super(NoiseCombiner, self).__init__()
        self.noise_module = noise_module
        self.config = config
    
    def forward(self, bev_embedding: torch.Tensor):
        if self.config == 'Sequential':
            for module in self.noise_module:
                bev_embedding = module(bev_embedding)
        elif self.config == 'RandomSequential':
            # mix noise modules
            for module in self.noise_module[torch.randperm(len(self.noise_module))]:
                bev_embedding = module(bev_embedding)
        elif self.config == 'RandomSingle':
            # pick random noise module
            noise_module = self.noise_module[torch.randint(0, len(self.noise_module), (1,))]
            bev_embedding = noise_module(bev_embedding)
        else:
            raise NotImplementedError(f'config {self.config} is not implemented.')
        return bev_embedding


class ApplyNoise(nn.Module):
    def __init__(self, apply_probability: float):
        super(ApplyNoise, self).__init__()
        self.apply_probability = apply_probability

    def forward(self, bev_embedding: torch.Tensor):
        if torch.rand(1) < self.apply_probability:
            return self._forward(bev_embedding)
        else:
            return bev_embedding
    
    def _forward(self, bev_embedding: torch.Tensor):
        raise NotImplementedError


class SignalToNoise(ApplyNoise):
    def __init__(self, apply_probability: float, ratio: float = 0.5):
        super(SignalToNoise, self).__init__(apply_probability=apply_probability)
        self.ratio = ratio

    def _forward(self, bev_embedding: torch.Tensor):
        noisy_bev = signal_to_noise(bev_embedding, snr=self.ratio)
        return noisy_bev


class ZeroOut(ApplyNoise):
    def __init__(self, apply_probability: float, zero_out_probability: float = 0.1):
        super(ZeroOut, self).__init__(apply_probability=apply_probability)
        self.probability = zero_out_probability
    
    def _forward(self, bev_embedding: torch.Tensor):
        noisy_bev = zero_out(bev_embedding, probability=self.probability)
        return noisy_bev


class FullZeroOut(ApplyNoise):
    def __init__(self, apply_probability: float = 0.1):
        super(FullZeroOut, self).__init__(apply_probability=apply_probability)
    
    def _forward(self, bev_embedding: torch.Tensor):
        noisy_bev = full_zero_out(bev_embedding)
        return noisy_bev
