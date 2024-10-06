import torch

from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import LlamaWeight

class LlamaPreLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        weights: LlamaWeight,
    ):
        self.model_config = model_config
        self.weights = weights
    
    def forward(
        self,
        input_ids: list[int]
    ) -> torch.Tensor:
        input_gpu = torch.tensor(input_ids, dtype=torch.int32, device='cuda')
        input_embdings = torch.embedding(self.weights.wte, input_gpu, padding_idx=-1)
        return input_embdings
    