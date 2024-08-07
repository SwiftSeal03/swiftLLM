import torch

from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import LlamaWeight
from swiftllm.worker.kernels.rmsnorm import rmsnorm_inplace
from swiftllm.worker.infer_state import LlamaInferState
from swiftllm.worker.kernels.linear import linear

class LlamaPostLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        weights: LlamaWeight,
    ):
        self.model_config = model_config
        self.weights = weights

    def _get_last_input(
        self,
        input_embds: torch.Tensor,
        infer_state: LlamaInferState
    ) -> torch.Tensor:
        # Slice to get the last token embedding for each request
        last_token_indices = torch.cat(
            (
                infer_state.prefill_seq_start_locs + infer_state.prefill_seq_lens - 1,
                torch.arange(infer_state.num_prefill_tokens, infer_state.num_tokens, device=input_embds.device, dtype=torch.int32)
            ), dim=0
        )
        last_input = torch.empty((infer_state.batch_size, self.model_config.hidden_size), device=input_embds.device, dtype=input_embds.dtype)
        last_input[:, :] = input_embds[last_token_indices, :]
        return last_input
    
    def _forward(
        self,
        last_input: torch.Tensor
    ) -> torch.Tensor:
        # Apply RMS-norm
        rmsnorm_inplace(
            last_input,
            self.weights.final_norm,
            self.model_config.rms_norm_eps
        )
        logits = linear(last_input, self.weights.lm_head)    # [batch_size, vocab_size]
        output_tokens = torch.argmax(logits, dim=1)
        return output_tokens
    
    def forward(
        self,
        input_embds: torch.Tensor,	# [num_total_tokens, hidden_size]
        infer_state: LlamaInferState
    ) -> torch.Tensor:
        last_input = self._get_last_input(input_embds, infer_state)
        return self._forward(last_input)
    
    def forward_double(
        self,
        input_embds0: torch.Tensor,
        input_embds1: torch.Tensor,
        infer_state0: LlamaInferState,
        infer_state1: LlamaInferState
    ):
        last_inputs0 = self._get_last_input(input_embds0, infer_state0)
        last_inputs1 = self._get_last_input(input_embds1, infer_state1)
        last_inputs = torch.cat((last_inputs0, last_inputs1), dim=0)
        return self._forward(last_inputs)
    