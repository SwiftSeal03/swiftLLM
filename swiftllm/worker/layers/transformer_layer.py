import time
import torch
import vllm_flash_attn

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.weight import LlamaTransformerLayerWeight
from swiftllm.worker.infer_state import LlamaInferState

from swiftllm.worker.kernels.linear import linear
from swiftllm.worker.kernels.rmsnorm import fused_add_rmsnorm_inplace
from swiftllm.worker.kernels.rotary_emb import rotary_embedding_inplace
from swiftllm.worker.kernels.paged_attn import paged_attention, cpu_paged_attention
from swiftllm.worker.kernels.kvcache_mgmt import store_kvcache
from swiftllm.worker.kernels.silu_and_mul import silu_and_mul_inplace

class LlamaTransformerLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        engine_config: EngineConfig,
        weight: LlamaTransformerLayerWeight,
        prefilling_stream: torch.cuda.Stream,
        decoding_piggyback_stream: torch.cuda.Stream,
        cpu_communication_stream: torch.cuda.Stream,
        layer_id: int
    ):
        self.model_config = model_config
        self.engine_config = engine_config
        self.weight = weight
        self.prefilling_stream = prefilling_stream
        self.decoding_piggyback_stream = decoding_piggyback_stream
        self.cpu_communication_stream = cpu_communication_stream
        self.layer_id = layer_id

        if engine_config.monitor_performance:
            self.attn_s_event = torch.cuda.Event(enable_timing=True)
            self.attn_e_event = torch.cuda.Event(enable_timing=True)

    def _preproj(
        self,
        input_embds: torch.Tensor,
        residual_buf: torch.Tensor,
        infer_state: LlamaInferState
    ) -> tuple[torch.Tensor]:
        fused_add_rmsnorm_inplace(
            input_embds,
            residual_buf,
            self.weight.attn_norm,
            self.model_config.rms_norm_eps
        )

        # Calculate QKV
        q = linear(input_embds, self.weight.q_proj)		# [num_total_tokens, hidden_size]
        k = linear(input_embds, self.weight.k_proj)		# [num_total_tokens, num_kv_heads*head_dim]
        v = linear(input_embds, self.weight.v_proj)		# [num_total_tokens, num_kv_heads*head_dim]
        q = q.view(-1, self.model_config.num_q_heads,  self.model_config.head_dim)	# [num_total_tokens, num_q_heads, head_dim]
        k = k.view(-1, self.model_config.num_kv_heads, self.model_config.head_dim)	# [num_total_tokens, num_kv_heads, head_dim]
        v = v.view(-1, self.model_config.num_kv_heads, self.model_config.head_dim)	# [num_total_tokens, num_kv_heads, head_dim]

        # Rotary emb
        rotary_embedding_inplace(
            q,
            k,
            infer_state
        )

        return q, k, v
    
    def _attention(
        self,
        o: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_swap: torch.Tensor,
        v_swap: torch.Tensor,
        gpu_block_table: torch.Tensor,
        cpu_block_table: torch.Tensor,
        infer_state: LlamaInferState,
        linear_end_event: torch.cuda.Event,
        layer_id_offs: int = 0,
    ):
        """
        Stores attention output of current batch into buffer o
        """

        # Attention
        if self.engine_config.monitor_performance:
            self.attn_s_event.record()

        cur_layer_id = self.layer_id + layer_id_offs

        if infer_state.num_prefill_seqs > 0:
            with torch.cuda.stream(self.prefilling_stream):
                torch.cuda.current_stream().wait_event(linear_end_event)
                # Here the performance of vLLM's flash attention is better than us,
                # so use vllm_flash_attn
                o[:infer_state.num_prefill_tokens, :] = vllm_flash_attn.flash_attn_varlen_func(
                    q[:infer_state.num_prefill_tokens, :, :],
                    k[:infer_state.num_prefill_tokens, :, :],
                    v[:infer_state.num_prefill_tokens, :, :],
                    infer_state.prefill_seq_start_locs_with_end,
                    infer_state.prefill_seq_start_locs_with_end,
                    infer_state.max_prefill_len,
                    infer_state.max_prefill_len,
                    softmax_scale=infer_state.softmax_scale,
                    causal=True
                ).reshape(-1, self.model_config.hidden_size)
                prefill_end_event = torch.cuda.Event()
                prefill_end_event.record()
            torch.cuda.default_stream().wait_event(prefill_end_event)

        # Actually we can further separate KV-cache storing for prefilling and decoding,
        # but since the kernel is fast enough, we put all to decoding stream for simplicity
        if not infer_state.ignore_kvcache and infer_state.gpu_token_end > 0:
            with torch.cuda.stream(self.decoding_piggyback_stream):
                torch.cuda.current_stream().wait_event(linear_end_event)
                store_kvcache(
                    k[:infer_state.gpu_token_end, :, :],
                    v[:infer_state.gpu_token_end, :, :],
                    k_cache, v_cache,
                    gpu_block_table,
                    self.model_config,
                    self.engine_config,
                    infer_state,
                    cur_layer_id
                )
            # Default stream doesn't need to wait for store_kvcache because:
            #   1. If there is no GPU decoding, the data won't be used until next iteration.
            #   2. If there is GPU decoding, the stream will be waited later.

        # Decoding stream doesn't need to wait for linear-end-event because it must be waited by store_kvcache
        if infer_state.gpu_num_decoding_seqs > 0:
            with torch.cuda.stream(self.decoding_piggyback_stream):
                paged_attention(
                    q[infer_state.num_prefill_tokens:infer_state.gpu_token_end, :, :],
                    k_cache, v_cache, gpu_block_table,
                    self.model_config, self.engine_config, infer_state,
                    cur_layer_id,
                    o[infer_state.num_prefill_tokens:infer_state.gpu_token_end, :],
                )
                pagpu_end_event = torch.cuda.Event()
                pagpu_end_event.record()
            torch.cuda.default_stream().wait_event(pagpu_end_event)
                
        if infer_state.cpu_num_decoding_seqs > 0:
            with torch.cuda.stream(self.cpu_communication_stream):
                torch.cuda.current_stream().wait_event(linear_end_event)
                cpu_paged_attention(
                    q[infer_state.gpu_token_end:, :, :],
                    k[infer_state.gpu_token_end:, :, :], 
                    v[infer_state.gpu_token_end:, :, :],
                    k_swap, v_swap, cpu_block_table,
                    self.model_config, self.engine_config, infer_state,
                    cur_layer_id,
                    o[infer_state.gpu_token_end:, :],
                )
                pacpu_end_event = torch.cuda.Event()
                pacpu_end_event.record()
            torch.cuda.default_stream().wait_event(pacpu_end_event)

        if self.engine_config.monitor_performance:
            self.attn_e_event.record()

        return o

    def _postproj(
        self,
        o: torch.Tensor,
        residual_buf: torch.Tensor
    ) -> torch.Tensor:
        # Output GEMM
        o = linear(o, self.weight.o_proj)	# [num_total_tokens, hidden_size]

        # residual & FFN norm
        fused_add_rmsnorm_inplace(o, residual_buf, self.weight.ffn_norm, self.model_config.rms_norm_eps)

        # FFN
        up_gate_proj = linear(o, self.weight.up_gate_proj)
        silu_and_mul_inplace(up_gate_proj)
        ffn_out = linear(up_gate_proj[:, :self.model_config.ffn_inter_dim], self.weight.down_proj)
        return ffn_out
    
    def forward(
        self,
        input_embds: torch.Tensor,  # [num_tokens, hidden_size]
        residual_buf: torch.Tensor, # [num_tokens, hidden_size]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_swap: torch.Tensor,
        v_swap: torch.Tensor,
        block_table: torch.Tensor,
        cpu_block_table: torch.Tensor,
        infer_state: LlamaInferState,
    ) -> torch.Tensor:
        # (fused) Add last layer's residual, and perform RMSNorm
        # Before: input_embds is the output of the last FFN block, and residual_buf
        #         is the residual to be added to input_embds
        # After: input_embds will be RMSNorm(input_embds + residual_buf), and
        #        residual_buf will be input_embds + residual_buf (which will be
        #        used as the residual after the attention block)
        q, k, v = self._preproj(input_embds, residual_buf, infer_state)
        linear_end_event = torch.cuda.Event()
        linear_end_event.record()
        self._attention(
            input_embds, q, k, v, 
            k_cache, v_cache, k_swap, v_swap, block_table, cpu_block_table, infer_state, 
            linear_end_event
        )
        q, k, v = None, None, None
        ffn_out = self._postproj(input_embds, residual_buf)
        
        return ffn_out

    def _forward_pipeline_stage(
        self,
        o0: torch.Tensor,  # [num_tokens, hidden_size]
        residual_buf0: torch.Tensor, # [num_tokens, hidden_size]
        o1: torch.Tensor,  # [num_tokens, hidden_size]
        q1: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
        k1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_swap: torch.Tensor,
        v_swap: torch.Tensor,
        gpu_block_table: torch.Tensor,
        cpu_block_table: torch.Tensor,
        infer_state0: LlamaInferState,
        infer_state1: LlamaInferState,
        attn_layer_id_offs: int
    ) -> tuple[torch.Tensor]:
        """
        Do 1 pipeline stage:

            batch 0 : o0   |=> post-projection[i] -> pre-projection[i+1] |=> qkv0
            batch 1 : qkv1 |=>      attention[i + attn_layer_id_offs]    |=> [o1]

        buffer of o1 is given as input
        """
        
        # Here we put the linear_end_event at the beginning because batch 1 don't need to wait for batch 0's linear
        linear_end_event = torch.cuda.Event()
        linear_end_event.record()

        f0 = self._postproj(o0, residual_buf0)
        q0, k0, v0 = self._preproj(f0, residual_buf0, infer_state0)
        self._attention(
            o1, q1, k1, v1,
            k_cache, v_cache, k_swap, v_swap, gpu_block_table, cpu_block_table, infer_state1, 
            linear_end_event, attn_layer_id_offs
        )

        return q0, k0, v0

    def _forward_double(
        self,
        o0: torch.Tensor,  # [num_tokens, hidden_size]
        residual_buf0: torch.Tensor, # [num_tokens, hidden_size]
        residual_buf1: torch.Tensor, # [num_tokens, hidden_size]
        o1: torch.Tensor,  # [num_tokens, hidden_size]
        q1: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
        k1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_swap: torch.Tensor,
        v_swap: torch.Tensor,
        gpu_block_table: torch.Tensor,
        cpu_block_table: torch.Tensor,
        infer_state0: LlamaInferState,
        infer_state1: LlamaInferState,
    ) -> tuple[torch.Tensor]:
        """
        Do all jobs for 1 transformer layer for 2 batches

        Note that the weights of pre-projection need to be of the next layer compared to the post-projection

            batch 0 : o0   |=>  post-projection[i] -> pre-projection[i+1]  |        attention[i+1]                     |=> o0'
            batch 1 : qkv1 |=>       attention[i]                          | post-projection[i] -> pre-projection[i+1] |=> qkv1'
        """
        q0, k0, v0 = self._forward_pipeline_stage(
            o0, residual_buf0, o1, q1, k1, v1, 
            k_cache, v_cache, k_swap, v_swap, gpu_block_table, cpu_block_table, infer_state0, infer_state1, 0
        )
        q1, k1, v1 = self._forward_pipeline_stage(
            o1, residual_buf1, o0, q0, k0, v0,
            k_cache, v_cache, k_swap, v_swap, gpu_block_table, cpu_block_table, infer_state1, infer_state0, 1
        )

        return q1, k1, v1
    