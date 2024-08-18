import time
import dataclasses
import torch
import vllm_flash_attn
from swiftllm_c import fused_add_rmsnorm_inplace, silu_and_mul_inplace, rotary_embedding_inplace
from concurrent.futures import ThreadPoolExecutor

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.weight import LlamaTransformerLayerWeight
from swiftllm.worker.infer_state import LlamaInferState

from swiftllm.worker.kernels.linear import linear
# from swiftllm.worker.kernels.rotary_emb import rotary_embedding_inplace
from swiftllm.worker.kernels.paged_attn import paged_attention
from swiftllm.worker.kernels.kvcache_mgmt import store_kvcache
# from swiftllm.worker.kernels.silu_and_mul import silu_and_mul_inplace
# from swiftllm.worker.kernels.rmsnorm import fused_add_rmsnorm_inplace

@dataclasses.dataclass
class KVCacheArgs:
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_swap: torch.Tensor
    v_swap: torch.Tensor
    gpu_block_table: torch.Tensor
    cpu_block_table: torch.Tensor

class LlamaTransformerLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        engine_config: EngineConfig,
        weight: LlamaTransformerLayerWeight,
        next_layer_weight: LlamaTransformerLayerWeight | None,
        prefilling_stream: torch.cuda.Stream,
        decoding_piggyback_stream: torch.cuda.Stream,
        cpu_communication_stream: torch.cuda.Stream,
        executor: ThreadPoolExecutor,
        layer_id: int
    ):
        self.model_config = model_config
        self.engine_config = engine_config
        self.weight = weight
        self.next_layer_weight = next_layer_weight
        self.prefilling_stream = prefilling_stream
        self.decoding_piggyback_stream = decoding_piggyback_stream
        self.cpu_communication_stream = cpu_communication_stream
        self.executor = executor
        self.layer_id = layer_id
        
        self.stage_s_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]    
        self.linear_e_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        self.prefill_e_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        self.gpudec_e_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        self.cpudec_e_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        self.cpudec_s_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]


    def _preproj(
        self,
        input_embds: torch.Tensor,
        residual_buf: torch.Tensor,
        infer_state: LlamaInferState,
        use_next_layer: bool = False
    ) -> tuple[torch.Tensor]:
        # We may need to use the next layer in piplined setting
        weight = self.weight if not use_next_layer else self.next_layer_weight

        start = time.perf_counter()
        fused_add_rmsnorm_inplace(
            input_embds,
            residual_buf,
            weight.attn_norm,
            self.model_config.rms_norm_eps
        )

        # Calculate QKV
        q = linear(input_embds, weight.q_proj)		# [num_total_tokens, hidden_size]
        k = linear(input_embds, weight.k_proj)		# [num_total_tokens, num_kv_heads*head_dim]
        v = linear(input_embds, weight.v_proj)		# [num_total_tokens, num_kv_heads*head_dim]
        q = q.view(-1, self.model_config.num_q_heads,  self.model_config.head_dim)	# [num_total_tokens, num_q_heads, head_dim]
        k = k.view(-1, self.model_config.num_kv_heads, self.model_config.head_dim)	# [num_total_tokens, num_kv_heads, head_dim]
        v = v.view(-1, self.model_config.num_kv_heads, self.model_config.head_dim)	# [num_total_tokens, num_kv_heads, head_dim]

        # Rotary emb
        rotary_embedding_inplace(
            q,
            k,
            infer_state.position_sin,
            infer_state.position_cos
        )
        end = time.perf_counter()
        print(f"Preproj launch time: {(end - start)*1000:.3f} ms")

        return q, k, v
    
    def _attention(
        self,
        o: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kvargs: KVCacheArgs,
        infer_state: LlamaInferState,
        cur_stage: int = 0, # also as the offset of layer_id
    ):
        """
        Stores attention output of current batch into buffer o
        """

        k_cache = kvargs.k_cache
        v_cache = kvargs.v_cache
        k_swap = kvargs.k_swap
        v_swap = kvargs.v_swap
        gpu_block_table = kvargs.gpu_block_table
        cpu_block_table = kvargs.cpu_block_table

        cur_layer_id = self.layer_id + cur_stage

        start = time.perf_counter()
        if infer_state.num_prefill_seqs > 0:
            with torch.cuda.stream(self.prefilling_stream):
                torch.cuda.current_stream().wait_event(self.stage_s_events[cur_stage])
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
                self.prefill_e_events[cur_stage].record()
            torch.cuda.default_stream().wait_event(self.prefill_e_events[cur_stage])

        # Actually we can further separate KV-cache storing for prefilling and decoding,
        # but since the kernel is fast enough, we put all to decoding stream for simplicity
        if not infer_state.ignore_kvcache and infer_state.gpu_token_end > 0:
            with torch.cuda.stream(self.decoding_piggyback_stream):
                torch.cuda.current_stream().wait_event(self.stage_s_events[cur_stage])
                store_kvcache(
                    k[:infer_state.num_prefill_tokens, :, :],
                    v[:infer_state.num_prefill_tokens, :, :],
                    k_cache, v_cache,
                    gpu_block_table,
                    infer_state.gpu_seq_ids[:infer_state.num_prefill_seqs],
                    infer_state.prefill_seq_start_locs,
                    infer_state.prefill_seq_lens,
                    cur_layer_id,
                    infer_state.max_prefill_len
                )
            # Default stream doesn't need to wait for store_kvcache because:
            #   1. If there is no GPU decoding, the data won't be used until next iteration.
            #   2. If there is GPU decoding, the stream will be waited later.

        # GPU decoding stream doesn't need to wait for linear-end-event because it must be waited by store_kvcache
        if infer_state.gpu_num_decoding_seqs > 0:
            with torch.cuda.stream(self.decoding_piggyback_stream):
                paged_attention(
                    q[infer_state.num_prefill_tokens:infer_state.gpu_token_end, :, :],
                    k[infer_state.num_prefill_tokens:infer_state.gpu_token_end, :, :],
                    v[infer_state.num_prefill_tokens:infer_state.gpu_token_end, :, :],
                    o[infer_state.num_prefill_tokens:infer_state.gpu_token_end, :],
                    k_cache, v_cache, gpu_block_table,
                    infer_state.gpu_seq_ids[infer_state.num_prefill_seqs:],
                    infer_state.gpu_decoding_seq_lens,
                    cur_layer_id,
                    infer_state.seq_block_size,
                    infer_state.num_seq_blocks,
                    infer_state.softmax_scale
                )
                self.gpudec_e_events[cur_stage].record()
            torch.cuda.default_stream().wait_event(self.gpudec_e_events[cur_stage])

        elapsed = time.perf_counter() - start
        print(f"Attention launch time: {elapsed*1000:.3f} ms")
                
        if infer_state.cpu_num_decoding_seqs > 0:
            with torch.cuda.stream(self.cpu_communication_stream):
                torch.cuda.current_stream().wait_event(self.stage_s_events[cur_stage])
                self.cpudec_s_events[cur_stage].record()
                q_cpu = q[infer_state.gpu_token_end:, :, :].cpu()
                k_cpu = k[infer_state.gpu_token_end:, :, :].cpu()
                v_cpu = v[infer_state.gpu_token_end:, :, :].cpu()
                oc = o[infer_state.gpu_token_end:, :]
                o_cpu = torch.empty_like(oc, device='cpu', dtype=torch.float32)
                torch.ops.pacpu.paged_attention_cpu(
                    cur_layer_id,
                    infer_state.softmax_scale,
                    infer_state.cpu_seq_ids.tolist(),
                    infer_state.cpu_decoding_seq_lens.tolist(),

                    q_cpu,
                    k_cpu,
                    v_cpu,
                    k_swap,
                    v_swap,
                    cpu_block_table,
                    o_cpu
                )
                oc.copy_(o_cpu.to(torch.float16), non_blocking=True)
                self.cpudec_e_events[cur_stage].record()
            torch.cuda.default_stream().wait_event(self.cpudec_e_events[cur_stage])

    def _postproj(
        self,
        o: torch.Tensor,
        residual_buf: torch.Tensor
    ) -> torch.Tensor:
        # Output GEMM
        start = time.perf_counter()
        o = linear(o, self.weight.o_proj)	# [num_total_tokens, hidden_size]

        # residual & FFN norm
        fused_add_rmsnorm_inplace(o, residual_buf, self.weight.ffn_norm, self.model_config.rms_norm_eps)

        # FFN
        up_gate_proj = linear(o, self.weight.up_gate_proj)
        del o
        silu_and_mul_inplace(up_gate_proj)
        ffn_out = linear(up_gate_proj[:, :self.model_config.ffn_inter_dim], self.weight.down_proj)
        end = time.perf_counter()
        print(f"Postproj launch time: {(end - start)*1000:.3f} ms\n")
        return ffn_out
    
    def forward(
        self,
        input_embds: torch.Tensor,  # [num_tokens, hidden_size]
        residual_buf: torch.Tensor, # [num_tokens, hidden_size]
        kvargs: KVCacheArgs,
        infer_state: LlamaInferState,
    ) -> torch.Tensor:
        # (fused) Add last layer's residual, and perform RMSNorm
        # Before: input_embds is the output of the last FFN block, and residual_buf
        #         is the residual to be added to input_embds
        # After: input_embds will be RMSNorm(input_embds + residual_buf), and
        #        residual_buf will be input_embds + residual_buf (which will be
        #        used as the residual after the attention block)
        q, k, v = self._preproj(input_embds, residual_buf, infer_state)
        self.stage_s_events[0].record()
        self._attention(
            input_embds, q, k, v, 
            kvargs, infer_state
        )
        del q, k, v
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
        kvargs: KVCacheArgs,
        infer_states: list[LlamaInferState],
        cur_stage: int
    ) -> tuple[torch.Tensor]:
        """
        Do 1 pipeline stage:

            batch 0 : o0   |=> post-projection[i] -> pre-projection[i+1] |=> qkv0
            batch 1 : qkv1 |=>      attention[i + attn_layer_id_offs]    |=> [o1]

        buffer of o1 is given as input
        """
        
        # Here we put the linear_end_event at the beginning because batch 1 don't need to wait for batch 0's linear
        self.stage_s_events[cur_stage].record()

        f0 = self._postproj(o0, residual_buf0)
        q0, k0, v0 = self._preproj(f0, residual_buf0, infer_states[0], use_next_layer=True)
        del f0

        if self.engine_config.monitor_performance:
            self.linear_e_events[cur_stage].record()

        self._attention(
            o1, q1, k1, v1,
            kvargs, infer_states[1], cur_stage
        )

        return q0, k0, v0

    def forward_double(
        self,
        q1: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
        k1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        o0: torch.Tensor,  # [num_tokens, hidden_size], contains the output of the last layer
        o1: torch.Tensor,  # [num_tokens, hidden_size], needs to be updated
        residual_buf0: torch.Tensor, # [num_tokens, hidden_size]
        residual_buf1: torch.Tensor, # [num_tokens, hidden_size]
        kvargs: KVCacheArgs,
        infer_states: list[LlamaInferState],
    ) -> tuple[torch.Tensor]:
        """
        Do all jobs for 1 transformer layer for 2 batches

        Note that the weights of pre-projection need to be of the next layer compared to the post-projection

            batch 0 : o0   |=>  post-projection[i] -> pre-projection[i+1]  |        attention[i+1]                     |=> [o0']
            batch 1 : qkv1 |=>       attention[i]                          | post-projection[i] -> pre-projection[i+1] |=> qkv1'
        """
        q0, k0, v0 = self._forward_pipeline_stage(
            o0, residual_buf0, o1, q1, k1, v1, 
            kvargs, infer_states, cur_stage=0
        )
        q1, k1, v1 = self._forward_pipeline_stage(
            o1, residual_buf1, o0, q0, k0, v0,
            kvargs, infer_states, cur_stage=1
        )

        return q1, k1, v1

    def forward_first_stage(
        self,
        input_embds0: torch.Tensor,  # [num_tokens, hidden_size], would contain o0
        input_embds1: torch.Tensor,  # [num_tokens, hidden_size]
        residual_buf0: torch.Tensor, # [num_tokens, hidden_size]
        residual_buf1: torch.Tensor, # [num_tokens, hidden_size]
        kvargs: KVCacheArgs,
        infer_states: list[LlamaInferState],
    ) -> tuple[torch.Tensor]:
        """
        Do the first stage of the pipeline for 2 batches

        batch0 : input_embeds0 |=> pre-projection -> attention       |=> [o0]
        batch1 : input_embeds1 |=>                  pre-projection   |=> q1, k1, v1
        """
        q0, k0, v0 = self._preproj(input_embds0, residual_buf0, infer_states[0])
        # This event of first layer would be recorded again
        self.stage_s_events[0].record()
        q1, k1, v1 = self._preproj(input_embds1, residual_buf1, infer_states[1])
        self._attention(
            input_embds0, q0, k0, v0,
            kvargs, infer_states[0], cur_stage=0
        )
        return q1, k1, v1

    def forward_last_stage(
        self,
        q1: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
        k1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        o0: torch.Tensor,  # [num_tokens, hidden_size]
        o1: torch.Tensor,  # [num_tokens, hidden_size]
        residual_buf0: torch.Tensor, # [num_tokens, hidden_size]
        residual_buf1: torch.Tensor, # [num_tokens, hidden_size]
        kvargs: KVCacheArgs,
        infer_states: list[LlamaInferState]
    ) -> tuple[torch.Tensor]:
        """
        Do the last stage of the pipeline for 2 batches

        batch0 : o0   |=> post-projection              |=> [f0]
        batch1 : qkv1 |=> attention -> post-projection |=> [f1]
        """
        self.stage_s_events[0].record()
        f0 = self._postproj(o0, residual_buf0)
        # Here cur_stage is an offset of layer_id, we use last layer here
        self._attention(
            o1, q1, k1, v1,
            kvargs, infer_states[1], cur_stage=0
        )
        f1 = self._postproj(o1, residual_buf1)
        return f0, f1
    
    