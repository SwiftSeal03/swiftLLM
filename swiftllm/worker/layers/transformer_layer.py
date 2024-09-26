import time
import dataclasses
import torch
import vllm_flash_attn_2_cuda as flash_attn_cuda
from swiftllm_c import \
    fused_add_rmsnorm_inplace, \
    silu_and_mul_inplace, \
    rotary_embedding_inplace, \
    linear, \
    store_kvcache#, paged_attention

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.weight import LlamaTransformerLayerWeight
from swiftllm.worker.infer_state import LlamaInferState

# from swiftllm.worker.kernels.linear import linear
# from swiftllm.worker.kernels.rotary_emb import rotary_embedding_inplace
# from swiftllm.worker.kernels.kvcache_mgmt import store_kvcache
# from swiftllm.worker.kernels.silu_and_mul import silu_and_mul_inplace
# from swiftllm.worker.kernels.rmsnorm import fused_add_rmsnorm_inplace
from swiftllm.worker.kernels.paged_attn import paged_attention

@dataclasses.dataclass
class KVCacheArgs:
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_swap: torch.Tensor
    v_swap: torch.Tensor
    q_cpu: torch.Tensor
    k_cpu: torch.Tensor
    v_cpu: torch.Tensor
    gpu_block_table: torch.Tensor | None
    cpu_block_table: torch.Tensor | None

class TransformerEvents:
    def __init__(self):
        self.stage_s = torch.cuda.Event(enable_timing=True)
        self.linear_e = torch.cuda.Event(enable_timing=True)
        self.prefill_e = torch.cuda.Event(enable_timing=True)
        self.gpudec_e = torch.cuda.Event(enable_timing=True)
        self.cpudec_e = torch.cuda.Event(enable_timing=True)
        self.cpudec_s = torch.cuda.Event(enable_timing=True)
        self.qkvtr_e = torch.cuda.Event(enable_timing=True)

    def get_prefill_time(self) -> float:
        return self.stage_s.elapsed_time(self.prefill_e)

    def get_gpudec_time(self) -> float:
        return self.stage_s.elapsed_time(self.gpudec_e)
    
    def get_linear_time(self) -> float:
        return self.stage_s.elapsed_time(self.linear_e)
    
    def get_cpuwrk_time(self) -> float:
        return self.stage_s.elapsed_time(self.cpudec_e)
    
    def get_cpudec_time(self) -> float:
        return self.cpudec_s.elapsed_time(self.cpudec_e)

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
        layer_id: int
    ):
        self.model_config = model_config
        self.engine_config = engine_config
        self.weight = weight
        self.next_layer_weight = next_layer_weight
        self.prefilling_stream = prefilling_stream
        self.decoding_piggyback_stream = decoding_piggyback_stream
        self.cpu_communication_stream = cpu_communication_stream
        self.layer_id = layer_id
        self.swapper = None
        
        self.events = [TransformerEvents() for _ in range(2)]

    def set_swapper(self, swapper):
        self.swapper = swapper

    def _transfer_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kvargs: KVCacheArgs,
        infer_state: LlamaInferState,
        cur_stage: int = 0
    ):
        """
        Initiate transfer of QKV to CPU buffers
        """
        if infer_state.cpu_num_decoding_seqs > 0:
            with torch.cuda.stream(self.cpu_communication_stream):
                torch.cuda.current_stream().wait_event(self.events[cur_stage].stage_s)
                qc = kvargs.q_cpu[:infer_state.cpu_num_decoding_seqs]
                kc = kvargs.k_cpu[:infer_state.cpu_num_decoding_seqs]
                vc = kvargs.v_cpu[:infer_state.cpu_num_decoding_seqs]
                qc.copy_(q[infer_state.gpu_token_end:], non_blocking=True)
                kc.copy_(k[infer_state.gpu_token_end:], non_blocking=True)
                vc.copy_(v[infer_state.gpu_token_end:], non_blocking=True)
                self.events[cur_stage].qkvtr_e.record()

    def _preproj(
        self,
        input_embds: torch.Tensor,
        residual_buf: torch.Tensor,
        infer_state: LlamaInferState,
        use_next_layer: bool = False
    ) -> tuple[torch.Tensor]:
        # We may need to use the next layer in piplined setting
        weight = self.weight if not use_next_layer else self.next_layer_weight

        start = time.perf_counter()*1e6
        fused_add_rmsnorm_inplace(
            input_embds,
            residual_buf,
            weight.attn_norm,
            self.model_config.rms_norm_eps
        )

        mid0 = time.perf_counter()*1e6
        # Calculate QKV
        q = linear(input_embds, weight.q_proj)		# [num_total_tokens, hidden_size]
        k = linear(input_embds, weight.k_proj)		# [num_total_tokens, num_kv_heads*head_dim]
        v = linear(input_embds, weight.v_proj)		# [num_total_tokens, num_kv_heads*head_dim]
        q = q.view(-1, self.model_config.num_q_heads,  self.model_config.head_dim)	# [num_total_tokens, num_q_heads, head_dim]
        k = k.view(-1, self.model_config.num_kv_heads, self.model_config.head_dim)	# [num_total_tokens, num_kv_heads, head_dim]
        v = v.view(-1, self.model_config.num_kv_heads, self.model_config.head_dim)	# [num_total_tokens, num_kv_heads, head_dim]

        mid1 = time.perf_counter()*1e6
        # Rotary emb
        rotary_embedding_inplace(
            q,
            k,
            infer_state.position_sin,
            infer_state.position_cos
        )
        end = time.perf_counter()*1e6
        # print(f"RMSNorm: {mid0 - start:.2f}, QKV GEMM: {mid1 - mid0:.2f}, Rotary emb: {end - mid1:.2f}")

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
        q_cpu = kvargs.q_cpu
        k_cpu = kvargs.k_cpu
        v_cpu = kvargs.v_cpu
        gpu_block_table = kvargs.gpu_block_table
        cpu_block_table = kvargs.cpu_block_table

        cur_layer_id = (self.layer_id + cur_stage) % self.model_config.num_layers

        start = time.perf_counter()*1e6
        if infer_state.num_prefill_seqs > 0:
            with torch.cuda.stream(self.prefilling_stream):
                torch.cuda.current_stream().wait_event(self.events[cur_stage].stage_s)
                # Here the performance of vLLM's flash attention is better than us,
                # so use vllm_flash_attn
                o[:infer_state.num_prefill_tokens, :] = flash_attn_cuda.varlen_fwd(
                    q[:infer_state.num_prefill_tokens, :, :],
                    k[:infer_state.num_prefill_tokens, :, :],
                    v[:infer_state.num_prefill_tokens, :, :],
                    None,
                    infer_state.prefill_seq_start_locs_with_end,
                    infer_state.prefill_seq_start_locs_with_end,
                    None,
                    None,
                    None,
                    infer_state.max_prefill_len,
                    infer_state.max_prefill_len,
                    0.0,
                    infer_state.softmax_scale,
                    False,
                    True,
                    -1, 
                    -1,
                    False,
                    None
                )[0].reshape(-1, self.model_config.hidden_size)
                self.events[cur_stage].prefill_e.record()
            torch.cuda.default_stream().wait_event(self.events[cur_stage].prefill_e)
        mid0 = time.perf_counter()*1e6

        # Actually we can further separate KV-cache storing for prefilling and decoding,
        # but since the kernel is fast enough, we put all to decoding stream for simplicity
        with torch.cuda.stream(self.decoding_piggyback_stream):
            torch.cuda.current_stream().wait_event(self.events[cur_stage].stage_s)
            if infer_state.num_prefill_seqs > 0 and not self.engine_config.ignore_kvcache:
                store_kvcache(
                    k,
                    v, # Here we only store k, v for prefilling, the kernel won't store decoding KVs
                    k_cache, v_cache,
                    gpu_block_table,
                    infer_state.gpu_seq_ids[:infer_state.num_prefill_seqs],
                    infer_state.prefill_seq_start_locs,
                    infer_state.prefill_seq_lens,
                    cur_layer_id,
                    infer_state.max_prefill_len
                )
            mid1 = time.perf_counter()*1e6

            if infer_state.gpu_num_decoding_seqs > 0:
                paged_attention(
                    q[infer_state.num_prefill_tokens:infer_state.gpu_token_end, :, :],
                    k[infer_state.num_prefill_tokens:infer_state.gpu_token_end, :, :],
                    v[infer_state.num_prefill_tokens:infer_state.gpu_token_end, :, :],
                    o[infer_state.num_prefill_tokens:infer_state.gpu_token_end, :],
                    k_cache, v_cache,
                    infer_state.softmax_scale,
                    gpu_block_table,
                    infer_state.gpu_seq_ids[infer_state.num_prefill_seqs:],
                    infer_state.gpu_decoding_seq_lens,
                    cur_layer_id,
                    infer_state.seq_block_size,
                    infer_state.num_seq_blocks,
                )
            self.events[cur_stage].gpudec_e.record()
        torch.cuda.default_stream().wait_event(self.events[cur_stage].gpudec_e)

        end = time.perf_counter()*1e6
        # print(f"Prefill launch: {mid0 - start:.2f}, KV-cache store: {mid1 - mid0:.2f}, GPU decoding: {end - mid1:.2f}")
                
        if infer_state.cpu_num_decoding_seqs > 0:
            with torch.cuda.stream(self.cpu_communication_stream):
                self.events[cur_stage].qkvtr_e.synchronize()
                self.events[cur_stage].cpudec_s.record()
                oc = o[infer_state.gpu_token_end:, :]
                o_cpu = torch.empty_like(oc, device='cpu', dtype=torch.float32, pin_memory=True)
                torch.ops.pacpu.paged_attention_cpu(
                    cur_layer_id,
                    infer_state.softmax_scale,
                    infer_state.cpu_seq_ids.tolist(),
                    infer_state.cpu_decoding_seq_lens.tolist(),

                    q_cpu[:infer_state.cpu_num_decoding_seqs],
                    k_cpu[:infer_state.cpu_num_decoding_seqs],
                    v_cpu[:infer_state.cpu_num_decoding_seqs],
                    k_swap,
                    v_swap,
                    cpu_block_table,
                    o_cpu
                )
                oc.copy_(o_cpu.to(torch.float16), non_blocking=True)
                self.events[cur_stage].cpudec_e.record()
            torch.cuda.default_stream().wait_event(self.events[cur_stage].cpudec_e)

    def _postproj(
        self,
        o: torch.Tensor,
        residual_buf: torch.Tensor
    ) -> torch.Tensor:
        # Output GEMM
        start = time.perf_counter()*1e6
        o = linear(o, self.weight.o_proj)	# [num_total_tokens, hidden_size]

        mid0 = time.perf_counter()*1e6
        # residual & FFN norm
        fused_add_rmsnorm_inplace(o, residual_buf, self.weight.ffn_norm, self.model_config.rms_norm_eps)

        mid1 = time.perf_counter()*1e6
        # FFN
        up_gate_proj = linear(o, self.weight.up_gate_proj)
        del o
        mid2 = time.perf_counter()*1e6
        silu_and_mul_inplace(up_gate_proj)
        mid3 = time.perf_counter()*1e6
        ffn_out = torch.nn.functional.linear(up_gate_proj[:, :self.model_config.ffn_inter_dim], self.weight.down_proj)
        end = time.perf_counter()*1e6
        # print(f"Output GEMM: {mid0 - start:.2f}, RMSNorm: {mid1 - mid0:.2f}, FFN GEMM: {mid2 - mid1:.2f}, SiLU: {mid3 - mid2:.2f}, Down-proj GEMM: {end - mid3:.2f}")
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
        self.events[0].stage_s.record()
        self._transfer_qkv(
            q, k, v, 
            kvargs, infer_state
        )
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
        cur_stage: int,
        src_block_ids: list[int],
        dst_block_ids: list[int]
    ) -> tuple[torch.Tensor]:
        """
        Do 1 pipeline stage:

            batch 0 : o0   |=> post-projection[i] -> pre-projection[i+1] |=> qkv0
            batch 1 : qkv1 |=>      attention[i + attn_layer_id_offs]    |=> [o1]

        buffer of o1 is given as input

        Note that batch-0 here may actually be batch-1 in the forward pass, should reverse the list
        in the second stage
        """
        
        # Here we put the linear_end_event at the beginning because batch 1 don't need to wait for batch 0's linear
        self.events[cur_stage].stage_s.record()

        self._transfer_qkv(
            q1, k1, v1, 
            kvargs, infer_states[1], cur_stage
        )

        f0 = self._postproj(o0, residual_buf0)
        q0, k0, v0 = self._preproj(f0, residual_buf0, infer_states[0], use_next_layer=True)
        del f0

        self.events[cur_stage].linear_e.record()
        
        if src_block_ids:
            with torch.cuda.stream(self.cpu_communication_stream):
                self.swapper.swap_out_blocks(self.layer_id, src_block_ids, dst_block_ids)

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
        src_block_ids: list[int],
        dst_block_ids: list[int]
    ) -> tuple[torch.Tensor]:
        """
        Do all jobs for 1 transformer layer for 2 batches

        Note that the weights of pre-projection need to be of the next layer compared to the post-projection

            batch 0 : o0   |=>  post-projection[i] -> pre-projection[i+1]  |        attention[i+1]                     |=> [o0']
            batch 1 : qkv1 |=>       attention[i]                          | post-projection[i] -> pre-projection[i+1] |=> qkv1'
        """
        rev_infer_states = infer_states[::-1]
        q0, k0, v0 = self._forward_pipeline_stage(
            o0, residual_buf0, o1, q1, k1, v1, 
            kvargs, infer_states, 0, src_block_ids, dst_block_ids
        )
        q1, k1, v1 = self._forward_pipeline_stage(
            o1, residual_buf1, o0, q0, k0, v0,
            kvargs, rev_infer_states, 1, [], []
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
        q0, k0, v0 = self._preproj(input_embds0, residual_buf0, infer_states[0], use_next_layer=True)
        # Wait for swappings to finish
        torch.cuda.current_stream().wait_stream(self.cpu_communication_stream)
        self.events[1].stage_s.record()
        self._transfer_qkv(
            q0, k0, v0, 
            kvargs, infer_states[0], cur_stage=1
        )
        q1, k1, v1 = self._preproj(input_embds1, residual_buf1, infer_states[1], use_next_layer=True)
        self.events[1].linear_e.record()
        # We use the last layer's last stage for the very first stage
        self._attention(
            input_embds0, q0, k0, v0,
            kvargs, infer_states[0], cur_stage=1
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
        infer_states: list[LlamaInferState],
        src_block_ids: list[int],
        dst_block_ids: list[int]
    ) -> tuple[torch.Tensor]:
        """
        Do the last stage of the pipeline for 2 batches

        batch0 : o0   |=> post-projection              |=> [f0]
        batch1 : qkv1 |=> attention -> post-projection |=> [f1]
        """
        self.events[0].stage_s.record()
        self._transfer_qkv(
            q1, k1, v1, 
            kvargs, infer_states[1], cur_stage=0
        )
        f0 = self._postproj(o0, residual_buf0)
        self.events[0].linear_e.record()
        # Here cur_stage is an offset of layer_id, we use last layer here
        self._attention(
            o1, q1, k1, v1,
            kvargs, infer_states[1], cur_stage=0
        )
        # Initiate swap out, this is safe because we won't use KV-cache until first stage of next iteration
        if src_block_ids:
            with torch.cuda.stream(self.cpu_communication_stream):
                self.swapper.swap_out_blocks(self.layer_id, src_block_ids, dst_block_ids)
        f1 = self._postproj(o1, residual_buf1)
        return f0, f1
    
    