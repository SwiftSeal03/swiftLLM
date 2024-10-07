import time
import dataclasses
import torch
import vllm_flash_attn_2_cuda as flash_attn_cuda
from swiftllm_c import \
    fused_add_rmsnorm_inplace, \
    silu_and_mul_inplace, \
    rotary_embedding_inplace, \
    store_kvcache#, paged_attention
    # linear, \

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.weight import LlamaTransformerLayerWeight
from swiftllm.worker.block_manager import Swapper
from swiftllm.structs import SubBatch

from swiftllm.worker.kernels.linear import linear
# from swiftllm.worker.kernels.rotary_emb import rotary_embedding_inplace
# from swiftllm.worker.kernels.kvcache_mgmt import store_kvcache
# from swiftllm.worker.kernels.silu_and_mul import silu_and_mul_inplace
# from swiftllm.worker.kernels.rmsnorm import fused_add_rmsnorm_inplace
from swiftllm.worker.kernels.paged_attn import paged_attention

class TransformerEvents:
    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        self.stage_s = torch.cuda.Event(enable_timing=True)
        self.linr_e = torch.cuda.Event(enable_timing=True)
        self.pref_e = torch.cuda.Event(enable_timing=True)
        self.gdec_e = torch.cuda.Event(enable_timing=True)
        self.qkvtr_e = torch.cuda.Event()
        self.lnch_s = 0.0
        self.lnch_m = 0.0
        self.cdec_s = 0.0
        self.cdec_e = 0.0
        self.lnch_e = 0.0
    
    @property
    def linr_time(self) -> float:
        return self.stage_s.elapsed_time(self.linr_e)

    @property
    def pref_time(self) -> float:
        return self.linr_e.elapsed_time(self.pref_e)

    @property
    def gdec_time(self) -> float:
        return self.pref_e.elapsed_time(self.gdec_e)

    @property
    def lnch_time(self) -> float:
        return self.lnch_e - self.cdec_e + self.lnch_m - self.lnch_s

    @property
    def cdec_time(self) -> float:
        return self.cdec_e - self.cdec_s

    def pf_record(self, name: str):
        if self.engine_config.monitor_performance:
            getattr(self, name).record()

    def pf_time(self, name: str):
        if self.engine_config.monitor_performance:
            setattr(self, name, time.perf_counter() * 1e3) # ms

    def pf_time_nocpu(self):
        if self.engine_config.monitor_performance:
            self.lnch_m = self.cdec_s = self.cdec_e = time.perf_counter()

class LlamaTransformerLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        engine_config: EngineConfig,
        weight: LlamaTransformerLayerWeight,
        next_layer_weight: LlamaTransformerLayerWeight | None,
        cpu_communication_stream: torch.cuda.Stream,
        layer_id: int
    ):
        self.model_config = model_config
        self.engine_config = engine_config
        self.weight = weight
        self.next_layer_weight = next_layer_weight
        self.cpu_communication_stream = cpu_communication_stream
        self.layer_id = layer_id

        self.events = [TransformerEvents(engine_config) for _ in range(2)]

        self.swapper = None

    def set_meta_args(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_swap: torch.Tensor,
        v_swap: torch.Tensor,
        q_cpu: torch.Tensor,
        k_cpu: torch.Tensor,
        v_cpu: torch.Tensor,
        o_cpu: torch.Tensor,
        gpu_block_table: torch.Tensor,
        cpu_block_table: torch.Tensor,
        swapper: Swapper
    ):
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.k_swap = k_swap
        self.v_swap = v_swap
        self.q_cpu = q_cpu
        self.k_cpu = k_cpu
        self.v_cpu = v_cpu
        self.o_cpu = o_cpu
        self.gpu_block_table = gpu_block_table
        self.cpu_block_table = cpu_block_table
        self.swapper = swapper

    def set_batches(self, batches: list[SubBatch]):
        self.batches = batches

    def set_buffers(
        self,
        input_embedss: list[torch.Tensor],
        residual_bufs: list[torch.Tensor]
    ):
        self.input_embedss = input_embedss
        self.residual_bufs = residual_bufs

    def _transfer_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_id: int = 0,
        cur_stage: int = 0
    ):
        """
        Initiate transfer of QKV to CPU buffers
        """
        batch = self.batches[batch_id]
        if batch.num_cdecs > 0:
            with torch.cuda.stream(self.cpu_communication_stream):
                # Wait until QKV is ready
                torch.cuda.current_stream().wait_stream(torch.cuda.default_stream())
                qc = self.q_cpu[:batch.num_cdecs]
                kc = self.k_cpu[:batch.num_cdecs]
                vc = self.v_cpu[:batch.num_cdecs]
                qc.copy_(q[-batch.num_cdecs:], non_blocking=True)
                kc.copy_(k[-batch.num_cdecs:], non_blocking=True)
                vc.copy_(v[-batch.num_cdecs:], non_blocking=True)
                self.events[cur_stage].qkvtr_e.record()

    def _swap_out_blocks(
        self,
        batch_id: int = 0
    ):
        """
        Swap blocks from GPU to CPU, assume that new prefilled KVs are ready in the last stage
        """
        batch = self.batches[batch_id]
        if batch.num_cprfs > 0:
            with torch.cuda.stream(self.cpu_communication_stream):
                # If there are no decoding sequences, we need to wait for the last stage to finish; otherwise, we already waited
                if not batch.cdec_reqs:
                    torch.cuda.current_stream().wait_stream(torch.cuda.default_stream())
                self.swapper.swap_out_blocks(self.layer_id, batch.src_blk_ids, batch.dst_blk_ids)

    def _preproj(
        self,
        input_embds: torch.Tensor,
        batch_id: int = 0,
        layer_off: int = 0
    ) -> tuple[torch.Tensor]:
        """
        Perform pre-projection, including RMSNorm, QKV calculation, and rotary embedding
        """
        batch = self.batches[batch_id]
        weight = self.weight if not layer_off else self.next_layer_weight

        fused_add_rmsnorm_inplace(
            input_embds,
            self.residual_bufs[batch_id],
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
            batch.position_sin,
            batch.position_cos
        )

        # Here we only store k, v for prefilling, the kernel won't store decoding KVs
        if batch.num_prefs > 0 and self.swapper is not None:
            store_kvcache(
                k,
                v,
                self.k_cache, self.v_cache,
                self.gpu_block_table,
                batch.prgd_seq_ids[:batch.num_prefs],
                batch.pref_st_locs_we,
                batch.prgd_seq_lens[:batch.num_prefs],
                (self.layer_id + layer_off) % self.model_config.num_layers,
                batch.max_pref_toks
            )

        return q, k, v
    
    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_id: int = 0,
        cur_stage: int = 0 # also as the offset of layer_id
    ):
        """
        Stores attention output of current batch into buffer o
        """

        batch = self.batches[batch_id]
        events = self.events[cur_stage]
        o = self.input_embedss[batch_id]
        cur_layer_id = (self.layer_id + cur_stage) % self.model_config.num_layers

        if batch.num_prefs > 0:
            # with torch.cuda.stream(self.prefilling_stream):
                # torch.cuda.current_stream().wait_event(self.events[cur_stage].stage_s)
                # Here the performance of vLLM's flash attention is better than us,
                # so use vllm_flash_attn
            o[:batch.sum_pref_toks, :] = flash_attn_cuda.varlen_fwd(
                q[:batch.sum_pref_toks, :, :],
                k[:batch.sum_pref_toks, :, :],
                v[:batch.sum_pref_toks, :, :],
                None,
                batch.pref_st_locs_we,
                batch.pref_st_locs_we,
                None,
                None,
                None,
                batch.max_pref_toks,
                batch.max_pref_toks,
                0.0,
                batch.softmax_scale,
                False,
                True,
                -1, 
                -1,
                False,
                None
            )[0].view(-1, self.model_config.hidden_size)
        events.pf_record("pref_e")

        # Actually we can further separate KV-cache storing for prefilling and decoding,
        # but since the kernel is fast enough, we put all to decoding stream for simplicity
        if batch.num_gdecs > 0:
            # with torch.cuda.stream(self.decoding_piggyback_stream):
            #     torch.cuda.current_stream().wait_event(self.events[cur_stage].stage_s)
            gpu_token_end = batch.metadata.s - batch.num_cdecs
            paged_attention(
                q[batch.sum_pref_toks:gpu_token_end, :, :],
                k[batch.sum_pref_toks:gpu_token_end, :, :],
                v[batch.sum_pref_toks:gpu_token_end, :, :],
                o[batch.sum_pref_toks:gpu_token_end, :],
                self.k_cache, self.v_cache,
                batch.softmax_scale,
                self.gpu_block_table,
                batch.prgd_seq_ids[batch.num_prefs:],
                batch.prgd_seq_lens[batch.num_prefs:],
                cur_layer_id,
                batch.seq_block_size,
                batch.num_seq_blocks,
            )
        events.pf_record("gdec_e")
                
        if batch.num_cdecs > 0:
            og = o[-batch.num_cdecs:, :]
            oc = self.o_cpu[:batch.num_cdecs]
            events.pf_time("lnch_m")
            self.events[cur_stage].qkvtr_e.synchronize()
            events.pf_time("cdec_s")
            torch.ops.pacpu.paged_attention_cpu(
                cur_layer_id,
                batch.softmax_scale,
                batch.cdec_seq_ids_list,
                batch.cdec_seq_lens_list,

                self.q_cpu[:batch.num_cdecs],
                self.k_cpu[:batch.num_cdecs],
                self.v_cpu[:batch.num_cdecs],
                self.k_swap,
                self.v_swap,
                self.cpu_block_table,
                oc
            )
            events.pf_time("cdec_e")
            with torch.cuda.stream(self.cpu_communication_stream):
                og.copy_(oc, non_blocking=True)
            torch.cuda.default_stream().wait_stream(self.cpu_communication_stream)
        else:
            events.pf_time_nocpu()

    def _postproj(
        self,
        batch_id: int = 0
    ) -> torch.Tensor:
        o = linear(self.input_embedss[batch_id], self.weight.o_proj)
        fused_add_rmsnorm_inplace(o, self.residual_bufs[batch_id], self.weight.ffn_norm, self.model_config.rms_norm_eps)
        up_gate_proj = linear(o, self.weight.up_gate_proj)
        del o
        silu_and_mul_inplace(up_gate_proj)
        ffn_out = torch.nn.functional.linear(up_gate_proj[:, :self.model_config.ffn_inter_dim], self.weight.down_proj)
        del up_gate_proj
        return ffn_out
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # (fused) Add last layer's residual, and perform RMSNorm
        # Before: input_embds is the output of the last FFN block, and residual_buf
        #         is the residual to be added to input_embds
        # After: input_embds will be RMSNorm(input_embds + residual_buf), and
        #        residual_buf will be input_embds + residual_buf (which will be
        #        used as the residual after the attention block)
        self.events[0].pf_record("stage_s")
        self.events[0].pf_time("lnch_s")
        q, k, v = self._preproj(input)
        self.events[0].pf_record("linr_e")
        self._transfer_qkv(q, k, v)
        self._swap_out_blocks()
        self._attention(q, k, v)
        del q, k, v
        self.events[1].pf_record("stage_s")
        ffn_out = self._postproj()
        self.events[0].pf_time("lnch_e")
        self.events[1].pf_record("linr_e")
        return ffn_out

    def _forward_pipeline_stage(
        self,
        q1: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
        k1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        cur_stage: int,
    ) -> tuple[torch.Tensor]:
        """
        Do 1 pipeline stage:

            batch 0 : o0   |=> post-projection[i] -> pre-projection[i+1] |=> qkv0
            batch 1 : qkv1 |=>      attention[i + attn_layer_id_offs]    |=> [o1]

        buffer of o1 is given as input

        Note that batch-0 here may actually be batch-1 in the forward pass, should reverse the list
        in the second stage
        """
        self.events[cur_stage].pf_record("stage_s")
        self.events[cur_stage].pf_time("lnch_s")
        self._transfer_qkv(q1, k1, v1, batch_id=cur_stage^1, cur_stage=cur_stage)
        self._swap_out_blocks(batch_id=cur_stage)
        f0 = self._postproj(batch_id=cur_stage)
        q0, k0, v0 = self._preproj(f0, batch_id=cur_stage, layer_off=1)
        del f0
        self.events[cur_stage].pf_record("linr_e")
        self._attention(q1, k1, v1, batch_id=cur_stage^1, cur_stage=cur_stage)
        self.events[cur_stage].pf_time("lnch_e")

        return q0, k0, v0

    def forward_double(
        self,
        q1: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
        k1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
    ) -> tuple[torch.Tensor]:
        """
        Do all jobs for 1 transformer layer for 2 batches

        Note that the weights of pre-projection need to be of the next layer compared to the post-projection

            batch 0 : o0   |=>  post-projection[i] -> pre-projection[i+1]  |        attention[i+1]                     |=> [o0']
            batch 1 : qkv1 |=>       attention[i]                          | post-projection[i] -> pre-projection[i+1] |=> qkv1'
        """
        q0, k0, v0 = self._forward_pipeline_stage(q1, k1, v1, cur_stage=0)
        q1, k1, v1 = self._forward_pipeline_stage(q0, k0, v0, cur_stage=1)

        return q1, k1, v1

    def forward_first_stage(
        self,
    ) -> tuple[torch.Tensor]:
        """
        Do the first stage of the pipeline for 2 batches

        batch0 : input_embeds0 |=> pre-projection -> attention       |=> [o0]
        batch1 : input_embeds1 |=>                  pre-projection   |=> q1, k1, v1
        """
        q0, k0, v0 = self._preproj(self.input_embedss[0], batch_id=0, layer_off=1)
        torch.cuda.current_stream().wait_stream(self.cpu_communication_stream) # Wait for swappings to finish

        self.events[1].pf_record("stage_s")
        self.events[1].pf_time("lnch_s")
        self._transfer_qkv(q0, k0, v0, batch_id=0, cur_stage=1)
        q1, k1, v1 = self._preproj(self.input_embedss[1], batch_id=1, layer_off=1)
        self.events[1].pf_record("linr_e")
        self._attention(q0, k0, v0, batch_id=0, cur_stage=1)
        self.events[1].pf_time("lnch_e")

        return q1, k1, v1

    def forward_last_stage(
        self,
        q1: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
        k1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
    ) -> tuple[torch.Tensor]:
        """
        Do the last stage of the pipeline for 2 batches

        batch0 : o0   |=> post-projection              |=> [f0]
        batch1 : qkv1 |=> attention -> post-projection |=> [f1]
        """
        self.events[0].pf_record("stage_s")
        self.events[0].pf_time("lnch_s")
        self._transfer_qkv(q1, k1, v1, batch_id=1, cur_stage=0)
        self._swap_out_blocks(batch_id=0)
        f0 = self._postproj(batch_id=0)
        self.events[0].pf_record("linr_e")
        self._attention(q1, k1, v1, batch_id=1, cur_stage=0)
        self.events[0].pf_time("lnch_e")

        f1 = self._postproj(batch_id=1)

        return f0, f1
    
    