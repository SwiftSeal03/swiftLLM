import itertools
import math
import dataclasses
import json
from pprint import pprint

import numpy as np
import torch

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import load_weights
from swiftllm.worker.block_manager import BlockManager, Swapper
from swiftllm.utils import GB

from .layers.pre_layer import LlamaPreLayer
from .layers.transformer_layer import LlamaTransformerLayer, TransformerEvents
from .layers.post_layer import LlamaPostLayer
from .infer_state import LlamaInferState

@dataclasses.dataclass
class ModelForwardArgs:
    input_ids_list: list[list[int]]
    seq_ids_list: list[int]
    decoding_seq_lens_list: list[int]
    swap_out_seq_ids: list[int]
    cpu_num_decoding_seqs: int = 0

class ModelEvents:
    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        self.frwd_s = torch.cuda.Event(enable_timing=True)
        self.fstg_s = torch.cuda.Event(enable_timing=True)
        self.mnbd_s = torch.cuda.Event(enable_timing=True)
        self.mnbd_e = torch.cuda.Event(enable_timing=True)
        self.lstg_e = torch.cuda.Event(enable_timing=True)
        self.frwd_e = torch.cuda.Event(enable_timing=True)

    def pf_record(self, name:str):
        if self.engine_config.monitor_performance:
            getattr(self, name).record()


class ModelPerfResult:
    def __init__(
        self, 
        layers: list[LlamaTransformerLayer],
        model_events: ModelEvents,
        use_pipline: bool
    ):
        torch.cuda.synchronize() # Ensure all events are recorded
        if use_pipline:
            self.linr_times = np.array([[layer.events[i].linr_time for i in range(2)] for layer in layers[:-1]])
            self.pref_times = np.array([[layer.events[i].pref_time for i in range(2)] for layer in layers])
            self.gdec_times = np.array([[layer.events[i].gdec_time for i in range(2)] for layer in layers])
            self.cdec_times = np.array([[layer.events[i].cdec_time for i in range(2)] for layer in layers])
            self.lnch_times = np.array([[layer.events[i].lnch_time for i in range(2)] for layer in layers])
        else:
            self.linr_times = np.array([sum(layer.events[i].linr_time for i in range(2)) for layer in layers])
            self.pref_times = np.array([layer.events[0].pref_time for layer in layers])
            self.gdec_times = np.array([layer.events[0].gdec_time for layer in layers])
            self.cdec_times = np.array([layer.events[0].cdec_time for layer in layers])
            self.lnch_times = np.array([layer.events[0].lnch_time for layer in layers])

        self.prlr_time = model_events.frwd_s.elapsed_time(model_events.fstg_s)
        self.fstg_time = model_events.fstg_s.elapsed_time(model_events.mnbd_s)
        self.mnbd_time = model_events.mnbd_s.elapsed_time(model_events.mnbd_e)
        self.lstg_time = model_events.mnbd_e.elapsed_time(model_events.lstg_e)
        self.polr_time = model_events.lstg_e.elapsed_time(model_events.frwd_e)

        self.avg_linr_time = self.linr_times.mean()
        self.avg_pref_time = self.pref_times.mean()
        self.avg_gdec_time = self.gdec_times.mean()
        self.avg_cdec_time = self.cdec_times.mean()
        self.avg_lnch_time = self.lnch_times.mean()

    def __repr__(self):
        return json.dumps({
            "avg_linr_time": self.avg_linr_time,
            "avg_pref_time": self.avg_pref_time,
            "avg_gdec_time": self.avg_gdec_time,
            "avg_cdec_time": self.avg_cdec_time,
            "avg_lnch_time": self.avg_lnch_time
        }, indent=2)

    @staticmethod
    def mean(results: list["ModelPerfResult"], name: str) -> float:
        """
        Compute the average of a field in a list of ModelPerfResult objects.
        """
        return np.array([getattr(result, name) for result in results]).mean()



class LlamaModel:
    """
    LlamaModel - A Llama model that can be used for inference.

    This class also acts as a "worker" that resides on a particular GPU, waiting
    for the control plane (the scheduler) to send commands.

    To initialize, please:
    - call __init__()
    - call load_weights()
    - call profile_num_blocks() on one worker
    - call init_kvcache_and_swap()
    """

    @torch.inference_mode()
    def __init__(
        self,
        engine_config: EngineConfig
    ):
        """
        Initialize the LlamaModel.
        """
        self.engine_config = engine_config

        # Load model config
        self.model_config = LlamaModelConfig.load_from_model_path(engine_config.model_path)

        # Weight and RoPE cache
        self.weight = None
        self._cos_cached = self._sin_cached = None

        # Layers
        self.pre_layer = None
        self.transformer_layers = None
        self.post_layer = None

        # KV Cache
        self.num_blocks = None
        self.k_cache = self.v_cache = None
        self.k_swap = self.v_swap = None

        if engine_config.library_path:
            torch.ops.load_library(engine_config.library_path)
        
        self.cpu_communication_stream = torch.cuda.Stream()

        # List of performance results, unused if monitor_performance is False
        self.perf_results = []
        self.events = ModelEvents(engine_config)

        
    @torch.inference_mode()
    def load_weights(self):
        """
        Load weights and initialize layers
        """
        # Load weights
        self.weight = load_weights(
            self.model_config,
            torch.float16,
            self.engine_config.model_path,
            self.engine_config.use_dummy
        )

        # Initialize rotary embeddings
        self._init_to_get_rotary()

        # Initialize layers
        self.pre_layer = LlamaPreLayer(self.model_config, self.weight)
        self.transformer_layers = [
            LlamaTransformerLayer(
                self.model_config,
                self.engine_config,
                self.weight.layers[layer_id],
                self.weight.layers[layer_id + 1 - self.model_config.num_layers],
                self.cpu_communication_stream,
                layer_id
            )
            for layer_id in range(self.model_config.num_layers)
        ]
        self.post_layer = LlamaPostLayer(self.model_config, self.weight)

    @torch.inference_mode()
    def profile_num_blocks(self) -> int:
        """
        Profiler the number of GPU blocks

        We run a forged prefill batch with the maximum number of tokens and
        sequences, record the peak memory usage, and infer the number of blocks
        that can be allocated.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Synthesis a prefill batch
        num_tokens = self.engine_config.max_tokens_in_batch
        batch_size = self.engine_config.max_batch_size
        input_lens = [num_tokens // batch_size] * batch_size
        input_lens[-1] += num_tokens % batch_size
        input_ids = [
            [0 for _ in range(input_len)]
            for input_len in input_lens
        ]
        seq_ids = list(range(batch_size))
        self.k_cache = self.v_cache = None # pylint: disable=attribute-defined-outside-init
        self.engine_config.ignore_kvcache = True
        _ = self.forward(ModelForwardArgs(input_ids, seq_ids, [], []))
        self.engine_config.ignore_kvcache = False
        torch.cuda.synchronize()

        # peak_memory = torch.cuda.max_memory_allocated()
        # total_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory, total_memory = torch.cuda.mem_get_info()
        peak_memory = total_memory - free_memory
        useable_memory = total_memory*self.engine_config.gpu_mem_utilization
        print(f"[Model.profile] GPU total memory: {total_memory/GB:.2f} GB, runtime peak memory: {peak_memory/GB:.2f} GB")
        if useable_memory < peak_memory:
            raise RuntimeError(f"Peak memory {peak_memory/GB:.2f} GB exceeds usable memory {useable_memory/GB:.2f} GB ({total_memory/GB:.2f} GB * {self.engine_config.gpu_mem_utilization})")
        block_size_bytes = self.engine_config.block_size * self.model_config.get_kvslot_size()
        num_gpu_blocks = math.floor((useable_memory - peak_memory) / block_size_bytes)

        torch.cuda.empty_cache()
        return num_gpu_blocks
    
    @torch.inference_mode()
    def init_kvcache_and_swap(self, num_blocks: int):
        self.num_blocks = num_blocks

        # Initialize KV cache
        kvcache_shape = (
            self.model_config.num_layers,
            self.num_blocks,
            self.model_config.num_kv_heads,
            self.engine_config.block_size,
            self.model_config.head_dim
        )
        # Here we use torch.zeros instead of torch.empty, since that torch.empty
        # has the possibility to contain NaNs, which will cause the model to output NaNs.
        self.k_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")
        self.v_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")

        # Initialize KV swap space
        kvswap_shape = (
            self.model_config.num_layers,
            self.engine_config.num_cpu_blocks,
            self.model_config.num_kv_heads,
            self.engine_config.block_size,
            self.model_config.head_dim
        )
        self.k_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu", pin_memory=True)
        self.v_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu", pin_memory=True)

        # Initialize CPU QKV buffer
        self.q_cpu = torch.zeros((self.engine_config.max_batch_size, self.model_config.num_q_heads, self.model_config.head_dim), dtype=torch.float16, device="cpu", pin_memory=True)
        self.k_cpu = torch.zeros((self.engine_config.max_batch_size, self.model_config.num_kv_heads, self.model_config.head_dim), dtype=torch.float16, device="cpu", pin_memory=True)
        self.v_cpu = torch.zeros((self.engine_config.max_batch_size, self.model_config.num_kv_heads, self.model_config.head_dim), dtype=torch.float16, device="cpu", pin_memory=True)
        # We store float32 tensors for the output, but convert them to float16 when copying back to GPU
        self.o_cpu = torch.zeros((self.engine_config.max_batch_size, self.model_config.num_q_heads * self.model_config.head_dim), dtype=torch.float32, device="cpu", pin_memory=True)

        # Initialize block manager
        self.gpu_block_manager = BlockManager(
            "cuda",
            self.num_blocks,
            self.engine_config.max_seqs_in_block_table,
            self.engine_config.max_blocks_per_seq,
            self.engine_config.block_size
        )
        self.cpu_block_manager = BlockManager(
            "cpu",
            self.engine_config.num_cpu_blocks,
            self.engine_config.max_seqs_in_block_table,
            self.engine_config.max_blocks_per_seq,
            self.engine_config.block_size
        )

        self.swapper = Swapper(
            self.engine_config,
            self.k_cache,
            self.v_cache,
            self.k_swap,
            self.v_swap,
            self.gpu_block_manager,
            self.cpu_block_manager
        )

        for layer in self.transformer_layers:
            layer.set_meta_args(
                self.k_cache,
                self.v_cache,
                self.k_swap,
                self.v_swap,
                self.q_cpu,
                self.k_cpu,
                self.v_cpu,
                self.o_cpu,
                self.gpu_block_manager.block_table,
                self.cpu_block_manager.block_table,
                self.swapper
            )

    def _init_to_get_rotary(self):
        rope_scaling_factor = self.model_config.rope_scaling
        base = self.model_config.rope_theta
        max_position_embeddings = self.model_config.max_position_embeddings
        max_seq_len = max_position_embeddings * rope_scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, self.model_config.head_dim, 2, device="cuda", dtype=torch.float32) / self.model_config.head_dim))
        t = torch.arange(max_seq_len + 128, device="cuda", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float16)
        self._sin_cached = torch.sin(freqs).to(torch.float16)

    @torch.inference_mode()
    def _prepare_inputs(
        self,
        args: ModelForwardArgs,
    ) -> tuple[list[int], LlamaInferState]:
        """
        Gets input lists and reformats them into a single tensor.

        Also generates the infer_state object and allocates blocks for the sequences.
        """

        _flattened_input_ids = list(itertools.chain(*args.input_ids_list))

        batch_size = len(args.input_ids_list)
        _num_decoding_seqs = len(args.decoding_seq_lens_list)
        num_prefill_seqs = batch_size - _num_decoding_seqs
        gpu_num_decoding_seqs = _num_decoding_seqs - args.cpu_num_decoding_seqs
        _gpu_seq_end = batch_size - args.cpu_num_decoding_seqs

        _prefill_seq_lens_list = [len(seq) for seq in args.input_ids_list[:num_prefill_seqs]]
        _prefill_start_locs_with_end = [0] + list(itertools.accumulate(_prefill_seq_lens_list))

        position_indices = torch.tensor(
            [i for seq_len in _prefill_seq_lens_list for i in range(seq_len)] +
            [seq_len - 1 for seq_len in args.decoding_seq_lens_list],
            dtype=torch.int32,
            device="cuda"
        )

        # Select the seq_block_size
        #
        # Here we use a simple heuristic:
        #
        # In paged attention phase 1, the grid shape is (num_decoding_seqs, num_kv_heads, cdiv(max_decoding_len, seq_block_size))
        # and among these blocks, num_kv_heads * sum(cdiv(decoding_seq_lens, seq_block_size)) blocks are useful.
        # Thus we set seq_block_size to be the largest integer that satisfies
        #      num_kv_heads * sum(cdiv(decoding_seq_lens, seq_block_size)) >= 1024
        # to fully utilize the GPU. Here 1024 is a magic number (since most high-end
        # GPUs have ~128 SMs, so ~512 SMSPs. Since the decoding-stage attention
        # is mostly a memory-bound operation, I think 1024 is a reasonable number.)
        #
        # In practice, we use `decoding_seq_lens_sum/seq_block_size` to approximate
        # sum(cdiv(decoding_seq_lens, seq_block_size))

        _max_decoding_len = max(args.decoding_seq_lens_list + [0])
        seq_block_size = 2048
        decoding_seq_lens_sum = sum(args.decoding_seq_lens_list)
        while self.model_config.num_kv_heads*(decoding_seq_lens_sum/seq_block_size) < 1024 and seq_block_size//2 >= 64 and \
            _max_decoding_len / (seq_block_size//2) <= 128:
            seq_block_size //= 2
        num_seq_blocks = (_max_decoding_len + seq_block_size - 1) // seq_block_size

        _position_cos = self._cos_cached.index_select(0, position_indices)
        _position_sin = self._sin_cached.index_select(0, position_indices)

        infer_state = LlamaInferState(
            softmax_scale = self.model_config.head_dim ** -0.5,

            batch_size = batch_size,
            num_tokens = len(_flattened_input_ids),
            num_prefill_seqs = num_prefill_seqs,
            gpu_num_decoding_seqs = gpu_num_decoding_seqs,
            cpu_num_decoding_seqs = args.cpu_num_decoding_seqs,

            gpu_seq_ids = torch.tensor(args.seq_ids_list[:_gpu_seq_end], dtype=torch.int32, device="cuda"),
            cpu_seq_ids = torch.tensor(args.seq_ids_list[_gpu_seq_end:], dtype=torch.int32, device="cpu"),

            num_prefill_tokens = sum(_prefill_seq_lens_list),
            max_prefill_len = max(_prefill_seq_lens_list + [0]),

            prefill_seq_lens = torch.tensor(_prefill_seq_lens_list, dtype=torch.int32, device="cuda"),
            prefill_seq_start_locs = torch.tensor(_prefill_start_locs_with_end[:-1], dtype=torch.int32, device="cuda"),
            prefill_seq_start_locs_with_end = torch.tensor(_prefill_start_locs_with_end, dtype=torch.int32, device="cuda"),

            gpu_decoding_seq_lens = torch.tensor(args.decoding_seq_lens_list[:gpu_num_decoding_seqs], dtype=torch.int32, device="cuda"),
            cpu_decoding_seq_lens = torch.tensor(args.decoding_seq_lens_list[gpu_num_decoding_seqs:], dtype=torch.int32, device="cpu"),

            seq_block_size = seq_block_size,
            num_seq_blocks = num_seq_blocks,

            position_cos = _position_cos,
            position_sin = _position_sin,

            src_block_ids=[],
            dst_block_ids=[]
        )

        if not self.engine_config.ignore_kvcache:
            self.gpu_block_manager.allocate_blocks_for_seqs(
                infer_state.gpu_seq_ids,
                torch.cat([infer_state.prefill_seq_lens, infer_state.gpu_decoding_seq_lens])
            )

            self.cpu_block_manager.allocate_blocks_for_seqs(
                infer_state.cpu_seq_ids,
                infer_state.cpu_decoding_seq_lens
            )
        
        # Need to initiate swapping out after allocating blocks, so that blocks of the swapped-outs are not taken away
        if args.swap_out_seq_ids:
            infer_state.src_block_ids, infer_state.dst_block_ids = self.swapper.initiate_swap_out(args.swap_out_seq_ids)
        
        return _flattened_input_ids, infer_state


    @torch.inference_mode()
    def _forward(
        self,
        input_ids: torch.Tensor,    # [total_token_num]
        infer_state: LlamaInferState
    ) -> torch.Tensor:
        """
        Run a forward pass of the LlamaModel.
        """
        self.events.pf_record("frwd_s")

        # Main body of the forward pass
        input_embds = self.pre_layer.forward(input_ids)

        # Wait for swappings to finish
        torch.cuda.current_stream().wait_stream(self.cpu_communication_stream)

        self.events.pf_record("fstg_s")
        self.events.pf_record("mnbd_s")
        
        residual_buf = torch.zeros_like(input_embds)
        ffn_out = input_embds
        for layer in self.transformer_layers:
            layer.set_infer_states([infer_state])
            layer.set_buffers([input_embds], [residual_buf])
            ffn_out = layer.forward(ffn_out)

        self.events.pf_record("mnbd_e")
        self.events.pf_record("lstg_e")

        ffn_out += residual_buf
        output_tokens = self.post_layer.forward(ffn_out, infer_state)

        self.events.pf_record("frwd_e")

        if self.engine_config.monitor_performance:
            self.perf_results.append(ModelPerfResult(self.transformer_layers, self.events, False))

        return output_tokens
    
    @torch.inference_mode()
    def forward(
        self,
        args: ModelForwardArgs,
    ) -> list[int]:
        """
        Run a forward pass of the LlamaModel.

        This function is a wrapper of the `_forward` function. It prepares the infer_state
        and calls the `_forward` function.

        This function is intended to be called by the server.
        """

        flattened_input_ids, infer_state = self._prepare_inputs(args)

        return self._forward(
            torch.tensor(flattened_input_ids, dtype=torch.int32, device="cuda"),
            infer_state,
        ).tolist()
    
    @torch.inference_mode()
    def _forward_pipeline(
        self,
        input_ids: torch.Tensor,
        infer_states: list[LlamaInferState],
    ) -> torch.Tensor:
        """
        Run a forward pass of the LlamaModel in a pipelined manner.
        """

        self.events.pf_record("frwd_s")

        # input_embds would serve as a buffer for all attention outputs
        input_embeds = self.pre_layer.forward(input_ids)
        residual_buf = torch.zeros_like(input_embeds)
        input_embedss = torch.split(input_embeds, [infer_states[0].num_tokens, infer_states[1].num_tokens], dim=0)
        residual_bufs = torch.split(residual_buf, [infer_states[0].num_tokens, infer_states[1].num_tokens], dim=0)


        for layer in self.transformer_layers:
            layer.set_infer_states(infer_states)
            layer.set_buffers(input_embedss, residual_bufs)
            
        self.events.pf_record("fstg_s")

        # First stage of the forward pass
        q1, k1, v1 = self.transformer_layers[-1].forward_first_stage()

        self.events.pf_record("mnbd_s")

        # Main body of the forward pass
        # In every iteration, input_embds0 is updated to newer version of batch 0's attention output and 
        # q1, k1, v1 are updated to batch 1's newer version of q, k, v
        for layer in self.transformer_layers[:-1]:
            q1, k1, v1 = layer.forward_double(q1, k1, v1)

        self.events.pf_record("mnbd_e")

        # Last stage of the forward pass, also begin predicted swapping
        f0, f1 = self.transformer_layers[-1].forward_last_stage(q1, k1, v1)

        self.events.pf_record("lstg_e")

        f0 += residual_bufs[0]
        f1 += residual_bufs[1]
        output_tokens = self.post_layer.forward_double(f0, f1, infer_states)

        self.events.pf_record("frwd_e")

        if self.engine_config.monitor_performance:
            self.perf_results.append(ModelPerfResult(self.transformer_layers, self.events, True))

        return output_tokens

    @torch.inference_mode()
    def forward_pipeline(
        self,
        argss: list[ModelForwardArgs],
    ) -> list[int]:
        """
        Forward 2 sub-batches in a pipelined manner.
        """
        assert len(argss) == 2
        finput_ids0, infer_state0 = self._prepare_inputs(argss[0])
        finput_ids1, infer_state1 = self._prepare_inputs(argss[1])

        return self._forward_pipeline(
            torch.tensor(finput_ids0 + finput_ids1, dtype=torch.int32, device="cuda"),
            [infer_state0, infer_state1]
        ).tolist()
        
    @torch.inference_mode()
    def swap_in_seqs(
        self,
        seq_ids_list: list[int]
    ):
        """
        Swap in (move blocks from CPU to GPU) the specified sequences.
        """
        if not seq_ids_list:
            return
        src_block_ids, dst_block_ids = self.swapper.initiate_swap_in(seq_ids_list)
        with torch.cuda.stream(self.cpu_communication_stream):
            for layer_id in range(len(self.transformer_layers)):
                self.swapper.swap_in_blocks(layer_id, src_block_ids, dst_block_ids)
    
    @torch.inference_mode()
    def swap_out_seqs(
        self,
        seq_ids_list: list[int]
    ):
        """
        Swap out (move blocks from GPU to CPU) the specified sequences.
        """
        if not seq_ids_list:
            return
        src_block_ids, dst_block_ids = self.swapper.initiate_swap_out(seq_ids_list)
        with torch.cuda.stream(self.cpu_communication_stream):
            for layer_id in range(len(self.transformer_layers)):
                self.swapper.swap_out_blocks(layer_id, src_block_ids, dst_block_ids)

    @torch.inference_mode()
    def free_seqs_resources(self, seq_ids_list: list[int]):
        """
        Free the resources of the specified sequences.
        """
        if not seq_ids_list:
            return
        gpu_seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        cpu_seq_ids = gpu_seq_ids.to("cpu")
        self.gpu_block_manager.free_blocks_for_seqs(gpu_seq_ids)
        self.cpu_block_manager.free_blocks_for_seqs(cpu_seq_ids)

    def turn_on_perf_monitor(self):
        self.engine_config.monitor_performance = True

    def flush_perf_results_and_turn_off_perf_monitor(self):
        self.engine_config.monitor_performance = False
        ret = self.perf_results
        self.perf_results = []
        return ret
        
