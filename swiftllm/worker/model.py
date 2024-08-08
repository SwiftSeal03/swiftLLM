import itertools
import math
import dataclasses
from concurrent.futures import ThreadPoolExecutor

import torch

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import load_weights
from swiftllm.worker.block_manager import BlockManager
from swiftllm.utils import GB
import swiftllm_c

from .layers.pre_layer import LlamaPreLayer
from .layers.transformer_layer import LlamaTransformerLayer, KVCacheArgs
from .layers.post_layer import LlamaPostLayer
from .infer_state import LlamaInferState

@dataclasses.dataclass
class ModelForwardArgs:
    input_ids_list: list[list[int]]
    seq_ids_list: list[int]
    decoding_seq_lens_list: list[int]
    cpu_num_decoding_seqs: int = 0
    ignore_kvcache: bool = False

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

        # Block manager
        self.cpu_block_manager = self.gpu_block_manager = None
        self.kvargs = None

        if engine_config.library_path:
            torch.ops.load_library(engine_config.library_path)
        
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
        prefilling_stream = torch.cuda.Stream()
        decoding_piggyback_stream = torch.cuda.Stream()
        cpu_communication_stream = torch.cuda.Stream()
        self.pre_layer = LlamaPreLayer(self.model_config, self.weight)
        self.transformer_layers = [
            LlamaTransformerLayer(
                self.model_config,
                self.engine_config,
                self.weight.layers[layer_id],
                self.weight.layers[layer_id + 1] if layer_id + 1 < self.model_config.num_layers else None,
                prefilling_stream,
                decoding_piggyback_stream,
                cpu_communication_stream,
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
        _ = self.forward(input_ids, seq_ids, [], ignore_kvcache=True)
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
            self.num_blocks,
            self.model_config.num_layers,
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
            self.engine_config.num_cpu_blocks,
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.engine_config.block_size,
            self.model_config.head_dim
        )
        self.k_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu")
        self.v_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu")

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
        args: ModelForwardArgs
    ):
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

            position_cos = self._cos_cached[position_indices],
            position_sin = self._sin_cached[position_indices],

            ignore_kvcache = args.ignore_kvcache
        )

        if not args.ignore_kvcache:
            self.gpu_block_manager.allocate_blocks_for_seqs(
                infer_state.gpu_seq_ids,
                torch.cat([infer_state.prefill_seq_lens, infer_state.gpu_decoding_seq_lens])
            )

            self.cpu_block_manager.allocate_blocks_for_seqs(
                infer_state.cpu_seq_ids,
                infer_state.cpu_decoding_seq_lens
            )
        
        return _flattened_input_ids, infer_state


    @torch.inference_mode()
    def _forward(
        self,
        input_ids: torch.Tensor,    # [total_token_num]
        infer_state: LlamaInferState,
    ) -> torch.Tensor:
        """
        Run a forward pass of the LlamaModel.
        """
        if self.engine_config.monitor_performance:
            forward_s_event = torch.cuda.Event(enable_timing=True)
            forward_s_event.record()

        kvargs = KVCacheArgs(
            self.k_cache,
            self.v_cache,
            self.k_swap,
            self.v_swap,
            gpu_block_table = self.gpu_block_manager.block_table if not infer_state.ignore_kvcache else None,
            cpu_block_table = self.cpu_block_manager.block_table if not infer_state.ignore_kvcache else None
        )

        # Main body of the forward pass
        input_embds = self.pre_layer.forward(input_ids)
        residual_buf = torch.zeros_like(input_embds)
        for layer in self.transformer_layers:
            input_embds = layer.forward(
                input_embds,
                residual_buf,
                kvargs,
                infer_state,
            )
        input_embds += residual_buf
        output_tokens = self.post_layer.forward(input_embds, infer_state)

        if self.engine_config.monitor_performance:
            forward_e_event = torch.cuda.Event(enable_timing=True)
            forward_e_event.record()
            torch.cuda.synchronize()
            attn_total_time = sum(layer.attn_s_event.elapsed_time(layer.attn_e_event) for layer in self.transformer_layers)
            linr_total_time = forward_s_event.elapsed_time(forward_e_event) - attn_total_time
            print(f"[Model.forward] Linear time: {linr_total_time:.3f} ms, Attention time: {attn_total_time:.3f} ms")

        return output_tokens
    
    @torch.inference_mode()
    def forward(
        self,
        args: ModelForwardArgs
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
            infer_state
        ).tolist()
    
    @torch.inference_mode()
    def _forward_pipeline(
        self,
        input_ids0: torch.Tensor,
        input_ids1: torch.Tensor,
        infer_state0: LlamaInferState,
        infer_state1: LlamaInferState
    ) -> torch.Tensor:
        """
        Run a forward pass of the LlamaModel in a pipelined manner.
        """
        assert infer_state0.ignore_kvcache == False and infer_state1.ignore_kvcache == False

        kvargs = KVCacheArgs(
            self.k_cache,
            self.v_cache,
            self.k_swap,
            self.v_swap,
            self.gpu_block_manager.block_table,
            self.cpu_block_manager.block_table
        )

        # input_embds would serve as a buffer for all attention outputs
        input_embds = self.pre_layer.forward(torch.cat([input_ids0, input_ids1]))
        residual_buf = torch.zeros_like(input_embds)
        input_embds0, input_embds1 = torch.split(input_embds, infer_state0.num_tokens, dim=0)
        residual_buf0, residual_buf1 = torch.split(residual_buf, infer_state0.num_tokens, dim=0)

        q1, k1, v1 = self.transformer_layers[0].forward_first_stage(
            input_embds0, input_embds1, residual_buf0, residual_buf1,
            kvargs, infer_state0, infer_state1
        )

        # Main body of the forward pass
        # In every iteration, input_embds0 is updated to newer version of batch 0's attention output and 
        # q1, k1, v1 are updated to batch 1's newer version of q, k, v
        for layer in self.transformer_layers[:-1]:
            q1, k1, v1 = layer.forward_double(
                q1, k1, v1, 
                input_embds0, input_embds1, residual_buf0, residual_buf1,
                kvargs, infer_state0, infer_state1
            )

        f0, f1 = self.transformer_layers[-1].forward_last_stage(
            q1, k1, v1, input_embds0, input_embds1, residual_buf0, residual_buf1,
            kvargs, infer_state0, infer_state1
        )

        f0 += residual_buf0
        f1 += residual_buf1

        output_tokens = self.post_layer.forward_double(f0, f1, infer_state0, infer_state1)
        return output_tokens

    @torch.inference_mode()
    def forward_pipeline(
        self,
        args0: ModelForwardArgs,
        args1: ModelForwardArgs
    ) -> list[int]:
        """
        Forward 2 sub-batches in a pipelined manner.
        """
        finput_ids0, infer_state0 = self._prepare_inputs(args0)
        finput_ids1, infer_state1 = self._prepare_inputs(args1)

        return self._forward_pipeline(
            torch.tensor(finput_ids0, dtype=torch.int32, device="cuda"),
            torch.tensor(finput_ids1, dtype=torch.int32, device="cuda"),
            infer_state0,
            infer_state1
        ).tolist()

    def _swap(
        self,
        seq_ids_list: list[int],
        is_swap_in: bool
    ):
        src_block_manager = self.cpu_block_manager if is_swap_in else self.gpu_block_manager
        dst_block_manager = self.gpu_block_manager if is_swap_in else self.cpu_block_manager
        src_seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device=src_block_manager.device_name)
        dst_seq_ids = src_seq_ids.to(dst_block_manager.device_name)
        src_seq_lengths = src_block_manager.get_num_allocated_blocks(src_seq_ids) * self.engine_config.block_size
        dst_seq_lengths = src_seq_lengths.to(dst_block_manager.device_name)
        src_block_ids = src_block_manager.gather_allocated_blocks_and_free(src_seq_ids)
        dst_block_ids = dst_block_manager.allocate_blocks_for_seqs(dst_seq_ids, dst_seq_lengths)
        swiftllm_c.swap_blocks(
            src_block_ids.tolist(),
            dst_block_ids.tolist(),
            is_swap_in,

            self.k_cache, self.v_cache,
            self.k_swap, self.v_swap
        )
        
    @torch.inference_mode()
    def swap_in_seqs(
        self,
        seq_ids_list: list[int]
    ):
        """
        Swap in (move blocks from CPU to GPU) the specified sequences.
        """
        self._swap(seq_ids_list, True)
    
    @torch.inference_mode()
    def swap_out_seqs(
        self,
        seq_ids_list: list[int]
    ):
        """
        Swap out (move blocks from GPU to CPU) the specified sequences.
        """
        self._swap(seq_ids_list, False)

    @torch.inference_mode()
    def free_seqs_resources(self, seq_ids_list: list[int]):
        """
        Free the resources of the specified sequences.
        """
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        self.gpu_block_manager.free_blocks_for_seqs(seq_ids)
        self.cpu_block_manager.free_blocks_for_seqs(seq_ids)
