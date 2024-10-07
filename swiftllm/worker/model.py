import json

import numpy as np
import torch

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import load_weights
from swiftllm.worker.block_manager import Swapper
from swiftllm.structs import SubBatch

from .layers.pre_layer import LlamaPreLayer
from .layers.transformer_layer import LlamaTransformerLayer
from .layers.post_layer import LlamaPostLayer

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
    fields_to_dump = [
        "avg_linr_time",
        "avg_pref_time",
        "avg_gdec_time",
        "avg_cdec_time",
        "avg_lnch_time"
    ]
    def __init__(
        self, 
        layers: list[LlamaTransformerLayer],
        model_events: ModelEvents,
        use_pipline: bool
    ):
        torch.cuda.synchronize() # Ensure all events are recorded
        if use_pipline:
            self.linr_times = np.array([[layer.events[i].linr_time for layer in layers[:-1]] for i in range(2)])
            self.pref_times = np.array([[layer.events[i^1].pref_time for layer in layers] for i in range(2)])
            self.gdec_times = np.array([[layer.events[i^1].gdec_time for layer in layers] for i in range(2)])
            self.cdec_times = np.array([[layer.events[i^1].cdec_time for layer in layers] for i in range(2)])
            self.lnch_times = np.array([[layer.events[i].lnch_time for layer in layers] for i in range(2)])
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

        self.avg_linr_time = self.linr_times.mean(-1)
        self.avg_pref_time = self.pref_times.mean(-1)
        self.avg_gdec_time = self.gdec_times.mean(-1)
        self.avg_cdec_time = self.cdec_times.mean(-1)
        self.avg_lnch_time = self.lnch_times.mean(-1)

    def __repr__(self):
        return json.dumps({
            field: getattr(self, field).tolist() for field in self.fields_to_dump
        }, indent=2)

    @staticmethod
    def mean(results: list["ModelPerfResult"], name: str) -> float:
        """
        Compute the average of a field in a list of ModelPerfResult objects.
        """
        ret = np.array([getattr(result, name) for result in results]).mean(0).tolist()
        return ret

    @staticmethod
    def mean_all(results: list["ModelPerfResult"]) -> dict[str, float]:
        """
        Compute the average of all fields in a list of ModelPerfResult objects.
        """
        return {
            field: ModelPerfResult.mean(results, field) for field in ModelPerfResult.fields_to_dump
        }



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

        # Swapper
        self.swapper = None

        # Cached cos and sin values for RoPE
        self.cos_cached = None
        self.sin_cached = None

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
    def init_kvcache_and_swap(self, num_blocks: int):
        """
        Initialize the key-value cache on both CPU and GPU.
        """
        self.engine_config.num_gpu_blocks = num_blocks

        self.swapper = Swapper(self.engine_config, self.model_config)

        for layer in self.transformer_layers:
            layer.set_swapper(self.swapper)

    def _init_to_get_rotary(self):
        rope_scaling_factor = self.model_config.rope_scaling
        base = self.model_config.rope_theta
        max_position_embeddings = self.model_config.max_position_embeddings
        max_seq_len = max_position_embeddings * rope_scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, self.model_config.head_dim, 2, device="cuda", dtype=torch.float32) / self.model_config.head_dim))
        t = torch.arange(max_seq_len + 128, device="cuda", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self.cos_cached = torch.cos(freqs).to(torch.float16)
        self.sin_cached = torch.sin(freqs).to(torch.float16)

    def _prepare_inputs(
        self,
        batch: SubBatch
    ):
        """
        Prepare the inputs for the forward pass.
        """
        batch.set_model_forward_args()

        batch.softmax_scale = self.model_config.head_dim ** -0.5

        position_indices = torch.tensor(
            sum([list(range(req.prompt_len)) for req in batch.pref_reqs], []) + \
                [req.seq_len - 1 for req in batch.deco_reqs],
            dtype=torch.int32, device='cuda'
        )
        batch.position_cos = self.cos_cached[position_indices]
        batch.position_sin = self.sin_cached[position_indices]

        sum_gdec_toks = sum(batch.gdec_seq_lens_list)
        max_gdec_toks = max(batch.gdec_seq_lens_list, default=0)
        seq_block_size = 2048
        num_kv_heads = self.model_config.num_kv_heads
        while num_kv_heads*(sum_gdec_toks/seq_block_size) < 1024 and seq_block_size//2 >= 64 and \
            max_gdec_toks / (seq_block_size//2) <= 128:
            seq_block_size //= 2
        batch.seq_block_size = seq_block_size
        batch.num_seq_blocks = (max_gdec_toks + seq_block_size - 1) // seq_block_size

        if self.swapper is not None:
            self.swapper.gpu_block_manager.allocate_blocks_for_seqs(
                batch.prgd_seq_ids, batch.prgd_seq_lens
            )
            self.swapper.cpu_block_manager.allocate_blocks_for_seqs(
                batch.cdec_seq_ids, batch.cdec_seq_lens
            )
        
            if batch.cprf_reqs:
                batch.src_blk_ids, batch.dst_blk_ids = self.swapper.initiate_swap_out(
                    [req.request_id for req in batch.cprf_reqs]
                )
        else:
            assert not batch.cprf_reqs and not batch.gdec_reqs and not batch.cdec_reqs

    @torch.inference_mode()
    def forward(
        self,
        batch: SubBatch
    ):
        """
        Run a forward pass of the LlamaModel.
        """
        self._prepare_inputs(batch)

        self.events.pf_record("frwd_s")

        # Main body of the forward pass
        # start = time.perf_counter()
        input_embds = self.pre_layer.forward(batch.input_token_ids)

        # Wait for swappings to finish
        torch.cuda.current_stream().wait_stream(self.cpu_communication_stream)

        self.events.pf_record("fstg_s")
        self.events.pf_record("mnbd_s")
        
        residual_buf = torch.zeros_like(input_embds)
        ffn_out = input_embds
        for layer in self.transformer_layers:
            layer.set_batches([batch])
            layer.set_buffers([input_embds], [residual_buf])
            ffn_out = layer.forward(ffn_out)

        self.events.pf_record("mnbd_e")
        self.events.pf_record("lstg_e")

        ffn_out += residual_buf
        output_tokens = self.post_layer.forward(ffn_out, batch)

        self.events.pf_record("frwd_e")
        # duration = time.perf_counter() - start
        # print(f"Forward time: {duration*1000:.2f}ms")

        batch.update_output(output_tokens)

        if self.engine_config.monitor_performance:
            self.perf_results.append(ModelPerfResult(self.transformer_layers, self.events, False))

    
    @torch.inference_mode()
    def forward_pipeline(
        self,
        batches: list[SubBatch]
    ):
        """
        Run a forward pass of the LlamaModel in a pipelined manner.
        """

        for batch in batches:
            self._prepare_inputs(batch)

        self.events.pf_record("frwd_s")

        # input_embds would serve as a buffer for all attention outputs
        input_embedss = self.pre_layer.forward(sum([b.input_token_ids for b in batches], []))
        residual_bufs = torch.zeros_like(input_embedss)
        s0 = batches[0].metadata.s
        s1 = batches[1].metadata.s
        input_embedss = torch.split(input_embedss, [s0, s1], dim=0)
        residual_bufs = torch.split(residual_bufs, [s0, s1], dim=0)


        for layer in self.transformer_layers:
            layer.set_batches(batches)
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
        output_tokens = self.post_layer.forward_double(f0, f1, batches)

        self.events.pf_record("frwd_e")

        x0 = batches[0].metadata.x
        batches[0].update_output(output_tokens[:x0])
        batches[1].update_output(output_tokens[x0:])

        if self.engine_config.monitor_performance:
            self.perf_results.append(ModelPerfResult(self.transformer_layers, self.events, True))
        
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
        self.swapper.gpu_block_manager.free_blocks_for_seqs(gpu_seq_ids)
        self.swapper.cpu_block_manager.free_blocks_for_seqs(cpu_seq_ids)

    def turn_on_perf_monitor(self):
        """
        Turn on performance monitoring.
        """
        self.engine_config.monitor_performance = True

    def flush_perf_results_and_turn_off_perf_monitor(self):
        """
        Flush the performance results and turn off performance monitoring.
        """
        self.engine_config.monitor_performance = False
        ret = self.perf_results
        self.perf_results = []
        return ret
        
