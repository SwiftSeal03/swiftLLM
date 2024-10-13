"""
LlamaModel - A Llama model that can be used for inference.

If performance monitoring is enabled, the model will record performance results.
"""

import json
import itertools

import numpy as np
import torch
import torch.distributed as dist
import ray

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import load_weights
from swiftllm.worker.block_swapper import Swapper
from swiftllm.structs import Request, SubBatch

from .layers.pre_layer import LlamaPreLayer
from .layers.transformer_layer import LlamaTransformerLayer
from .layers.post_layer import LlamaPostLayer

class ModelEvents:
    """
    ModelEvents - A class that represents the GPU events of a forward pass of a model.
    """

    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        self.frwd_s = torch.cuda.Event(enable_timing=True)
        self.fstg_s = torch.cuda.Event(enable_timing=True)
        self.mnbd_s = torch.cuda.Event(enable_timing=True)
        self.mnbd_e = torch.cuda.Event(enable_timing=True)
        self.lstg_e = torch.cuda.Event(enable_timing=True)
        self.frwd_e = torch.cuda.Event(enable_timing=True)

    def pf_record(self, name:str):
        """
        Record the event with the given name if performance monitoring is enabled.
        """
        if self.engine_config.monitor_performance:
            getattr(self, name).record()


class ModelPerfResult:
    """
    ModelPerfResult - A class that represents the performance results of a forward pass of a model.
    """

    # pylint: disable=too-many-instance-attributes
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
    for the control plane (the executor) to send commands.

    To initialize, please:
    - call __init__()
    - call load_weights()
    - call init_kvcache_and_swap()
    """

    @torch.inference_mode()
    def __init__(
        self,
        engine_config: EngineConfig,
        model_config: LlamaModelConfig,
        rank: int=0
    ):
        """
        Initialize the LlamaModel.

        Loads model weights, inits RoPE cache, and initializes layers.

        The block tables are not initialized here, as num_blocks is not known yet.
        """
        self.engine_config = engine_config
        self.model_config = model_config
        self.rank = rank

        # CPU kernel library & stream
        if engine_config.library_path:
            torch.ops.load_library(engine_config.library_path)        
        self.cpu_communication_stream = torch.cuda.Stream()

        # Load weights
        self.weight = load_weights(
            self.model_config,
            torch.float16,
            self.engine_config.model_path,
            self.engine_config.use_dummy
        )

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

        # Initialize rotary embeddings
        self.cos_cached = None
        self.sin_cached = None
        self._init_to_get_rotary()

        # Swapper
        self.swapper = None

        # List of performance results, unused if monitor_performance is False
        self.perf_results = []
        self.events = ModelEvents(engine_config)
    

    @torch.inference_mode()
    def init_kvcache_and_swap(self):
        """
        Initialize the key-value cache on both CPU and GPU.
        """
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


    def _prepare_inputs(self, batch: SubBatch):
        """
        Prepare the batch for the forward pass. 
        
        Most work should be already be done by the control plane. We only need to prepare GPU specific data, including:
            1. seq_ids and seq_lens
            2. position_cos and position_sin
        """
        batch.prgd_seq_ids = torch.tensor(batch.seq_ids_list[:batch.num_prgds], dtype=torch.int32, device='cuda')
        batch.prgd_seq_lens = torch.tensor(batch.seq_lens_list[:batch.num_prgds], dtype=torch.int32, device='cuda')
        batch.pref_st_locs_we = torch.tensor(
            [0] + list(itertools.accumulate(batch.seq_lens_list[:batch.num_prefs])), 
            dtype=torch.int32, device='cuda'
        )

        position_indices = torch.tensor(
            sum([list(range(req.prompt_len)) for req in batch.all_reqs[:batch.num_prefs]], []) + \
                [req.seq_len - 1 for req in batch.all_reqs[batch.num_prefs:]],
            dtype=torch.int32, device='cuda'
        )

        batch.position_cos = self.cos_cached[position_indices]
        batch.position_sin = self.sin_cached[position_indices]


    def _forward_sequential(self, batch: SubBatch) -> list[int]:
        """
        Run a forward pass of the LlamaModel in a sequential manner.

        Returns the output tokens.
        """
        self.events.pf_record("frwd_s")

        # Main body of the forward pass
        # start = time.perf_counter()
        input_embds = self.pre_layer.forward(Request.get_input_tokens(batch.all_reqs))

        # Wait for swappings to finish
        torch.cuda.current_stream().wait_stream(self.cpu_communication_stream)

        self.events.pf_record("fstg_s")
        self.events.pf_record("mnbd_s")
        
        residual_buf = torch.zeros_like(input_embds)
        ffn_out = input_embds
        for layer in self.transformer_layers:
            layer.set_batches_and_buffers([batch], [input_embds], [residual_buf])
            ffn_out = layer.forward(ffn_out)

        self.events.pf_record("mnbd_e")
        self.events.pf_record("lstg_e")

        ffn_out += residual_buf
        output_tokens = self.post_layer.forward(ffn_out, batch).tolist()

        self.events.pf_record("frwd_e")
        # duration = time.perf_counter() - start
        # print(f"Forward time: {duration*1000:.2f}ms")

        if self.engine_config.monitor_performance:
            self.perf_results.append(ModelPerfResult(self.transformer_layers, self.events, False))
        
        return output_tokens

    
    def _forward_pipeline(self, batches: list[SubBatch]) -> list[int]:
        """
        Run a forward pass of the LlamaModel in a pipelined manner.

        Returns the concatenated output tokens.
        """
        self.events.pf_record("frwd_s")

        # input_embds would serve as a buffer for all attention outputs
        input_embedss = self.pre_layer.forward(sum([Request.get_input_tokens(b.all_reqs) for b in batches], []))
        residual_bufs = torch.zeros_like(input_embedss)
        iter_widths = [b.iter_width for b in batches]
        input_embedss = torch.split(input_embedss, iter_widths, dim=0)
        residual_bufs = torch.split(residual_bufs, iter_widths, dim=0)

        for layer in self.transformer_layers:
            layer.set_batches_and_buffers(batches, input_embedss, residual_bufs)
            
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
        output_tokens = self.post_layer.forward_double(f0, f1, batches).tolist()

        self.events.pf_record("frwd_e")

        if self.engine_config.monitor_performance:
            self.perf_results.append(ModelPerfResult(self.transformer_layers, self.events, True))
        
        return output_tokens
    
    
    @torch.inference_mode()
    def _forward_batches(self, batches: list[SubBatch]) -> list[int]:
        """
        Run a forward pass of the LlamaModel.

        Requires that blocks of requests are allocated and the block tables are set.

        Returns the output tokens.
        """
        for batch in batches:
            self._prepare_inputs(batch)

        if len(batches) == 1:
            return self._forward_sequential(batches[0])
        
        if len(batches) == 2:
            return self._forward_pipeline(batches)
        
        raise ValueError("Invalid number of batches")


    def do_one_iteration(
        self,
        batches: list[SubBatch],
        mappings: tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]],
        swappings: tuple[list[int], list[int]],
        is_swap_out: bool = False
    ) -> list[int]:
        """
        Run a forward iteration of the LlamaModel, with the following steps:
            1. modify the block-tables according to the mappings
            2. swap the specified sequences by direction given by is_swap_out and physical IDs given by swappings
            3. run the forward pass of the model

        Returns the output tokens.
        """
        self.swapper.set_block_tables(mappings)

        if swappings[0]:
            with torch.cuda.stream(self.cpu_communication_stream):
                for layer_id in range(self.model_config.num_layers):
                    self.swapper.swap_blocks(*swappings, is_swap_out, layer_id, layer_id)

        return self._forward_batches(batches)
    

    def turn_on_perf_monitor(self):
        """
        Turn on performance monitoring.
        """
        self.engine_config.monitor_performance = True


    def turn_off_perf_monitor_and_flush_results(self):
        """
        Flush the performance results and turn off performance monitoring.
        """
        self.engine_config.monitor_performance = False
        ret = self.perf_results
        self.perf_results = []
        return ret
        

@ray.remote(num_gpus=1)
class RemoteLlamaModel(LlamaModel):
    """
    RemoteLlamaModel - A remote Llama model that can be used for inference.
    """
    
    @torch.inference_mode()
    def __init__(
        self,
        engine_config: EngineConfig,
        model_config: LlamaModelConfig,
        rank: int=0
    ):
        """
        Initialize the RemoteLlamaModel.
        """
        
        dist.init_process_group(
            backend="nccl", 
            world_size=engine_config.tensor_parallel_degree, 
            rank=rank
        )
        super().__init__(engine_config, model_config, rank)

