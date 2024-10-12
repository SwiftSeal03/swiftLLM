"""
The main engine of the server
"""

import time
import asyncio
import functools
from typing import AsyncGenerator

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.model import LlamaModel
from swiftllm.worker.profiler import ModelProfiler
from swiftllm.utils import GB
from swiftllm.structs import Request, RawRequest, StepOutput, SubBatch

from swiftllm.server.tokenization_engine import TokenizationEngine
from swiftllm.server.scheduler import Scheduler
from swiftllm.server.block_manager import BlockManager

class Engine:
    """
    Offline version of the engine, need to tokenize manually
    """

    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        self.model_config = LlamaModelConfig.load_from_model_path(engine_config.model_path)
        self.initialized = False

        assert engine_config.max_prefill_tokens <= engine_config.max_tokens_in_batch, \
            f"max_prefill_tokens {engine_config.max_prefill_tokens} exceeds max_tokens_in_batch {engine_config.max_tokens_in_batch}"
        assert engine_config.max_batch_size <= engine_config.max_tokens_in_batch, \
            f"max_batch_size {engine_config.max_batch_size} exceeds max_tokens_in_batch {engine_config.max_tokens_in_batch}"
        assert engine_config.max_batch_size <= engine_config.max_seqs_in_block_table, \
            f"max_batch_size {engine_config.max_batch_size} exceeds max_seqs_in_block_table {engine_config.max_seqs_in_block_table}"

        # The following fields will be created on `init_model()`
        self.model = None
        self.event_loop = None
        self.profiler = None
        self.block_manager = None

    
    def initialize(self):
        """
        Initialize the engine
        """
        print("[Engine] Initializing model...")
        self.model = LlamaModel(self.engine_config)

        print("[Engine] Loading weights...")
        self.model.load_weights()

        print("[Engine] Profiling kv blocks...")
        self.profiler = ModelProfiler(self.model)
        # self.profiler.profile_num_blocks()
        num_gpu_blocks = self.engine_config.num_gpu_blocks
        num_cpu_blocks = self.engine_config.num_cpu_blocks
        block_size_bytes = self.engine_config.block_size*self.model_config.get_kvslot_size()
        print(f"[Engine] Number of GPU blocks: {num_gpu_blocks} ({num_gpu_blocks*block_size_bytes/GB:.2f} GB)")
        print(f"[Engine] Number of CPU blocks: {num_cpu_blocks} ({num_cpu_blocks*block_size_bytes/GB:.2f} GB)")

        print("[Engine] Initializing block manager...")
        self.block_manager = BlockManager(self.engine_config)

        print("[Engine] Allocating kv cache and swap...")
        self.model.init_kvcache_and_swap()

        print("[Engine] Model initialized")
        self.initialized = True


    def prepare_model_forward_args(
        self,
        batches: list[SubBatch], 
        cur_swap_out: list[Request],
        cur_swap_in: list[Request]
    ) -> tuple[tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]], tuple[list[int], list[int]], bool]:
        """
        Prepare KV cache and swapping related arguments for the model forward pass

        Requires either cur_swap_out or cur_swap_in to be empty

        Return a triple, the first element is yet another tuple of mappings:

            (GPU block VIDs, GPU block PIDs), (CPU block VIDs, CPU block PIDs);

        The second element is a tuple of lists of block IDs:

            (source block PIDs, destination block PIDs);

        The third element is a boolean indicating whether it's a swap out operation
        """
        mappings = (([], []), ([], [])) # (GPU, CPU)
        swappings = ([], []) # (swap in: CPU -> GPU / swap out: GPU -> CPU)
        for batch in batches:
            batch.set_model_forward_args(self.model_config)
            (gv, gp), (cv, cp) = self.block_manager.alloc_blocks_for_batch(batch)
            mappings[0][0].extend(gv)
            mappings[0][1].extend(gp)
            mappings[1][0].extend(cv)
            mappings[1][1].extend(cp)
        
        for batch in batches:
            sp, dv, dp = self.block_manager.initiate_swap(
                batch.cprf_reqs, 
                is_swap_out=True, use_itm=self.engine_config.extra_layer_for_cprf
            )
            batch.src_blk_ids = sp
            batch.dst_blk_ids = dp
            mappings[1][0].extend(dv)
            mappings[1][1].extend(dp)

        is_swap_out = bool(cur_swap_out)

        sp, dv, dp = self.block_manager.initiate_swap(cur_swap_out or cur_swap_in, is_swap_out)
        mappings[is_swap_out][0].extend(dv)
        mappings[is_swap_out][1].extend(dp)
        swappings[0].extend(sp)
        swappings[1].extend(dp)

        return mappings, swappings, is_swap_out


    def update_outputs_and_free_finished_reqs(self, batches: list[SubBatch], output_token_ids: list[int]):
        """
        Called at the end of each iteration
        """
        all_reqs = sum([b.all_reqs for b in batches], [])
        finished_reqs = Request.update_output(all_reqs, output_token_ids)
        self.block_manager.free_blocks_of_requests(finished_reqs)


    def step(self, batches: list[SubBatch], cur_swap_out: list[Request]=None, cur_swap_in: list[Request]=None):
        """
        Perform a step of the engine
        """
        forward_args = self.prepare_model_forward_args(batches, cur_swap_out or [], cur_swap_in or [])
        output_token_ids = self.model.do_one_iteration(batches, *forward_args)
        self.update_outputs_and_free_finished_reqs(batches, output_token_ids)



class AsyncEngine(Engine):
    """
    The main engine of the server
    """

    def __init__(self, engine_config: EngineConfig):
        super().__init__(engine_config)

        # The following fields will be created on `init_model()`
        self.scheduler = None
        self.tokenization_engine = None

        self.untokenized_raw_requests: list[tuple[Request, str]] = []
        self.itr_end_times = []
        self.ntoken_of_itr = []

    async def _run_on_model_async(self, func, *args, **kwargs):
        """
        Run a function on the model asynchronously, and return the result
        """
        func_partial = functools.partial(func, *args, **kwargs)
        return await self.event_loop.run_in_executor(None, func_partial)

    async def initialize_async(self):
        """
        Initialize the engine
        """
        self.event_loop = asyncio.get_event_loop()

        super().initialize()

        print("[Engine] Initializing performance table...")
        self.profiler.init_profile_tables(self.block_manager)

        print("[Engine] Initializing scheduler...")
        self.scheduler = Scheduler(self.model, self.profiler.pp)

        print("[Engine] Initializing tokenization engine...")
        # pylint: disable=no-member
        self.tokenization_engine = TokenizationEngine.remote(self.engine_config)

        print("[Engine] Model initialized")
        self.initialized = True

    async def add_request_and_stream(self, raw_request: RawRequest) -> AsyncGenerator[StepOutput, None]:
        """
        Add a raw request to the engine and stream the output of the request (streaming mode)
        """
        request = Request(raw_request)
        self.untokenized_raw_requests.append((request, raw_request.prompt))
        while True:
            step_output = await request.output_q.get()
            yield step_output
            request.output_q.task_done()
            if step_output.request.is_finished():
                break
    
    async def add_request_and_wait(self, raw_request: RawRequest) -> tuple[Request, list[int]]:
        """
        Add a raw request to the engine and wait for the completion (non-streaming mode)

        Return the output token ids
        """
        request = Request(raw_request)
        self.untokenized_raw_requests.append((request, raw_request.prompt))
        await request.finished_event.wait()
        return (request, request.output_token_ids)

    async def _tokenize_raw_request_event_loop(self):
        """
        Event loop for tokenizing raw requests
        """
        while True:
            if not self.untokenized_raw_requests:
                # No new raw requests, sleep for a bit
                await asyncio.sleep(0.002)
                continue

            # Tokenize the raw request in batch
            cur_untokenized_raw_requests = self.untokenized_raw_requests
            self.untokenized_raw_requests = []

            prompts = [prompt for _, prompt in cur_untokenized_raw_requests]
            prompt_token_ids = await self.tokenization_engine.batched_tokenize.remote(prompts)

            new_requests = []
            for (request, _), prompt_token_id in zip(cur_untokenized_raw_requests, prompt_token_ids):
                request.prompt_token_ids = prompt_token_id
                request.prompt_len = len(prompt_token_id)
                new_requests.append(request)

            self.scheduler.on_requests_arrival(new_requests)
            await asyncio.sleep(0.001)  # yield the event loop

    
    async def _main_event_loop(self):
        """
        Event loop for forwarding the model
        """
        while True:
            # Get the next batch from the scheduler
            start = time.perf_counter()

            batches, cur_swap_out, cur_swap_in = self.scheduler.get_next_batch()
            # print(f"Got {len(batches)} batches, {len(cur_swap_out)} swap out, {len(cur_swap_in)} swap in")
            assert all(batches), "Batch shouldn't be empty"
            assert not (cur_swap_out and cur_swap_in), "Swap out and swap in should be mutually exclusive"
            assert len(batches) <= 2, "The number of batches should be at most 2"
            if not (batches or cur_swap_in or cur_swap_out):
                # Nothing to do, sleep for a bit
                await asyncio.sleep(0.005)
                continue

            # Prepare model forward arguments
            forward_args = self.prepare_model_forward_args(batches, cur_swap_out, cur_swap_in)
            prepare_end = time.perf_counter()
            
            # Forward the model
            output_token_ids = await self._run_on_model_async(self.model.do_one_iteration, batches, *forward_args)

            # Deal with output tokens
            self.update_outputs_and_free_finished_reqs(batches, output_token_ids)
            self.scheduler.remove_finished_requests()
            iter_end = time.perf_counter()

            print(
                f"Time: {iter_end-start:.3f}s, scheduler: {prepare_end-start:.3f}s, "
                f"forward: {iter_end-prepare_end:.3f}s"
            )
            self.itr_end_times.append(iter_end)
            self.ntoken_of_itr.append(sum(b.perfdata.x for b in batches))
    
    async def start_all_event_loops(self):
        """
        Start all event loops
        """
        assert self.initialized, "Engine not initialized. Please call `initialize()` before starting the event loop."
        await asyncio.gather(
            self._tokenize_raw_request_event_loop(),
            self._main_event_loop()
        )
