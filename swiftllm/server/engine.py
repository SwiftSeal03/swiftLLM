import time
import asyncio
import functools
from typing import AsyncGenerator
from pprint import pprint

import torch

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.model import LlamaModel, ModelForwardArgs
from swiftllm.utils import GB

from .tokenization_engine import TokenizationEngine
from .structs import Request, RawRequest, StepOutput
from .scheduler import Scheduler, SubBatch

class Engine:
    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        self.model_config = LlamaModelConfig.load_from_model_path(engine_config.model_path)
        self.initialized = False

        # The following fields will be created on `init_model()`
        self.model = None
        self.event_loop = None
        self.scheduler = None
        self.tokenization_engine = None

        self.untokenized_raw_requests: list[tuple[Request, str]] = []

    async def _run_on_model_async(self, func, *args, **kwargs):
        """
        Run a function on the model asynchronously, and return the result
        """
        func_partial = functools.partial(func, *args, **kwargs)
        return await self.event_loop.run_in_executor(None, func_partial)

    async def initialize(self):
        self.event_loop = asyncio.get_event_loop()

        print("[Engine] Initializing model...")
        self.model = LlamaModel(self.engine_config)

        print("[Engine] Loading weights...")
        self.model.load_weights()

        print("[Engine] Profiling kv blocks...")
        num_gpu_blocks = 1700 # self.model.profile_num_blocks()
        num_cpu_blocks = self.engine_config.num_cpu_blocks
        block_size_bytes = self.engine_config.block_size*self.model_config.get_kvslot_size()
        print(f"[Engine] Number of GPU blocks: {num_gpu_blocks} ({num_gpu_blocks*block_size_bytes/GB:.2f} GB)")
        print(f"[Engine] Number of CPU blocks: {num_cpu_blocks} ({num_cpu_blocks*block_size_bytes/GB:.2f} GB)")

        print("[Engine] Allocating kv cache and swap...")
        self.model.init_kvcache_and_swap(num_gpu_blocks)

        print("[Engine] Initializing scheduler...")
        self.scheduler = Scheduler(self.model)

        print("[Engine] Initializing tokenization engine...")
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
            batch = [None] * 2
            
            if self.engine_config.always_use_gpu:
                pre_swap_out = None
                batch[0], cur_swap_out, cur_swap_in = self.scheduler.get_next_batch_old()
            else:
                batch[0], batch[1], pre_swap_out, cur_swap_out, cur_swap_in = self.scheduler.get_next_batch()

            if all(not (batch[i] and batch[i].x) for i in range(2)) and not cur_swap_in and not cur_swap_out:
                # No new batch, sleep for a bit
                await asyncio.sleep(0.005)
                continue
            start = time.perf_counter()

            # Perform swap in/out
            if cur_swap_out:
                await self._run_on_model_async(
                    self.model.swap_out_seqs,
                    [req.request_id for req in cur_swap_out]
                )
            swp_out_finish = time.perf_counter()
            if cur_swap_in:
                await self._run_on_model_async(
                    self.model.swap_in_seqs,
                    [req.request_id for req in cur_swap_in]
                )
            swp_in_finish = time.perf_counter()
            
            # Forward the model
            argss = [
                batch[i].get_model_forward_args() if batch[i] else None
                for i in range(2)
            ]
            pre_swap_out_ids = [req.request_id for req in pre_swap_out] if pre_swap_out else None
            if not batch[1]:
                # Sequential mode
                reqs = batch[0].get_all_reqs()
                print(f"Using sequential mode (batch_size = {len(reqs)})")
                output_tokens = await self._run_on_model_async(self.model.forward, argss[0], pre_swap_out_ids)
            else:
                # Pipelined mode
                reqs = batch[0].get_all_reqs() + batch[1].get_all_reqs()
                print(f"Using pipelined mode (batch_size = {len(reqs)})")
                output_tokens = await self._run_on_model_async(self.model.forward_pipeline, argss, pre_swap_out_ids)

            # Deal with output tokens
            finished_req_ids = []
            for req, output_token in zip(reqs, output_tokens):
                req.cur_output_len += 1
                req.output_token_ids.append(output_token)
                req.output_q.put_nowait(StepOutput(output_token, req))
                if req.is_finished():
                    finished_req_ids.append(req.request_id)
                    req.finished_event.set()
            if finished_req_ids:
                await self._run_on_model_async(
                    self.model.free_seqs_resources,
                    finished_req_ids
                )
            
            iter_end = time.perf_counter()
            # Inform the scheduler, swap out newly prefilled requests if necessary
            self.scheduler.on_batch_finish(reqs)

            print(f"Time: {iter_end-start:.3f}s, Swap out: {swp_out_finish-start:.3f}s, Swap in: {swp_in_finish-swp_out_finish:.3f}s, Forward: {iter_end-swp_in_finish:.3f}s")
    
    async def start_all_event_loops(self):
        """
        Start all event loops
        """
        assert self.initialized, "Engine not initialized. Please call `initialize()` before starting the event loop."
        await asyncio.gather(
            self._tokenize_raw_request_event_loop(),
            self._main_event_loop()
        )
