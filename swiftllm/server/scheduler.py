from collections import deque
import dataclasses
import numpy as np

from swiftllm.worker.model import LlamaModel, ModelForwardArgs
from swiftllm.worker.profiler import ModelProfiler
from swiftllm.engine_config import EngineConfig
from swiftllm.utils import cdiv
from swiftllm.server.structs import Request

class SubBatch:
    def __init__(self):
        self.pref_reqs = []
        self.gdec_reqs = []
        self.cdec_reqs = []
        self.x = 0 # batch size
        self.s = 0 # iteration width
        self.x_c = 0 # number of CPU decoding requests
        self.n_c = 0 # total number of tokens in CPU decoding requests


    def add_pref(self, req: Request):
        self.pref_reqs.append(req)
        self.x += 1
        self.s += req.prompt_len
        
    def add_gdec(self, req: Request):
        self.gdec_reqs.append(req)
        self.x += 1
        self.s += 1

    def add_cdec(self, req: Request):
        self.cdec_reqs.append(req)
        self.x += 1
        self.s += 1
        self.x_c += 1
        self.n_c += req.prompt_len

    def get_all_reqs(self) -> list[Request]:
        return self.pref_reqs + self.gdec_reqs + self.cdec_reqs

    def get_model_forward_args(self) -> ModelForwardArgs:
        all_reqs = self.get_all_reqs()
        return ModelForwardArgs(
            input_ids_list = [
                req.prompt_token_ids if req.is_prefill_stage() else [req.output_token_ids[-1]]
                for req in all_reqs
            ],
            seq_ids_list = [req.request_id for req in all_reqs],
            decoding_seq_lens_list = [
                req.seq_len
                for req in self.gdec_reqs + self.cdec_reqs
            ],
            cpu_num_decoding_seqs=self.x_c
        )

class RequestIdManager:
    """
    A class that maintains available request ids
    """
    def __init__(self, max_id: int):
        # Id should be in range [0, max_id)
        self.max_id = max_id
        self.available_ids = deque(range(max_id))
    
    def get_id(self) -> int:
        if not self.available_ids:
            raise RuntimeError("No more available request ids. Please try to increase `max_seqs_in_block_table`")
        return self.available_ids.popleft()
    
    def free_id(self, req_id: int):
        self.available_ids.append(req_id)
    
    def free_ids(self, req_ids: list[int]):
        self.available_ids.extend(req_ids)

class ScheduleBudget:
    def __init__(self, max_batch_size: int, max_tokens_in_batch: int):
        self.remaining_batch_size = max_batch_size
        self.remaining_tokens_in_batch = max_tokens_in_batch
    
    @property
    def overspent(self) -> bool:
        return self.remaining_batch_size < 0 or self.remaining_tokens_in_batch < 0

    def check_and_substract(self, num_tokens) -> bool:
        if self.remaining_batch_size >= 1 and self.remaining_tokens_in_batch >= num_tokens:
            self.remaining_batch_size -= 1
            self.remaining_tokens_in_batch -= num_tokens
            return True
        return False

    def add(self, num_tokens) -> bool:
        self.remaining_batch_size += 1
        self.remaining_tokens_in_batch += num_tokens

class Scheduler:
    """
    A strict FCFS scheduler for the LLM engine, which supports paged attention
    as well as swapping in/out
    """

    def __init__(self, model: LlamaModel):
        self.engine_config = model.engine_config
        self.num_gpu_blocks = model.gpu_block_manager.num_blocks
        self.num_cpu_blocks = model.cpu_block_manager.num_blocks

        # Ensure engine can always prefill a sequence
        
        profiler = ModelProfiler(model)
        linr_S_list = list(range(32, 512, 32)) + list(range(512, self.engine_config.max_seq_len + 512, 512))
        linr_T_list = profiler.profile_linear(linr_S_list)
        cdec_S_list = [2 ** i for i in range(4, 7)] + list(range(128, self.engine_config.max_batch_size + 128, 128))
        cdec_N_list = [2 ** i for i in range(11, 13)] + list(range(8192, 131072 + 4096, 8192))
        cdec_T_list = profiler.profile_cpu_attn(cdec_S_list, cdec_N_list)
        del profiler

        self.linr_S_arr = np.array(linr_S_list)
        self.linr_T_arr = np.array(linr_T_list)
        self.cdec_S_arr = np.array(cdec_S_list)
        self.cdec_N_arr = np.array(cdec_N_list)
        self.cdec_T_arr = np.array(cdec_T_list)
        self.kernel_launch_time = 0.8

        # Request in the following three deques are sorted by their arrival time
        self.waiting_q: deque[Request] = deque()
        self.prefilling_q: list[Request] = []
        self.gpu_decoding_q: list[Request] = []
        self.cpu_decoding_q: deque[Request] = deque()

        # Number of GPU blocks occupied by decoding requests
        # This number should always equal to sum(self._get_block_needed(req) for req in self.running_q)
        self.num_decoding_gpu_blocks = 0
        self.num_free_cpu_blocks = self.engine_config.num_cpu_blocks

        self.request_id_manager = RequestIdManager(self.engine_config.max_seqs_in_block_table)
    
    def _get_block_needed(self, request: Request) -> int:
        """
        Get the number of blocks needed for a request
        """
        return cdiv(request.seq_len, self.engine_config.block_size)
    
    def on_requests_arrival(self, requests: list[Request]):
        """
        Called when a batch of new requests arrives and finishes tokenization
        """
        self.waiting_q.extend(requests)

    def _get_linr_T(self, S: int) -> float:
        """
        Get the linear time for iteration width S, using linear interpolation
        """
        # Note that when S < 32, we use the time for S=32
        assert S <= self.linr_S_arr[-1], f"Iteration width {S} exceeds the maximum {self.linr_S_arr[-1]}"
        return np.interp(S, self.linr_S_arr, self.linr_T_arr)
    
    def _get_cdec_T(self, S: int, N: int) -> float:
        """
        Get the CPU decoding time for iteration width S and number of tokens N,
        using bilinear interpolation
        """
        assert S <= self.cdec_S_arr[-1], f"Iteration width {S} exceeds the maximum {self.cdec_S_arr[-1]}"
        assert N <= self.cdec_N_arr[-1], f"Number of tokens {N} exceeds the maximum {self.cdec_N_arr[-1]}"
        if S == 0:
            return 0.0
        idx = np.searchsorted(self.cdec_S_arr, S)
        if idx == 0 or self.cdec_S_arr[idx] == S:
            return np.interp(N, self.cdec_N_arr, self.cdec_T_arr[idx])
        # S is between cdec_S_arr[idx-1] and cdec_S_arr[idx]
        s0 = self.cdec_S_arr[idx-1]
        s1 = self.cdec_S_arr[idx]
        t0 = np.interp(N, self.cdec_N_arr, self.cdec_T_arr[idx-1])
        t1 = np.interp(N, self.cdec_N_arr, self.cdec_T_arr[idx])
        return np.interp(S, np.array([s0, s1]), np.array([t0, t1]))
    
    def _get_cpu_remaining_capacity(self, s: int, x_c: int, n_c: int):
        """
        Get the remaining CPU capacity for a batch with iteration width s, number of CPU decoding requests x_c,
        and total number of tokens n_c
        """
        return self._get_linr_T(s) - self._get_cdec_T(x_c, n_c) - self.kernel_launch_time

    def decide_mode_and_gen_batch(
        self, 
        budget: ScheduleBudget
    ) -> tuple[SubBatch, SubBatch | None]:
        """
        Assume that self.gpu_decoding_q and self.prefilling_q are fixed.

        Pick 2 sub-batches using heuristics and compare the TP with seqential mode.

        Returns:
            (batch, None) if using sequential mode, or
            (batch0, batch1) if using pipelined mode
        """
        batches = SubBatch(), SubBatch()
        sequential_batch = SubBatch()

        # Step 1: split prefilling requests
        for req in self.prefilling_q:
            b1_has_less_s = int(batches[0].s > batches[1].s)
            batches[b1_has_less_s].add_pref(req)
            sequential_batch.add_pref(req)

        # Step 2: split GPU decoding requests
        for req in self.gpu_decoding_q:
            b1_has_less_s = int(batches[0].s > batches[1].s)
            batches[b1_has_less_s].add_gdec(req)
            sequential_batch.add_gdec(req)
        
        if self.engine_config.always_use_gpu:
            return sequential_batch, None

        # Step 3: split CPU decoding requests
        for req in self.cpu_decoding_q:
            if not budget.check_and_substract(1):
                break
            # We insert the req to the batch that maximizes the min R value, i.e. (T_l - T_c)
            old_Rs = []
            new_Rs = [] 
            for i in range(2):
                old_Rs.append(self._get_cpu_remaining_capacity(batches[i].s, batches[i].x_c, batches[i].n_c))
                new_Rs.append(self._get_cpu_remaining_capacity(batches[i].s + 1, batches[i].x_c + 1, batches[i].n_c + req.seq_len))
            min_R0 = min(new_Rs[0], old_Rs[1])
            min_R1 = min(new_Rs[1], old_Rs[0])
            if min_R0 < 0 and min_R1 < 0:
                # Both are negative, skip this request
                budget.add(1)
                continue

            should_pick_1 = min_R0 < min_R1
            batches[should_pick_1].add_cdec(req)

        # The coefficient theoretically should be 2.0, but we use 2.4 here for performance guarantee
        if sum(b.x for b in batches) > 2.4 * sequential_batch.x:
            return batches[0], batches[1]
        else:
            return sequential_batch, None

    
    def get_next_batch(self) -> tuple[SubBatch, SubBatch | None, list[Request], list[Request]]:
        """
        Called when the engine wants a new batch to be forwarded
        Returns (new_batch, newly_swapped_in, newly_swapped_out)
        """
        # We have a budget mainly because we should avoid CUDA OOM
        budget = ScheduleBudget(self.engine_config.max_batch_size, self.engine_config.max_tokens_in_batch)
        swpout_reqs = []
        swpin_reqs = []

        # Policy may change, should recompute these thresholds on every iteration
        if self.engine_config.always_use_gpu:
            swap_out_threshold = self.num_gpu_blocks
            cpu_threshold = 0
        else:
            swap_out_threshold = self.num_gpu_blocks - self.engine_config.max_tokens_in_batch // self.engine_config.block_size
            cpu_threshold = self.num_cpu_blocks - self.engine_config.max_blocks_per_seq 
        
        swap_in_threshold = round(swap_out_threshold * 0.95)


        # Step 1: Try to launch as many GPU decoding requests as possible
        gpu_block_needed = sum(self._get_block_needed(req) for req in self.gpu_decoding_q)
        budget.remaining_batch_size -= len(self.gpu_decoding_q)
        budget.remaining_tokens_in_batch -= len(self.gpu_decoding_q)

        # Step 2: Swap out requests if necessary
        while budget.overspent or gpu_block_needed > swap_out_threshold:
            # Preempt the last running seq
            victim = self.gpu_decoding_q.pop()
            self.cpu_decoding_q.appendleft(victim)
            swpout_reqs.append(victim)
            gpu_block_needed -= self._get_block_needed(victim)
            budget.add(1)

        # Step 3: Swap in requests if possible
        while self.cpu_decoding_q:
            candidate = self.cpu_decoding_q[0]
            cur_block_needed = self._get_block_needed(candidate)
            if gpu_block_needed + cur_block_needed > swap_in_threshold or \
               not budget.check_and_substract(1):
                break
            gpu_block_needed += cur_block_needed
            swpin_reqs.append(candidate)
            self.cpu_decoding_q.popleft()
            self.gpu_decoding_q.append(candidate)
        assert not swpout_reqs or not swpin_reqs

        # Step 4: Launch prefilling requests
        assert not self.prefilling_q
        cpu_block_needed = sum(self._get_block_needed(req) for req in self.cpu_decoding_q) # for bounding new prefillings
        while self.waiting_q:
            candidate = self.waiting_q[0]
            cur_block_needed = self._get_block_needed(candidate)
            if gpu_block_needed + cur_block_needed > self.num_gpu_blocks or \
               (self.cpu_decoding_q and cpu_block_needed + cur_block_needed > cpu_threshold) or \
               not budget.check_and_substract(candidate.prompt_len):
                break
            gpu_block_needed += cur_block_needed
            cpu_block_needed += cur_block_needed
            candidate.request_id = self.request_id_manager.get_id()
            self.waiting_q.popleft()
            self.prefilling_q.append(candidate)

        # Step 5: Launch CPU decoding requests and form batches
        batch0, batch1 = self.decide_mode_and_gen_batch(budget)

        if self.gpu_decoding_q or self.cpu_decoding_q or self.prefilling_q or self.waiting_q:
            print(f"Gdecs: {len(self.gpu_decoding_q)}, Cdecs: {len(self.cpu_decoding_q)}, Prefs: {len(self.prefilling_q)}, Waiting: {len(self.waiting_q)}")

        return batch0, batch1, swpout_reqs, swpin_reqs

        # if not self.swapped_q:
        #     # Try to launch a new prefill batch
        #     cur_batch = []
        #     cur_batch_block_needed = 0
        #     cur_num_tokens_sum = 0
        #     while self.waiting_q:
        #         cur_seq: Request = self.waiting_q[0]
        #         cur_seq_block_needed = self._get_block_needed(cur_seq)
        #         if  len(cur_batch)+1 <= self.engine_config.max_batch_size and \
        #             len(self.running_q)+len(cur_batch)+1 <= self.engine_config.max_batch_size and \
        #             cur_batch_block_needed + cur_seq_block_needed + self.num_decoding_gpu_blocks <= self.num_gpu_blocks and \
        #             cur_num_tokens_sum + cur_seq.prompt_len <= self.engine_config.max_tokens_in_batch:
        #             cur_batch.append(cur_seq)
        #             cur_batch_block_needed += cur_seq_block_needed
        #             cur_num_tokens_sum += cur_seq.prompt_len
        #             self.waiting_q.popleft()
                    
        #         else:
        #             # Strict FCFS
        #             break
        #     if cur_batch:
        #         # Going to launch a prefill batch
        #         # If you want decoding requests to be piggybacked, you can do it here
        #         for req in cur_batch:
        #             req.request_id = self.request_id_manager.get_id()
        #         self.running_q.extend(cur_batch)
        #         self.num_decoding_gpu_blocks += cur_batch_block_needed
        #         return cur_batch, [], []
        
        # # Try to launch a decoding batch
        # # TODO Optimize this `sum` if possible
        # self.num_decoding_gpu_blocks = sum(self._get_block_needed(req) for req in self.running_q)
        # newly_swapped_out = []
        # while len(self.running_q) > self.engine_config.max_batch_size or \
        #       self.num_decoding_gpu_blocks > self.num_gpu_blocks:
        #     # Preempt the last running seq
        #     victim = self.running_q.pop()
        #     self.num_decoding_gpu_blocks -= self._get_block_needed(victim)
        #     newly_swapped_out.append(victim)
        # newly_swapped_out.reverse()   # Keep it in the order of arrival time

        # newly_swapped_in = []
        # if newly_swapped_out:
        #     self.swapped_q.extendleft(newly_swapped_out)
        # else:
        #     # No swap-out triggered, try to swap in some requests if possible
        #     while self.swapped_q:
        #         cur_seq = self.swapped_q[0]
        #         num_cur_seq_blocks = self._get_block_needed(cur_seq)
        #         if len(self.running_q) + 1 <= self.engine_config.max_batch_size and \
        #            self.num_decoding_gpu_blocks + num_cur_seq_blocks <= self.num_gpu_blocks:
        #             self.running_q.append(cur_seq)
        #             self.num_decoding_gpu_blocks += num_cur_seq_blocks
        #             self.swapped_q.popleft()
        #             newly_swapped_in.append(cur_seq)
        #         else:
        #             break
        
        # return self.running_q, newly_swapped_in, newly_swapped_out
    
    def on_batch_finish(self, batch: list[Request]):
        """
        Called when a batch finishes
        """
        not_finished_func = lambda req: not req.is_finished()
        self.request_id_manager.free_ids([
            req.request_id
            for req in batch
            if req.is_finished()
        ])
        self.gpu_decoding_q = list(filter(not_finished_func, self.gpu_decoding_q))
        self.cpu_decoding_q = deque(filter(not_finished_func, self.cpu_decoding_q))
        self.prefilling_q = list(filter(not_finished_func, self.prefilling_q))

        if self.cpu_decoding_q:
            # Should swap out prefilled requests for fairness
            swpout_reqs = [req for req in self.prefilling_q]
            self.prefilling_q = []
            self.cpu_decoding_q.extend(swpout_reqs)
            return swpout_reqs
        else:
            # They could be swapped out if exceeds threshold in the next iteration
            self.gpu_decoding_q.extend(self.prefilling_q)
            self.prefilling_q = []
            return []
