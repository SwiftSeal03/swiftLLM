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
        self.gprf_reqs = []
        self.cprf_reqs = []
        self.gdec_reqs = []
        self.cdec_reqs = []
        self.x = 0 # batch size
        self.s = 0 # iteration width
        self.x_c = 0 # number of CPU decoding requests
        self.n_c = 0 # total number of tokens in CPU decoding requests


    def add_gprf(self, req: Request):
        self.gprf_reqs.append(req)
        self.x += 1
        self.s += req.prompt_len
    
    def add_cprf(self, req: Request):
        self.cprf_reqs.append(req)
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
        return self.gprf_reqs + self.cprf_reqs + self.gdec_reqs + self.cdec_reqs

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
            swap_out_seq_ids = [req.request_id for req in self.cprf_reqs],
            cpu_num_decoding_seqs=self.x_c
        )

    def print_profile(self):
        print(f"gprf lens: {[req.prompt_len for req in self.gprf_reqs]}, cprf lens: {[req.prompt_len for req in self.cprf_reqs]}, \
gdec lens: {[req.seq_len for req in self.gdec_reqs]}, cdec lens: {[req.seq_len for req in self.cdec_reqs]}")

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

    def get_num_available_ids(self) -> int:
        return len(self.available_ids)
    
    def free_id(self, req_id: int):
        self.available_ids.append(req_id)
    
    def free_ids(self, req_ids: list[int]):
        self.available_ids.extend(req_ids)

class ScheduleBudget:
    def __init__(self, max_batch_size: int, max_prefill_tokens: int, max_tokens_in_batch: int):
        self.remaining_batch_size = max_batch_size
        self.remaining_prefill_tokens = max_prefill_tokens
        self.remaining_tokens_in_batch = max_tokens_in_batch
    
    @property
    def overspent(self) -> bool:
        return self.remaining_batch_size < 0 or self.remaining_tokens_in_batch < 0

    def check_and_substract(self, num_tokens, is_prefill = False) -> bool:
        if self.remaining_batch_size >= 1 and \
            self.remaining_tokens_in_batch >= num_tokens and \
            self.remaining_prefill_tokens >= (num_tokens if is_prefill else 0):
            self.remaining_batch_size -= 1
            self.remaining_tokens_in_batch -= num_tokens
            self.remaining_prefill_tokens -= num_tokens if is_prefill else 0
            return True
        return False

    def add(self, num_tokens, is_prefill = False) -> bool:
        self.remaining_batch_size += 1
        self.remaining_tokens_in_batch += num_tokens
        self.remaining_prefill_tokens += num_tokens if is_prefill else 0

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
        assert self.engine_config.max_prefill_tokens <= self.engine_config.max_tokens_in_batch, \
            f"max_prefill_tokens {self.engine_config.max_prefill_tokens} exceeds max_tokens_in_batch {self.engine_config.max_tokens_in_batch}"
        assert self.engine_config.max_batch_size <= self.engine_config.max_tokens_in_batch, \
            f"max_batch_size {self.engine_config.max_batch_size} exceeds max_tokens_in_batch {self.engine_config.max_tokens_in_batch}"
        assert self.engine_config.max_batch_size <= self.engine_config.max_seqs_in_block_table, \
            f"max_batch_size {self.engine_config.max_batch_size} exceeds max_seqs_in_block_table {self.engine_config.max_seqs_in_block_table}"
        
        profiler = ModelProfiler(model)
        linr_S_list = [2 ** i for i in range(
            (self.engine_config.block_size - 1).bit_length(),
            (self.engine_config.max_tokens_in_batch - 1).bit_length(),
        )] + [self.engine_config.max_tokens_in_batch]
        linr_T_list = profiler.profile_linear(linr_S_list)

        cdec_S_list = [2 ** i for i in range(
            ((self.engine_config.max_tokens_on_cpu - 1) // self.engine_config.max_seq_len).bit_length(),
            (self.engine_config.max_batch_size - 1).bit_length()
        )] + [self.engine_config.max_batch_size]
        cdec_N_list = [2 ** i for i in range(
            ((self.engine_config.max_batch_size * self.engine_config.block_size) - 1).bit_length(),
            (self.engine_config.max_tokens_on_cpu - 1).bit_length()
        )] + [self.engine_config.max_tokens_on_cpu]

        assert all(cdec_S_list[i] < cdec_S_list[i+1] for i in range(len(cdec_S_list)-1)), f"cdec_S_list {cdec_S_list} is not strictly increasing"
        assert all(cdec_N_list[i] < cdec_N_list[i+1] for i in range(len(cdec_N_list)-1)), f"cdec_N_list {cdec_N_list} is not strictly increasing"
        assert cdec_S_list[-1] <= self.engine_config.max_batch_size, f"cdec_S_list {cdec_S_list} exceeds max_batch_size {self.engine_config.max_batch_size}"
        assert cdec_N_list[-1] <= self.engine_config.max_tokens_on_cpu, f"cdec_N_list {cdec_N_list} exceeds max_tokens_on_cpu {self.engine_config.max_tokens_on_cpu}"
        assert (cdec_N_list[-1] - 1) // cdec_S_list[0] + 1 <= self.engine_config.max_seq_len, \
            f"max_seq_len {self.engine_config.max_seq_len} is not enough for cdec_S_list[0] {cdec_S_list[0]} and cdec_N_list[-1] {cdec_N_list[-1]}"
        assert cdec_N_list[0] // cdec_S_list[-1] >= self.engine_config.block_size, \
            f"block_size {self.engine_config.block_size} is not enough for cdec_S_list[-1] {cdec_S_list[-1]} and cdec_N_list[0] {cdec_N_list[0]}"

        cdec_T_list = profiler.profile_cpu_attn(cdec_S_list, cdec_N_list)
        del profiler

        self.linr_S_arr = np.array(linr_S_list)
        self.linr_T_arr = np.array(linr_T_list)
        self.cdec_S_arr = np.array(cdec_S_list)
        self.cdec_N_arr = np.array(cdec_N_list)
        self.cdec_T_arr = np.array(cdec_T_list)
        self.kernel_launch_time = 1.0

        # Request in the following three deques are sorted by their arrival time
        self.waiting_q: deque[Request] = deque()
        self.gpu_decoding_q: list[Request] = []
        self.cpu_decoding_q: deque[Request] = deque()

        # Number of GPU blocks occupied by decoding requests
        # This number should always equal to sum(self._get_block_needed(req) for req in self.gpu_decoding_q)
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
        assert S <= self.cdec_S_arr[-1], f"CPU batch size {S} exceeds the maximum {self.cdec_S_arr[-1]}"
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
        Get the remaining CPU capacity for a batch with 
        - (opposite batch's) iteration width s
        - number of CPU decoding requests x_c,
        - total number of tokens n_c
        """
        return self._get_linr_T(s) - self._get_cdec_T(x_c, n_c) - self.kernel_launch_time

    def decide_mode_and_gen_batch(
        self, 
        gpu_prefill_reqs: list[Request],
        cpu_prefill_reqs: list[Request],
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

        # Simple heuristic: Put all GPU related seqs in the same batch, all CPU related seqs in the other
        for req in gpu_prefill_reqs:
            batches[0].add_gprf(req)
            sequential_batch.add_gprf(req)

        for req in cpu_prefill_reqs:
            batches[0].add_cprf(req)
            sequential_batch.add_cprf(req)
        
        for req in self.gpu_decoding_q:
            batches[0].add_gdec(req)
            sequential_batch.add_gdec(req)
        
        if self.engine_config.always_use_gpu:
            return sequential_batch, None
        
        for req in self.cpu_decoding_q:
            if not budget.check_and_substract(1):
                break
            batches[1].add_cdec(req)

        if batches[0].x == 0:
            return batches[1], None

        # # Step 1: split prefilling requests
        # for req in prefill_reqs:
        #     b1_has_less_s = int(batches[0].s > batches[1].s)
        #     batches[b1_has_less_s].add_pref(req)
        #     sequential_batch.add_pref(req)

        # # Step 2: split GPU decoding requests
        # for req in self.gpu_decoding_q:
        #     b1_has_less_s = int(batches[0].s > batches[1].s)
        #     batches[b1_has_less_s].add_gdec(req)
        #     sequential_batch.add_gdec(req)
        
        # if self.engine_config.always_use_gpu:
        #     return sequential_batch, None

        # # Step 3: split CPU decoding requests
        # for req in self.cpu_decoding_q:
        #     if not budget.check_and_substract(1):
        #         break
        #     # We insert the req to the batch that maximizes the min R value, i.e. (T_l - T_c)
        #     slf_Rs = [] # slf_Rs[i] is the R value of batch-i if we insert the req to batch-i
        #     opp_Rs = [] # opp_Rs[i] is the R value of batch-i if we insert the req to batch-(1-i)
        #     for i in range(2):
        #         opp_Rs.append(self._get_cpu_remaining_capacity(batches[1-i].s + 1, batches[i].x_c, batches[i].n_c))
        #         slf_Rs.append(self._get_cpu_remaining_capacity(batches[1-i].s, batches[i].x_c + 1, batches[i].n_c + req.seq_len))
        #     min_R0 = min(slf_Rs[0], opp_Rs[1])
        #     min_R1 = min(slf_Rs[1], opp_Rs[0])
        #     # if min_R0 < 0 and min_R1 < 0:
        #     #     # Both are negative, skip this request
        #     #     budget.add(1)
        #     #     continue

        #     should_pick_1 = min_R0 < min_R1
        #     batches[should_pick_1].add_cdec(req)

        # The coefficient theoretically should be 2.0, but we use 2.4 here for performance guarantee
        if sum(b.x for b in batches) > 2.4 * sequential_batch.x:
#             print(f"Estimated linear time: {(self._get_linr_T(batches[0].s) + self._get_linr_T(batches[1].s))/2}")
#             print(f"Estimated CPU decoding time: {(self._get_cdec_T(batches[0].x_c, batches[0].n_c) + self._get_cdec_T(batches[1].x_c, batches[1].n_c))/2}")
#             print(f"Remaining cpu capacity: {self._get_cpu_remaining_capacity(batches[0].s, batches[0].x_c, batches[0].n_c)},\
#  {self._get_cpu_remaining_capacity(batches[1].s, batches[1].x_c, batches[1].n_c)}")
            return batches[0], batches[1]
        else:
            return sequential_batch, None

    def get_next_batch(self) -> tuple[SubBatch, SubBatch | None, list[Request], list[Request], list[Request]]:
        """
        Called when the engine wants a new batch to be forwarded
        Returns (new_batch, newly_swapped_in, newly_swapped_out)
        """
        # We have a budget mainly because we should avoid CUDA OOM
        budget = ScheduleBudget(
            self.engine_config.max_batch_size,
            self.engine_config.max_prefill_tokens,
            self.engine_config.max_tokens_in_batch
        )
        pref_to_cpu = []
        pref_to_gpu = []
        swpout_reqs = []
        swpin_reqs = []

        # Policy may change, should recompute these thresholds on every iteration
        if self.engine_config.always_use_gpu:
            swap_out_threshold = self.num_gpu_blocks
            swap_in_threshold = round(swap_out_threshold * 0.97)
            cpu_threshold = 0
        else:
            swap_out_threshold = self.num_gpu_blocks - self.engine_config.max_prefill_tokens // self.engine_config.block_size
            swap_in_threshold = round(swap_out_threshold * 0.8)
            cpu_threshold = self.num_cpu_blocks - self.num_gpu_blocks
        
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
        cpu_block_needed = sum(self._get_block_needed(req) for req in self.cpu_decoding_q) # for bounding new prefillings
        while self.waiting_q:
            candidate = self.waiting_q[0]
            cur_block_needed = self._get_block_needed(candidate)
            if gpu_block_needed + cur_block_needed > self.num_gpu_blocks or \
               (self.cpu_decoding_q and cpu_block_needed + cur_block_needed > cpu_threshold) or \
               self.request_id_manager.get_num_available_ids() == 0 or \
               not budget.check_and_substract(candidate.prompt_len, True):
                break
            gpu_block_needed += cur_block_needed
            cpu_block_needed += cur_block_needed
            candidate.request_id = self.request_id_manager.get_id()
            self.waiting_q.popleft()
            # The newly prefilled sequences are the major part of swapping out. Here we use a simple heuristic.
            # Note that this might not be necessary since some sequences might finish in this iteration. However, this will not hurt
            #   because the next group of prefilled sequences won't need to be swapped out.
            # Also note that in GPU-only mode, there will never be any pre-swapped-out requests.
            if gpu_block_needed <= swap_out_threshold:
                pref_to_gpu.append(candidate)
            else:
                pref_to_cpu.append(candidate)

        assert not (pref_to_cpu and self.engine_config.always_use_gpu), "In GPU-only mode, there should be no pre-swapped-out requests"

        # Step 5: Launch CPU decoding requests and form batches
        batch0, batch1 = self.decide_mode_and_gen_batch(pref_to_gpu, pref_to_cpu, budget)

        if self.gpu_decoding_q or self.cpu_decoding_q or pref_to_gpu or pref_to_cpu or self.waiting_q:
            print(f"Gdecs: {len(self.gpu_decoding_q)}, Cdecs: {len(self.cpu_decoding_q)}, Pr2gs: {len(pref_to_gpu)}, Pr2cs: {len(pref_to_cpu)}, Waiting: {len(self.waiting_q)}")

        self.gpu_decoding_q.extend(pref_to_gpu)
        self.cpu_decoding_q.extend(pref_to_cpu)

        return batch0, batch1, swpout_reqs, swpin_reqs

    def get_next_batch_old(self) -> tuple[SubBatch, list[Request], list[Request]]:
        print(f"Waiting: {len(self.waiting_q)}, Gdecs: {len(self.gpu_decoding_q)}, Cdecs: {len(self.cpu_decoding_q)}")
        if not self.cpu_decoding_q:
            # Try to launch a new prefill batch
            cur_batch = SubBatch()
            cur_batch_block_needed = 0
            cur_num_tokens_sum = 0
            while self.waiting_q:
                cur_seq: Request = self.waiting_q[0]
                cur_seq_block_needed = self._get_block_needed(cur_seq)
                if  cur_batch.x+1 <= self.engine_config.max_batch_size and \
                    len(self.gpu_decoding_q)+cur_batch.x+1 <= self.engine_config.max_batch_size and \
                    cur_batch_block_needed + cur_seq_block_needed + self.num_decoding_gpu_blocks <= self.num_gpu_blocks and \
                    cur_num_tokens_sum + cur_seq.prompt_len <= self.engine_config.max_tokens_in_batch:
                    cur_batch.add_pref(cur_seq)
                    cur_batch_block_needed += cur_seq_block_needed
                    cur_num_tokens_sum += cur_seq.prompt_len
                    self.waiting_q.popleft()
                    
                else:
                    # Strict FCFS
                    break
            if cur_batch.x:
                # Going to launch a prefill batch
                # If you want decoding requests to be piggybacked, you can do it here
                for req in cur_batch.pref_reqs:
                    req.request_id = self.request_id_manager.get_id()
                self.gpu_decoding_q.extend(cur_batch.pref_reqs)
                self.num_decoding_gpu_blocks += cur_batch_block_needed
                return cur_batch, [], []
        
        # Try to launch a decoding batch
        # TODO Optimize this `sum` if possible
        self.num_decoding_gpu_blocks = sum(self._get_block_needed(req) for req in self.gpu_decoding_q)
        newly_swapped_out = []
        while len(self.gpu_decoding_q) > self.engine_config.max_batch_size or \
              self.num_decoding_gpu_blocks > self.num_gpu_blocks:
            # Preempt the last running seq
            victim = self.gpu_decoding_q.pop()
            self.num_decoding_gpu_blocks -= self._get_block_needed(victim)
            newly_swapped_out.append(victim)
        newly_swapped_out.reverse()   # Keep it in the order of arrival time
 
        newly_swapped_in = []
        if newly_swapped_out:
            self.cpu_decoding_q.extendleft(newly_swapped_out)
        else:
            # No swap-out triggered, try to swap in some requests if possible
            while self.cpu_decoding_q:
                cur_seq = self.cpu_decoding_q[0]
                num_cur_seq_blocks = self._get_block_needed(cur_seq)
                if len(self.gpu_decoding_q) + 1 <= self.engine_config.max_batch_size and \
                   self.num_decoding_gpu_blocks + num_cur_seq_blocks <= self.num_gpu_blocks:
                    self.gpu_decoding_q.append(cur_seq)
                    self.num_decoding_gpu_blocks += num_cur_seq_blocks
                    self.cpu_decoding_q.popleft()
                    newly_swapped_in.append(cur_seq)
                else:
                    break
        
        cur_batch = SubBatch()
        for req in self.gpu_decoding_q:
            cur_batch.add_gdec(req)
        return cur_batch, newly_swapped_out, newly_swapped_in
    
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
