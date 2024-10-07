import time, os, json, math
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from .model import LlamaModel, ModelPerfResult
from swiftllm.perfpredictor import TablePerfPredictor
from swiftllm.structs import create_request, Request, SubBatch
from swiftllm.utils import GB

class ModelProfiler:
  """
  A profiler for the Llama model.
  """
  @torch.inference_mode()
  def __init__(
    self, 
    model: LlamaModel
  ):
    self.model = model
    os.makedirs(model.engine_config.profile_result_path, exist_ok=True)

  def init_profile_tables(self):
    """
    Initialize the profile tables
    """
    # Validate necessary constraints
    engine_config = self.model.engine_config
    max_gpu_tokens = self.model.swapper.gpu_block_manager.num_blocks * engine_config.block_size
    max_cpu_tokens = self.model.swapper.cpu_block_manager.num_blocks * engine_config.block_size

    assert engine_config.max_prefill_tokens <= engine_config.max_tokens_in_batch, \
        f"max_prefill_tokens {engine_config.max_prefill_tokens} exceeds max_tokens_in_batch {engine_config.max_tokens_in_batch}"
    assert engine_config.max_batch_size <= engine_config.max_tokens_in_batch, \
        f"max_batch_size {engine_config.max_batch_size} exceeds max_tokens_in_batch {engine_config.max_tokens_in_batch}"
    assert engine_config.max_batch_size <= engine_config.max_seqs_in_block_table, \
        f"max_batch_size {engine_config.max_batch_size} exceeds max_seqs_in_block_table {engine_config.max_seqs_in_block_table}"

    self.pp = TablePerfPredictor(engine_config, max_gpu_tokens, max_cpu_tokens)

    self.pp.linr_T_list = self._profile_linr(self.pp.linr_S_list)
    self.pp.pref_T_list = self._profile_pref(self.pp.pref_S_list)
    self.pp.gdec_T_list = self._profile_gdec(self.pp.gdec_N_list)
    self.pp.cdec_T_lists = self._profile_cdec(self.pp.cdec_S_list, self.pp.cdec_N_lists)
    

  def _gpu_fake_prefill(
    self,
    reqs: list[Request]
  ):
    """
    Prefill the model with the given sequence ids and sequence lengths.

    All sequences are prefixes of the base tokens.

    We simple allocate and copy corresponding KV entries.
    """

    seq_ids = torch.tensor([req.request_id for req in reqs], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([req.seq_len for req in reqs], dtype=torch.int32, device="cuda")

    self.model.swapper.gpu_block_manager.allocate_blocks_for_seqs(seq_ids, seq_lens)

    block_size = self.model.engine_config.block_size
    nblocks = [(seq_len - 1) // block_size + 1 for seq_len in seq_lens]

    for i, seq_id in enumerate(seq_ids):
      nblock = nblocks[i]
      block_ids = self.model.swapper.gpu_block_manager.block_table[seq_id][:nblock]
      for layer_id in range(self.model.model_config.num_layers):
        self.model.k_cache[layer_id, block_ids].random_().normal_()
        self.model.v_cache[layer_id, block_ids].random_().normal_()

  def _cpu_fake_prefill(
    self,
    reqs: list[Request]
  ):
    # We should segregate it into parts in order not to exceed the maximum number of blocks
    # Note that this is just heuristic based
    num_gpu_free_blocks = self.model.swapper.gpu_block_manager.num_free_blocks
    i = 0
    # Since seq_lens may exceed GPU KV cache size, we need to divide it into parts
    while i < len(reqs):
      j = i
      block_needed = 0
      while j < len(reqs):
        seq_len = reqs[j].seq_len
        nblocks = (seq_len - 1) // self.model.engine_config.block_size + 1
        if block_needed + nblocks > num_gpu_free_blocks:
          break
        block_needed += nblocks
        j += 1
      self._gpu_fake_prefill(reqs[i:j])
      self.model.swap_out_seqs([req.request_id for req in reqs[i:j]])
      i = j

  def _run_test_case_seq(
    self,
    pref_lens: list[int] = [],
    gdec_lens: list[int] = [],
    cdec_lens: list[int] = [],
    nwarmup = 2,
    nrepeat = 3
  ):
    return self._run_test_case([pref_lens], [gdec_lens], [cdec_lens], nwarmup, nrepeat)

  def _run_test_case_pip_same(
    self,
    pref_lens: list[int] = [],
    gdec_lens: list[int] = [],
    cdec_lens: list[int] = [],
    nwarmup = 2,
    nrepeat = 3
  ):
    return self._run_test_case([pref_lens] * 2, [gdec_lens] * 2, [cdec_lens] * 2, nwarmup, nrepeat)

  @torch.inference_mode()
  def _run_test_case(
    self,
    pref_lens: list[list[int]],
    gdec_lens: list[list[int]],
    cdec_lens: list[list[int]],
    nwarmup = 2,
    nrepeat = 3
  ) -> ModelPerfResult:
    """
    Run a artificial test case and return the performance results.
    """
    # print(f"Running test case with pref_lens={pref_lens}, gdec_lens={gdec_lens}, cdec_lens={cdec_lens}")

    nbatches = len(pref_lens)
    assert nbatches == 1 or nbatches == 2, "Only support 1 or 2 batches"

    batches = []
    reqs = []
    all_pref_ids = []
    all_deco_ids = []

    offs = 0
    for i in range(nbatches):
      batch = SubBatch()
      npref = len(pref_lens[i])
      ngdec = len(gdec_lens[i])
      ncdec = len(cdec_lens[i])

      for j in range(npref):
        batch.add_pref(create_request([10] * pref_lens[i][j], offs + j), is_gpu=True)

      for j in range(ngdec):
        batch.add_gdec(create_request([10] * (gdec_lens[i][j] - 1), offs + npref + j, output_token_ids=[10]))

      for j in range(ncdec):
        batch.add_cdec(create_request([10] * (cdec_lens[i][j] - 1), offs + npref + ngdec + j, output_token_ids=[10]))

      offs += npref + ngdec + ncdec

      if ncdec > 0:
        self._cpu_fake_prefill(batch.cdec_reqs)
      if ngdec > 0:
        self._gpu_fake_prefill(batch.gdec_reqs)

      batches.append(batch)
        

    for i in range(-nwarmup, nrepeat):
      if i == 0:
        start = time.perf_counter()
        self.model.turn_on_perf_monitor()
      if nbatches == 1:
        self.model.forward(batches[0])
      else:
        self.model.forward_pipeline(batches)
      # Note that faked prefilling already allocates memory for the "new" token to be fed
      # into the model, so model.forward won't allocate memory for them and we only need
      # to free the resources of the prefilled tokens.
      self.model.free_seqs_resources(all_pref_ids)
      # Roll back the batch state
      for batch in batches:
        for req in batch.all_reqs:
          req.output_token_ids.pop()
          req.cur_output_len -= 1

    elapsed = time.perf_counter() - start
    print(f"Elapsed time: {elapsed * 1000 / nrepeat:.3f} ms")

    assert len(self.model.perf_results) == nrepeat
    res = self.model.flush_perf_results_and_turn_off_perf_monitor()
    self.model.free_seqs_resources(all_deco_ids)
    return res
  
  def _profile_linr(
      self, 
      S_list: list[int]
  ) -> list[float]:
    """
    Profile model's linear part performance.
    """
    result_path = self.model.engine_config.profile_result_path + "linr.json"

    if os.path.exists(result_path):
      with open(result_path, "r") as f:
        table = json.load(f)
        if table["S_list"] == S_list:
          return table["T_list"]
        
    print(f"Profiling linear part with S_list={S_list} ...")

    T_list = []
    for S in tqdm(S_list):
      if S in table["S_list"]:
        T_list.append(table["T_list"][table["S_list"].index(S)])
        continue
      res = self._run_test_case_seq(
        pref_lens=[S]
      )
      T_list.append(ModelPerfResult.mean(res, "avg_linr_time"))

    plt.figure(figsize=(16, 12))
    plt.plot(S_list, T_list)
    plt.xlim(0)
    plt.ylim(0)
    plt.xlabel("S")
    plt.ylabel("T_l(ms)")
    plt.savefig(self.model.engine_config.profile_result_path + "linr.png")
    plt.close()

    with open(result_path, "w") as f:
      json.dump({
        "S_list": S_list,
        "T_list": T_list
      }, f, indent=2)

    return T_list

  def _profile_pref(
    self,
    S_list: list[int]
  ) -> list[list[float]]:
    """
    Profile model's GPU prefilling attention part performance.
    """
    result_path = self.model.engine_config.profile_result_path + "pref.json"

    if os.path.exists(result_path):
      with open(result_path, "r") as f:
        res = json.load(f)
        if res["S_list"] == S_list:
          return res["T_list"]
        
    print(f"Profiling prefill part with S_list={S_list}...")

    T_list = []
    for S in tqdm(S_list):
      res1 = self._run_test_case_seq(
        pref_lens=[S]
      )
      res2 = self._run_test_case_seq(
        pref_lens=[S] * 2
      )
      T_list.append(ModelPerfResult.mean(res2, "avg_pref_time") - ModelPerfResult.mean(res1, "avg_pref_time"))

    plt.figure(figsize=(16, 12))
    plt.plot(S_list, T_list)
    plt.xlim(0)
    plt.ylim(0)
    plt.xlabel("S")
    plt.ylabel("T(ms)")
    plt.savefig(self.model.engine_config.profile_result_path + "pref.png")
    plt.close()

    with open(result_path, "w") as f:
      json.dump({
        "S_list": S_list,
        "T_list": T_list
      }, f, indent=2)

    return T_list

  def _profile_gdec(
    self,
    N_list: list[int]
  ) -> list[float]:
    """
    Profile model's GPU attention part performance.
    """
    result_path = self.model.engine_config.profile_result_path + "gdec.json"

    if os.path.exists(result_path):
      with open(result_path, "r") as f:
        res = json.load(f)
        if res["N_list"] == N_list:
          return res["T_list"]
        
    print(f"Profiling GPU attention part with N_list={N_list} ...")

    T_list = []
    for N in tqdm(N_list):
      res = self._run_test_case_seq(
        gdec_lens=[N]
      )
      T_list.append(ModelPerfResult.mean(res, "avg_gdec_time"))

    plt.figure(figsize=(16, 12))
    plt.plot(N_list, T_list)
    plt.xlim(0)
    plt.ylim(0)
    plt.xlabel("N")
    plt.ylabel("T(ms)")
    plt.savefig(self.model.engine_config.profile_result_path + "gdec.png")
    plt.close()

    with open(result_path, "w") as f:
      json.dump({
        "N_list": N_list,
        "T_list": T_list
      }, f, indent=2)

    return T_list

  def _profile_cdec(
    self,
    S_list: list[int],
    N_lists: list[list[int]]
  ) -> list[list[float]]:
    """
    Profile model's CPU attention part performance.
    """
    result_path = self.model.engine_config.profile_result_path + "cdec.json"

    if os.path.exists(result_path):
      with open(result_path, "r") as f:
        table = json.load(f)
        if table["S_list"] == S_list and table["N_lists"] == N_lists:
          return table["T_lists"]
        
    print(f"Profiling CPU attention part with S_list={S_list}, N_lists={N_lists} ...")
        
    T_lists = []
    block_size = self.model.engine_config.block_size
    for i, S in tqdm(enumerate(S_list)):
      T_lists.append([])
      for N in self.cdec_N_list_agg:
        if N < N_lists[i][0]:
          T_lists[-1].append(0.0)
          continue
        if N > N_lists[i][-1]:
          T_lists[-1].append(float("inf"))
          continue
        assert N % block_size == 0, "N must be divisible by block size"
        NB = N // block_size
        res = self._run_test_case_seq(
          # Divide N into S segments as even as possible
          cdec_lens=[NB // S * block_size] * (S - NB % S) + [(NB // S + 1) * block_size] * (NB % S),
        )
        T_lists[-1].append(ModelPerfResult.mean(res, "avg_cdec_time"))

    T_array = np.array(T_lists)

    plt.figure(figsize=(16, 12))
    ax = plt.axes(projection='3d')
    ax.plot_surface(
      np.outer(S_list, np.ones(len(self.cdec_N_list_agg))),
      np.outer(np.ones(len(S_list)), self.cdec_N_list_agg),
      T_array,
      label = "CPU"
    )

    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel("S_c")
    ax.set_ylabel("N_c")
    ax.set_zlabel("T(ms)")
    plt.savefig(self.model.engine_config.profile_result_path + "cdec.png")
    plt.close()

    with open(result_path, "w") as f:
      json.dump({
        "S_list": S_list,
        "N_lists": N_lists,
        "T_lists": T_lists
      }, f, indent=2)

    return T_lists

  def _profile_lnch(
    self,
    S_list: list[int]
  ) -> list[float]:
    """
    Profile model's kernel launch time.
    """
    result_path = self.model.engine_config.profile_result_path + "lnch.json"

    if os.path.exists(result_path):
      with open(result_path, "r") as f:
        res = json.load(f)
        if res["S_list"] == S_list:
          return res["T_list"]
        
    print(f"Profiling kernel launch time with S_list={S_list} ...")

    T_list = []
    for S in tqdm(S_list):
      res = self._run_test_case_pip_same(
        pref_lens=[S // 2 - 2 * (S // 10)],
        gdec_lens=[10] * (S // 10),
        cdec_lens=[10] * (S // 10),
        use_pipeline=True
      )
      T_list.append(ModelPerfResult.mean(res, "avg_lnch_time"))

    plt.figure(figsize=(16, 12))
    plt.plot(S_list, T_list)
    plt.xlim(0)
    plt.ylim(0)
    plt.xlabel("S")
    plt.ylabel("T(ms)")
    plt.savefig(self.model.engine_config.profile_result_path + "lnch.png")
    plt.close()

    with open(result_path, "w") as f:
      json.dump({
        "S_list": S_list,
        "T_list": T_list
      }, f, indent=2)

    T_mean = np.array(T_list).mean()

    return T_mean

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

    engine_config = self.model.engine_config
    model_config = self.model.model_config

    # Synthesis a prefill batch
    N = engine_config.max_tokens_in_batch
    S = engine_config.max_batch_size
    self._run_test_case_seq(
      pref_lens=[N // S] * (S - N % S) + [N // S + 1] * (N % S),
      nrepeat=1, nwarmup=0
    )
    torch.cuda.synchronize()

    # peak_memory = torch.cuda.max_memory_allocated()
    # total_memory = torch.cuda.get_device_properties(0).total_memory
    free_memory, total_memory = torch.cuda.mem_get_info()
    peak_memory = total_memory - free_memory
    useable_memory = total_memory * engine_config.gpu_mem_utilization
    print(f"[Model.profile] GPU total memory: {total_memory/GB:.2f} GB, runtime peak memory: {peak_memory/GB:.2f} GB")
    if useable_memory < peak_memory:
        raise RuntimeError(f"Peak memory {peak_memory/GB:.2f} GB exceeds usable memory {useable_memory/GB:.2f} GB ({total_memory/GB:.2f} GB * {engine_config.gpu_mem_utilization})")
    block_size_bytes = engine_config.block_size * model_config.get_kvslot_size()
    num_gpu_blocks = math.floor((useable_memory - peak_memory) / block_size_bytes)

    torch.cuda.empty_cache()
    return num_gpu_blocks