import time, os, json
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from .model import LlamaModel, ModelForwardArgs, ModelPerfResult

class ModelProfiler:
  """
  A profiler for the Llama model.
  """
  @torch.inference_mode()
  def __init__(
    self, 
    model: LlamaModel,
    nwarmup: int = 2,
    nrepeat: int = 3
  ):
    self.model = model
    self.nrepeat = nrepeat
    self.nwarmup = nwarmup
    os.makedirs(model.engine_config.profile_result_path, exist_ok=True)
    
    self._init_profile_tables()

  def _init_profile_tables(self):
    """
    Initialize the profile tables
    """
    # Validate necessary constraints
    engine_config = self.model.engine_config
    max_gpu_tokens = self.model.gpu_block_manager.num_blocks * engine_config.block_size
    max_cpu_tokens = self.model.cpu_block_manager.num_blocks * engine_config.block_size

    assert engine_config.max_prefill_tokens <= engine_config.max_tokens_in_batch, \
        f"max_prefill_tokens {engine_config.max_prefill_tokens} exceeds max_tokens_in_batch {engine_config.max_tokens_in_batch}"
    assert engine_config.max_batch_size <= engine_config.max_tokens_in_batch, \
        f"max_batch_size {engine_config.max_batch_size} exceeds max_tokens_in_batch {engine_config.max_tokens_in_batch}"
    assert engine_config.max_batch_size <= engine_config.max_seqs_in_block_table, \
        f"max_batch_size {engine_config.max_batch_size} exceeds max_seqs_in_block_table {engine_config.max_seqs_in_block_table}"

    # Linr
    linr_S_list = [2 ** i for i in range(
        (engine_config.block_size - 1).bit_length(),
        (engine_config.max_tokens_in_batch - 1).bit_length(),
    )] + [engine_config.max_tokens_in_batch]
    linr_T_list = self._profile_linr(linr_S_list)
    self.linr_S_arr = np.array(linr_S_list)
    self.linr_T_arr = np.array(linr_T_list)

    # Pref
    pref_S_list = [2 ** i for i in range(
        (engine_config.block_size - 1).bit_length(),
        (engine_config.max_prefill_tokens - 1).bit_length()
    )] + [engine_config.max_prefill_tokens]
    pref_T_list = self._profile_pref(pref_S_list)
    self.pref_S_arr = np.array(pref_S_list)
    self.pref_T_arr = np.array(pref_T_list)

    # Gdec
    gdec_N_list = [2 ** i for i in range(
        (engine_config.block_size - 1).bit_length(),
        (max_gpu_tokens - 1).bit_length()
    )] + [max_gpu_tokens]
    gdec_T_list = self._profile_gdec(gdec_N_list)
    self.gdec_N_arr = np.array(gdec_N_list)
    self.gdec_T_arr = np.array(gdec_T_list)

    # Cdec
    cdec_S_list = [2 ** i for i in range(
        ((engine_config.max_tokens_on_cpu - 1) // engine_config.max_seq_len).bit_length(),
        (engine_config.max_batch_size - 1).bit_length()
    )] + [engine_config.max_batch_size]
    cdec_N_list = [2 ** i for i in range(
        ((engine_config.max_batch_size * engine_config.block_size) - 1).bit_length(),
        (max_cpu_tokens - 1).bit_length()
    )] + [max_cpu_tokens]

    # Ensure monotonicity and sequence length are in proper range
    assert all(cdec_S_list[i] < cdec_S_list[i+1] for i in range(len(cdec_S_list)-1)), f"cdec_S_list {cdec_S_list} is not strictly increasing"
    assert all(cdec_N_list[i] < cdec_N_list[i+1] for i in range(len(cdec_N_list)-1)), f"cdec_N_list {cdec_N_list} is not strictly increasing"
    assert (cdec_N_list[-1] - 1) // cdec_S_list[0] + 1 <= engine_config.max_seq_len, \
        f"max_seq_len {engine_config.max_seq_len} is not enough for cdec_S_list[0] {cdec_S_list[0]} and cdec_N_list[-1] {cdec_N_list[-1]}"
    assert cdec_N_list[0] // cdec_S_list[-1] >= engine_config.block_size, \
        f"block_size {engine_config.block_size} is not enough for cdec_S_list[-1] {cdec_S_list[-1]} and cdec_N_list[0] {cdec_N_list[0]}"

    cdec_T_list = self._profile_cdec(cdec_S_list, cdec_N_list)
    self.cdec_S_arr = np.array(cdec_S_list)
    self.cdec_N_arr = np.array(cdec_N_list)
    self.cdec_T_arr = np.array(cdec_T_list)
    
    # Lnch
    lnch_S_list = linr_S_list
    self.lnch_T = self._profile_lnch(lnch_S_list)

  def _gpu_fake_prefill(
    self,
    seq_ids: list[int],
    seq_lens: list[int]
  ):
    """
    Prefill the model with the given sequence ids and sequence lengths.

    All sequences are prefixes of the base tokens.

    We simple allocate and copy corresponding KV entries.
    """

    assert all(0 <= seq_id for seq_id in seq_ids), "Sequence ids must be positive"

    seq_ids = torch.tensor(seq_ids, dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")

    self.model.gpu_block_manager.allocate_blocks_for_seqs(seq_ids, seq_lens)

    block_size = self.model.engine_config.block_size
    nblocks = [(seq_len - 1) // block_size + 1 for seq_len in seq_lens]

    for i, seq_id in enumerate(seq_ids):
      nblock = nblocks[i]
      block_ids = self.model.gpu_block_manager.block_table[seq_id][:nblock]
      for layer_id in range(self.model.model_config.num_layers):
        self.model.k_cache[layer_id, block_ids].random_().normal_()
        self.model.v_cache[layer_id, block_ids].random_().normal_()

  def _cpu_fake_prefill(
    self,
    seq_ids: list[int],
    seq_lens: list[int]
  ):
    # We should segregate it into parts in order not to exceed the maximum number of blocks
    # Note that this is just heuristic based
    num_gpu_free_blocks = self.model.gpu_block_manager.num_free_blocks
    i = 0
    # Since seq_lens may exceed GPU KV cache size, we need to divide it into parts
    while i < len(seq_ids):
      j = i
      block_needed = 0
      while j < len(seq_ids):
        seq_len = seq_lens[j]
        nblocks = (seq_len - 1) // self.model.engine_config.block_size + 1
        if block_needed + nblocks > num_gpu_free_blocks:
          break
        block_needed += nblocks
        j += 1
      self._gpu_fake_prefill(seq_ids[i:j], seq_lens[i:j])
      self.model.swap_out_seqs(seq_ids[i:j])
      i = j

  @torch.inference_mode()
  def _run_test_case(
    self,
    prefill_lens: list[int] = [],
    gpu_decode_lens: list[int] = [],
    cpu_decode_lens: list[int] = [],
    use_pipeline: bool = False
  ) -> ModelPerfResult:
    """
    Run a artificial test case and return the performance results.
    """
    # print(f"Running test case with prefill_lens={prefill_lens}, gpu_decode_lens={gpu_decode_lens}, cpu_decode_lens={cpu_decode_lens}")

    npref = len(prefill_lens)
    ngdec = len(gpu_decode_lens)
    ncdec = len(cpu_decode_lens)

    argss = []
    all_prefill_ids = []
    all_decode_ids = []

    offs = 0
    for _ in range(2 if use_pipeline else 1):
      prefill_ids = list(range(offs, offs + npref))
      gpu_decode_ids = list(range(offs + npref, offs + npref + ngdec))
      cpu_decode_ids = list(range(offs + npref + ngdec, offs + npref + ngdec + ncdec))
      offs += npref + ngdec + ncdec

      all_prefill_ids.extend(prefill_ids)
      all_decode_ids.extend(gpu_decode_ids + cpu_decode_ids)

      if ncdec > 0:
        self._cpu_fake_prefill(cpu_decode_ids, cpu_decode_lens)
      if ngdec > 0:
        self._gpu_fake_prefill(gpu_decode_ids, gpu_decode_lens)

      input_ids = [[10] * seq_len for seq_len in prefill_lens] + [[10]] * (ngdec + ncdec)

      argss.append(ModelForwardArgs(
        input_ids,
        prefill_ids + gpu_decode_ids + cpu_decode_ids,
        gpu_decode_lens + cpu_decode_lens,
        [],
        len(cpu_decode_ids)
      ))

    for i in range(-self.nwarmup, self.nrepeat):
      if i == 0:
        start = time.perf_counter()
        self.model.turn_on_perf_monitor()
      if not use_pipeline:
        self.model.forward(argss[0])
      else:
        self.model.forward_pipeline(argss)
      # Note that faked prefilling already allocates memory for the "new" token to be fed
      # into the model, so model.forward won't allocate memory for them and we only need
      # to free the resources of the prefilled tokens.
      self.model.free_seqs_resources(all_prefill_ids)

    elapsed = time.perf_counter() - start
    print(f"Elapsed time: {elapsed * 1000 / self.nrepeat:.3f} ms")

    assert len(self.model.perf_results) == self.nrepeat
    res = self.model.flush_perf_results_and_turn_off_perf_monitor()
    self.model.free_seqs_resources(all_decode_ids)
    return res
  
  def _profile_linr(
      self, 
      S_list: list[int]
  ) -> list[float]:
    """
    Profile model's linear part performance.
    """
    print(f"Profiling linear part with S_list={S_list} ...")
    result_path = self.model.engine_config.profile_result_path + "linr.json"

    if os.path.exists(result_path):
      with open(result_path, "r") as f:
        res = json.load(f)
        if res["S_list"] == S_list:
          return res["T_list"]

    T_list = []
    for S in tqdm(S_list):
      res = self._run_test_case(
        prefill_lens=[S]
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
    print(f"Profiling prefill part with S_list={S_list}...")
    result_path = self.model.engine_config.profile_result_path + "pref.json"

    if os.path.exists(result_path):
      with open(result_path, "r") as f:
        res = json.load(f)
        if res["S_list"] == S_list:
          return res["T_list"]

    T_list = []
    for S in tqdm(S_list):
      res1 = self._run_test_case(
        prefill_lens=[S]
      )
      res2 = self._run_test_case(
        prefill_lens=[S] * 2
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
    print(f"Profiling GPU attention part with N_list={N_list} ...")
    result_path = self.model.engine_config.profile_result_path + "gdec.json"

    if os.path.exists(result_path):
      with open(result_path, "r") as f:
        res = json.load(f)
        if res["N_list"] == N_list:
          return res["T_list"]

    T_list = []
    for N in tqdm(N_list):
      res = self._run_test_case(
        gpu_decode_lens=[N]
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
    N_list: list[int]
  ) -> list[list[float]]:
    """
    Profile model's CPU attention part performance.
    """
    print(f"Profiling CPU attention part with S_list={S_list}, N_list={N_list} ...")
    result_path = self.model.engine_config.profile_result_path + "cdec.json"

    if os.path.exists(result_path):
      with open(result_path, "r") as f:
        res = json.load(f)
        if res["S_list"] == S_list and res["N_list"] == N_list:
          return res["T_list"]
        
    T_list = []
    block_size = self.model.engine_config.block_size
    assert all(N % block_size == 0 for N in N_list), "N must be divisible by block size"
    for S in tqdm(S_list):
      T_list.append([])
      for N in N_list:
        NB = N // block_size
        res = self._run_test_case(
          # Divide N into S segments as even as possible
          cpu_decode_lens=[NB // S * block_size] * (S - NB % S) + [(NB // S + 1) * block_size] * (NB % S),
        )
        T_list[-1].append(ModelPerfResult.mean(res, "avg_cdec_time"))

    T_array = np.array(T_list)

    plt.figure(figsize=(16, 12))
    ax = plt.axes(projection='3d')
    ax.plot_surface(
      np.outer(S_list, np.ones(len(N_list))),
      np.outer(np.ones(len(S_list)), N_list),
      T_array,
      label = "CPU"
    )

    ax.set_xlim(0)
    ax.set_ylim(0)
    # ax.set_xticks(S_list)
    # ax.set_yticks([0] + N_list)
    # ax.set_zticks([0.2 * i for i in range(0, 16)])
    ax.set_xlabel("S_c")
    ax.set_ylabel("N_c")
    ax.set_zlabel("T(ms)")
    plt.savefig(self.model.engine_config.profile_result_path + "cdec.png")
    plt.close()

    with open(result_path, "w") as f:
      json.dump({
        "S_list": S_list,
        "N_list": N_list,
        "T_list": T_list
      }, f, indent=2)

    return T_list

  def _profile_lnch(
    self,
    S_list: list[int]
  ) -> list[float]:
    """
    Profile model's kernel launch time.
    """
    print(f"Profiling kernel launch time with S_list={S_list} ...")
    result_path = self.model.engine_config.profile_result_path + "lnch.json"

    if os.path.exists(result_path):
      with open(result_path, "r") as f:
        res = json.load(f)
        if res["S_list"] == S_list:
          return res["T_list"]

    T_list = []
    for S in tqdm(S_list):
      res = self._run_test_case(
        prefill_lens=[S // 2 - 8],
        gpu_decode_lens=[512] * 4,
        cpu_decode_lens=[512] * 4,
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

  def get_linr_T(self, S: int) -> float:
      """
      Get the linear time for iteration width S, using linear interpolation
      """
      assert S <= self.linr_S_arr[-1], f"Iteration width {S} exceeds the maximum {self.linr_S_arr[-1]}"
      return np.interp(S, self.linr_S_arr, self.linr_T_arr)

  def get_pref_T(self, S: int) -> float:
      """
      Get the GPU prefilling time for iteration width S, using linear interpolation
      """
      assert S <= self.pref_S_arr[-1], f"Iteration width {S} exceeds the maximum {self.pref_S_arr[-1]}"
      return np.interp(S, self.pref_S_arr, self.pref_T_arr)
    
  def get_gdec_T(self, N: int) -> float:
      """
      Get the GPU decoding time for number of tokens N, using linear interpolation
      """
      assert N <= self.gdec_N_arr[-1], f"Number of tokens {N} exceeds the maximum {self.gdec_N_arr[-1]}"
      return np.interp(N, self.gdec_N_arr, self.gdec_T_arr)
  
  def get_cdec_T(self, S: int, N: int) -> float:
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
  
  def get_cpu_remaining_capacity(self, s: int, x_c: int, n_c: int):
      """
      Get the remaining CPU capacity for a batch with 
      - (opposite batch's) iteration width s
      - number of CPU decoding requests x_c,
      - total number of tokens n_c
      """
      return self._get_linr_T(s) - self._get_cdec_T(x_c, n_c) - self.kernel_launch_time
