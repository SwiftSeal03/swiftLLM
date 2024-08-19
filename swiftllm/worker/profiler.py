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
  def __init__(
    self, 
    model: LlamaModel, 
    base_tokens: list[int] | None = None,
    nwarmup: int = 2,
    nrepeat: int = 3
  ):
    self.model = model
    self.base_tokens = base_tokens
    self.nrepeat = nrepeat
    self.nwarmup = nwarmup

    if not self.base_tokens:
      # Use dummy tokens
      self.base_tokens = [10] * model.engine_config.max_seq_len
    
    os.makedirs(model.engine_config.profile_result_path, exist_ok=True)

    print("[ModelProfiler] Initializing base KV entries")
    self.model.forward(ModelForwardArgs(
      [self.base_tokens],
      [0],
      []
    ))
  
  def __del__(self):
    self.model.engine_config.monitor_performance = False
    self.model.free_seqs_resources([0])

  @torch.inference_mode()
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

    assert all(0 < seq_id for seq_id in seq_ids), "Sequence ids must be positive"

    seq_ids = torch.tensor(seq_ids, dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")

    self.model.gpu_block_manager.allocate_blocks_for_seqs(seq_ids, seq_lens)

    nbase_blocks = (len(self.base_tokens) - 1) // self.model.engine_config.block_size + 1
    base_block_table = self.model.gpu_block_manager.block_table[0][:nbase_blocks]

    base_k = self.model.k_cache[base_block_table]
    base_v = self.model.v_cache[base_block_table]

    block_size = self.model.engine_config.block_size
    nblocks = [(seq_len - 1) // block_size + 1 for seq_len in seq_lens]

    for i, seq_id in enumerate(seq_ids):
      nblock = nblocks[i]
      assert nblock <= nbase_blocks, f"Sequence {seq_id} has {nblock} blocks, but only {nbase_blocks} are available"
      block_ids = self.model.gpu_block_manager.block_table[seq_id][:nblock]
      self.model.k_cache[block_ids] = base_k[:nblock]
      self.model.v_cache[block_ids] = base_v[:nblock]

  @torch.inference_mode()
  def _cpu_fake_prefill(
    self,
    seq_ids: list[int],
    seq_lens: list[int]
  ):
    # We should segregate it into parts in order not to exceed the maximum number of blocks
    # Note that this is just heuristic based
    step = 10
    for i in range(0, len(seq_ids), step):
      self._gpu_fake_prefill(seq_ids[i:i+step], seq_lens[i:i+step])
      self.model.swap_out_seqs(seq_ids[i:i+step])

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

    # Seq 0 is the base tokens
    offs = 1
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

      input_ids = [
        self.base_tokens[:seq_len] for seq_len in prefill_lens
      ] + [
        [self.base_tokens[seq_len - 1]] for seq_len in gpu_decode_lens
      ] + [
        [self.base_tokens[seq_len - 1]] for seq_len in cpu_decode_lens
      ]

      argss.append(ModelForwardArgs(
        input_ids,
        prefill_ids + gpu_decode_ids + cpu_decode_ids,
        gpu_decode_lens + cpu_decode_lens,
        len(cpu_decode_ids)
      ))

    for i in range(-self.nwarmup, self.nrepeat):
      self.model.engine_config.monitor_performance = i >= 0
      if i == 0:
        start = time.perf_counter()
      if not use_pipeline:
        self.model.forward(argss[0])
      else:
        self.model.forward_pipeline(argss)
      # Note that faked prefilling already allocates memory for the "new" token to be fed
      # into the model, so model.forward won't allocate memory for them and we only need
      # to free the resources for the prefill tokens.
      if all_prefill_ids:
        self.model.free_seqs_resources(all_prefill_ids)

    elapsed = time.perf_counter() - start
    print(f"Elapsed time: {elapsed * 1000 / self.nrepeat:.3f} ms")

    assert len(self.model.perf_results) == self.nrepeat
    res = self.model.get_perf_results(use_pipeline)
    self.model.free_seqs_resources(all_decode_ids)
    return res
  
  def profile_linear(
      self, 
      S_list: list[int]
  ) -> list[float]:
    """
    Profile model's linear part performance.
    """
    print(f"Profiling linear part ...")
    result_path = self.model.engine_config.profile_result_path + "linear.json"

    if os.path.exists(result_path):
      with open(result_path, "r") as f:
        res = json.load(f)
        if res["S_list"] == S_list:
          return res["T_list"]

    T_list = []
    for S in tqdm(S_list):
      T_list.append(self._run_test_case(
        prefill_lens=[S]
      ).avg_gpu_linr_time)

    plt.figure(figsize=(16, 12))
    plt.plot(S_list, T_list)
    plt.xlim(0)
    plt.ylim(0)
    plt.xlabel("S")
    plt.ylabel("T_l(ms)")
    plt.savefig(self.model.engine_config.profile_result_path + "linear.png")
    plt.close()

    with open(result_path, "w") as f:
      json.dump({
        "S_list": S_list,
        "T_list": T_list
      }, f, indent=2)

    return T_list
  
  def profile_cpu_attn(
    self,
    S_list: list[int],
    N_list: list[int]
  ) -> list[list[float]]:
    """
    Profile model's CPU attention part performance.
    """
    print(f"Profiling CPU attention part ...")
    result_path = self.model.engine_config.profile_result_path + "cpu_attn.json"

    if os.path.exists(result_path):
      with open(result_path, "r") as f:
        res = json.load(f)
        if res["S_list"] == S_list and res["N_list"] == N_list:
          return res["T_list"]

    T_list = []
    for S in tqdm(S_list):
      T_list.append([])
      for N in N_list:
        T_list[-1].append(self._run_test_case(
          # Divide N into S segments as even as possible
          cpu_decode_lens=[N // S] * (S - N % S) + [N // S + 1] * (N % S),
        ).avg_cpu_attn_time)

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
    plt.savefig(self.model.engine_config.profile_result_path + "cpu_attn.png")
    plt.close()

    with open(result_path, "w") as f:
      json.dump({
        "S_list": S_list,
        "N_list": N_list,
        "T_list": T_list
      }, f, indent=2)

    return T_list
