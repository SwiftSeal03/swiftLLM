import time
import torch

from .model import LlamaModel, ModelForwardArgs, ModelPerfResult

class ModelProfiler:
  """
  A profiler for the Llama model.
  """
  def __init__(
    self, 
    model: LlamaModel, 
    base_tokens: list[int],
    nwarmup: int = 2,
    nrepeat: int = 3
  ):
    self.model = model
    self.base_tokens = base_tokens
    self.nrepeat = nrepeat
    self.nwarmup = nwarmup

    print("[fake_prefill] Initializing base KV entries")
    self.model.forward(ModelForwardArgs(
      [self.base_tokens],
      [0],
      []
    ))

  @torch.inference_mode()
  def fake_prefill(
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

  def run_test_case(
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
        self.fake_prefill(cpu_decode_ids, cpu_decode_lens)
        self.model.swap_out_seqs(cpu_decode_ids)
      if ngdec > 0:
        self.fake_prefill(gpu_decode_ids, gpu_decode_lens)

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