import swiftllm
import torch

model = None
base_tokens = []

def init(
  _model: swiftllm.LlamaModel,
  _base_tokens: list[int],
):
  global model, base_tokens
  model = _model
  base_tokens = _base_tokens

  print("[fake_prefill] Initializing base KV entries")
  model.forward(swiftllm.ModelForwardArgs(
    [base_tokens],
    [0],
    []
  ))

@torch.inference_mode()
def prefill(
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

  model.gpu_block_manager.allocate_blocks_for_seqs(seq_ids, seq_lens)

  nbase_blocks = (len(base_tokens) - 1) // model.engine_config.block_size + 1
  base_block_table = model.gpu_block_manager.block_table[0][:nbase_blocks]

  base_k = model.k_cache[base_block_table]
  base_v = model.v_cache[base_block_table]

  block_size = model.engine_config.block_size
  nblocks = [(seq_len - 1) // block_size + 1 for seq_len in seq_lens]

  for i, seq_id in enumerate(seq_ids):
    nblock = nblocks[i]
    assert nblock <= nbase_blocks, f"Sequence {seq_id} has {nblock} blocks, but only {nbase_blocks} are available"
    block_ids = model.gpu_block_manager.block_table[seq_id][:nblock]
    model.k_cache[block_ids] = base_k[:nblock]
    model.v_cache[block_ids] = base_v[:nblock]
