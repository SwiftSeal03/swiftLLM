import time
from pprint import pprint
from transformers import AutoTokenizer
import swiftllm

import fake_prefill
import swiftllm.worker

model_path = "/home/ubuntu/weights/Llama-3-8B-Instruct-Gradient-1048k"
library_path = "/home/ubuntu/pacpu/build/libpacpu.so"

tokenizer, model = None, None

nwarmup = 2
nrepeat = 3

# All test cases will derive from this list of tokens
base_tokens = []


def init_model():
  global tokenizer, model
  tokenizer = AutoTokenizer.from_pretrained(model_path)

  engine_config = swiftllm.EngineConfig(
    model_path = model_path,
    use_dummy = False,
    
    block_size = 16,
    gpu_mem_utilization = 0.995,
    num_cpu_blocks = 1700,
    max_seqs_in_block_table = 512,
    max_blocks_per_seq = 2048,

    # The following are not used in the offline example
    max_batch_size = 16,
    max_tokens_in_batch = 2048*16,

    library_path=library_path
  )

  start_time = time.perf_counter()

  # Initialize the model
  # For instructions on how to initialize the model, see comments in swiftllm/worker/model.py
  model = swiftllm.LlamaModel(engine_config)
  model.load_weights()
  num_blocks = 1700
  print("Number of blocks:", num_blocks)
  model.init_kvcache_and_swap(num_blocks)

  model_creation_time = time.perf_counter() - start_time
  print(f"Model creation time: {model_creation_time:.2f} seconds")


def get_tokens():
  """
  Get the tokens from the example.txt file.

  Prefill the tokens, so that we can use the KV instead of prefilling
  on every test case.
  """

  global base_tokens
  with open("example.txt", "r") as f:
    prompt = ' '.join(f.readlines())
  
  base_tokens = tokenizer(prompt)['input_ids'] * 4


def run_test_case(
  prefill_lens: list[int] = [],
  gpu_decode_lens: list[int] = [],
  cpu_decode_lens: list[int] = [],
  use_pipeline: bool = False
) -> swiftllm.ModelPerfResult:
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
      fake_prefill.prefill(cpu_decode_ids, cpu_decode_lens)
      model.swap_out_seqs(cpu_decode_ids)
    if ngdec > 0:
      fake_prefill.prefill(gpu_decode_ids, gpu_decode_lens)

    input_ids = [
      base_tokens[:seq_len] for seq_len in prefill_lens
    ] + [
      [base_tokens[seq_len - 1]] for seq_len in gpu_decode_lens
    ] + [
      [base_tokens[seq_len - 1]] for seq_len in cpu_decode_lens
    ]

    argss.append(swiftllm.ModelForwardArgs(
      input_ids,
      prefill_ids + gpu_decode_ids + cpu_decode_ids,
      gpu_decode_lens + cpu_decode_lens,
      len(cpu_decode_ids)
    ))

  for i in range(-nwarmup, nrepeat):
    if i == 0:
        start = time.perf_counter()
    model.engine_config.monitor_performance = i >= 0
    if not use_pipeline:
      model.forward(argss[0])
    else:
      model.forward_pipeline(argss)
    # Note that faked prefilling already allocates memory for the "new" token to be fed
    # into the model, so model.forward won't allocate memory for them and we only need
    # to free the resources for the prefill tokens.
    if all_prefill_ids:
      model.free_seqs_resources(all_prefill_ids)

  elapsed = time.perf_counter() - start
  # print(f"Time taken: {elapsed * 1000/nrepeat:.2f} ms")

  assert len(model.perf_results) == nrepeat
  res = model.get_perf_results(use_pipeline)
  model.free_seqs_resources(all_decode_ids)
  return res

def init():
  init_model()
  get_tokens()
  fake_prefill.init(model, base_tokens)

if __name__ == "__main__":
  init()
  res = run_test_case(
    prefill_lens=[192, 192],
    gpu_decode_lens=[384] * 40,
    cpu_decode_lens=[384] * 40
  )
  pprint(res)
  res = run_test_case(
    prefill_lens=[192],
    gpu_decode_lens=[384] * 20,
    cpu_decode_lens=[384] * 20,
    use_pipeline=True
  )
  pprint(res)