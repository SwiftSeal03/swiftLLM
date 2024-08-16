import time
from pprint import pprint
from transformers import AutoTokenizer
import swiftllm

import fake_prefill
import swiftllm.worker

model_path = "/home/ubuntu/weights/Llama-3-8B-Instruct-Gradient-1048k"
library_path = "/home/ubuntu/pacpu/build/libpacpu.so"

tokenizer, model, profiler = None, None, None

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
    num_cpu_blocks = 4000,
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
  num_blocks = 1750
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

def init():
  init_model()
  get_tokens()
  global profiler
  profiler = swiftllm.ModelProfiler(
    model, 
    base_tokens,
    nwarmup, 
    nrepeat
  )

if __name__ == "__main__":
  init()
  res = profiler.run_test_case(
    prefill_lens=[192],
    gpu_decode_lens=[384] * 20,
    cpu_decode_lens=[384] * 20,
    use_pipeline=True
  )
  pprint(res)