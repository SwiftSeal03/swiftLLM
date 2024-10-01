import time
from pprint import pprint
from transformers import AutoTokenizer
from swiftllm.worker.model import LlamaModel, ModelPerfResult
from swiftllm.worker.profiler import ModelProfiler
from swiftllm.engine_config import EngineConfig

def init_model():
  global model, profiler
  engine_config = EngineConfig(
    model_path = "/home/ubuntu/weights/Llama-3-8B-Instruct-Gradient-1048k",
    use_dummy = False,
    
    block_size = 16,
    gpu_mem_utilization = 0.995,
    num_cpu_blocks = 15000,
    max_seqs_in_block_table = 1024,
    max_blocks_per_seq = 512,

    # The following are not used in the offline example
    max_batch_size = 512,
    max_prefill_tokens = 2500,
    max_tokens_in_batch = 3000,

    library_path="/home/ubuntu/pacpu/build/libpacpu.so",
    profile_result_path="/home/ubuntu/swiftLLM/profile_results/",
  )

  start_time = time.perf_counter()

  # Initialize the model
  # For instructions on how to initialize the model, see comments in swiftllm/worker/model.py
  model = LlamaModel(engine_config)
  model.load_weights()
  num_blocks = 1700
  print("Number of blocks:", num_blocks)
  model.init_kvcache_and_swap(num_blocks)

  model_creation_time = time.perf_counter() - start_time
  print(f"Model creation time: {model_creation_time:.2f} seconds")

  profiler = ModelProfiler(
    model, 
    nwarmup=2, 
    nrepeat=3
  )

if __name__ == "__main__":
  init_model()
  res = profiler._run_test_case(
    [1498],
    [2048],
    [2048],
  )
  print(res)