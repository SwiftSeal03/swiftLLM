import time
import argparse

from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer

import swiftllm

if __name__ == '__main__':
    model_path = "/home/ubuntu/weights/Llama-3-8B-Instruct-Gradient-1048k"
    data_path = "sample.txt"

    engine_config = swiftllm.EngineConfig(
        model_path = model_path,
        use_dummy = False,
        
        gpu_mem_utilization = 0.99,
        num_cpu_blocks = 0,
        max_seqs_in_block_table = 128,
        max_blocks_per_seq = 2048,

        # The following are not used in the offline example
        max_batch_size = 16,
        max_tokens_in_batch = 2048*16,

        offload_attn_to_cpu = False,
    )

    start_time = time.perf_counter()
    model = swiftllm.LlamaModel(engine_config)
    model.load_weights()
    if not engine_config.offload_attn_to_cpu:
        num_blocks = model.profile_num_blocks()
        print("Number of blocks:", num_blocks)
    else:
        num_blocks = 0
    model.init_kvcache_and_swap(num_blocks)
    model_creation_time = time.perf_counter()
    print(f"Model creation time: {model_creation_time - start_time:.2f} seconds")
    
    with open(data_path) as f:
        prompt = f.read()
    
    prompts = [prompt] * 10

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    outputs = []

    # Prompt phase
    print("Prefilling phase started...")
    input_ids = tokenizer(prompts)['input_ids']

    print(len(input_ids[0]))

    prompt_phase_outputs = model.forward(
        input_ids,
        list(range(0, len(prompts))),
        []
    )
    
    # print(tokenizer.batch_decode(prompt_phase_outputs, skip_special_tokens=True))
    outputs.append(prompt_phase_outputs)

    print("Decoding phase started...")
    seq_lens = [len(x) for x in input_ids]
    last_round_outputs = prompt_phase_outputs
    for _ in range(100):
        for i, _ in enumerate(prompts):
            seq_lens[i] += 1
        last_round_outputs = model.forward(
            [[x] for x in last_round_outputs],
            list(range(0, len(prompts))),
            seq_lens
        )
        # print(tokenizer.batch_decode(last_round_outputs, skip_special_tokens=True))
        outputs.append(last_round_outputs)
    
    for i, prompt in enumerate(prompts[:1]):
        output_tokens = [x[i] for x in outputs]
        output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
        print(f"{prompt}|{output_text}")
