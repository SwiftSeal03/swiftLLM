import time
import argparse

import torch
from transformers import AutoTokenizer

import swiftllm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/ubuntu/weights/Llama-3-8B-Instruct-Gradient-1048k")
    parser.add_argument("--data_path", type=str, default="sample.txt")
    parser.add_argument("--num_cpu_blocks", type=int, default=25000)
    parser.add_argument("--offload_attn_to_cpu", "--cpu", action="store_true")
    parser.add_argument("--num_cpu_threads", "-t", type=int, default=8)
    args = parser.parse_args()

    torch.ops.load_library("/home/ubuntu/pacpu/build/libpacpu.so")

    engine_config = swiftllm.EngineConfig(
        model_path = args.model_path,
        use_dummy = False,
        
        gpu_mem_utilization = 0.99,
        num_cpu_blocks = args.num_cpu_blocks,
        max_seqs_in_block_table = 128,
        max_blocks_per_seq = 2048,

        # The following are not used in the offline example
        max_batch_size = 16,
        max_tokens_in_batch = 2048*16,

        offload_attn_to_cpu = args.offload_attn_to_cpu
    )

    torch.set_num_threads(args.num_cpu_threads)
    print(f"Using {torch.get_num_threads()} threads.")

    start_time = time.perf_counter()
    model = swiftllm.LlamaModel(engine_config)
    model.load_weights()
    if not engine_config.offload_attn_to_cpu:
        num_gpu_blocks = model.profile_num_blocks()
        print("Number of blocks:", num_gpu_blocks)
    else:
        num_gpu_blocks = 0
    model.init_kvcache_and_swap(num_gpu_blocks)
    model_creation_time = time.perf_counter()
    print(f"Model creation time: {model_creation_time - start_time:.2f} seconds")
    
    with open(args.data_path) as f:
        prompt = f.read()
    
    prompts = [prompt] * 60

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
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
    for _ in range(10):
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
