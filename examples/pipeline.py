import time
import argparse
from transformers import AutoTokenizer

import swiftllm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = """
        An example script to demonstrate how to use the swiftllm model executor directly for inferencing without using the engine
    """
    parser.add_argument(
        "--model-path",
        help="Path to the model. Note: please download the model weights from HuggingFace in advance and specify the path here.",
        type=str,
        default="/home/ubuntu/weights/Llama-3-8B-Instruct-Gradient-1048k"
    )
    parser.add_argument(
        "--library-path",
        help="Path to the shared library",
        type=str,
        default="/home/ubuntu/pacpu/build/libpacpu.so"
    )
    model_path = parser.parse_args().model_path
    library_path = parser.parse_args().library_path

    engine_config = swiftllm.EngineConfig(
        model_path = model_path,
        use_dummy = False,
        
        block_size = 16,
        gpu_mem_utilization = 0.995,
        num_cpu_blocks = 4000,
        max_seqs_in_block_table = 256,
        max_blocks_per_seq = 2048,

        # The following are not used in the offline example
        max_batch_size = 16,
        max_tokens_in_batch = 2048*16,

        library_path=library_path,

        monitor_performance=True    
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
    
    ngpu_prompts = 30
    ncpu_prompts = 40
    nprompts = ncpu_prompts + ngpu_prompts
    with open("example.txt", "r") as f:
        prompts = f.readlines() * nprompts
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    outputs = []

    # Prompt phase
    input_ids = tokenizer(prompts)['input_ids']
    gpu_seq_ids = list(range(ngpu_prompts // 2)) + list(range(nprompts // 2, nprompts // 2 + ngpu_prompts // 2))
    cpu_seq_ids = list(range(ngpu_prompts // 2, nprompts // 2)) + list(range(nprompts // 2 + ngpu_prompts // 2, nprompts))
    if cpu_seq_ids:
        cpu_prompt_outputs0 = model.forward(swiftllm.ModelForwardArgs(
            input_ids[ngpu_prompts:ngpu_prompts + ncpu_prompts // 2],
            cpu_seq_ids[:ncpu_prompts // 2],
            []
        ))
        model.swap_out_seqs(cpu_seq_ids[:ncpu_prompts // 2])
        cpu_prompt_outputs1 = model.forward(swiftllm.ModelForwardArgs(
            input_ids[ngpu_prompts + ncpu_prompts // 2:],
            cpu_seq_ids[ncpu_prompts // 2:],
            []
        ))
        model.swap_out_seqs(cpu_seq_ids[ncpu_prompts // 2:])
    prompt_phase_outputs = []
    if gpu_seq_ids:
        prompt_phase_outputs = model.forward(swiftllm.ModelForwardArgs(
            input_ids[:ngpu_prompts],
            gpu_seq_ids,
            []
        ))
    if cpu_seq_ids:
        prompt_phase_outputs.extend(cpu_prompt_outputs0 + cpu_prompt_outputs1)
    outputs.append(prompt_phase_outputs)

    seq_lens = [len(x) for x in input_ids]
    last_round_outputs = prompt_phase_outputs
    for i in range(10):
        start = time.perf_counter()
        for j in range(nprompts):
            seq_lens[j] += 1
        last_round_outputs0 = last_round_outputs[:nprompts // 2]
        last_round_outputs1 = last_round_outputs[nprompts // 2:]
        argss = [None] * 2
        for j in range(2):
            argss[j] = swiftllm.ModelForwardArgs(
                [input_ids[0]] + 
                [[x] for x in last_round_outputs0],
                [nprompts + i * 2 + j] + 
                list(range(0, nprompts // 2) if j == 0 else range(nprompts // 2, nprompts)),
                seq_lens[:nprompts // 2] if j == 0 else seq_lens[nprompts // 2:],
                cpu_num_decoding_seqs=ncpu_prompts // 2
            )
        last_round_outputs = model.forward_pipeline(argss)
        last_round_outputs = last_round_outputs[1:nprompts // 2 + 1] + last_round_outputs[nprompts // 2 + 2:]
        # print(tokenizer.batch_decode(last_round_outputs, skip_special_tokens=True))
        outputs.append(last_round_outputs)
        end = time.perf_counter()
        print(f"E2E decoding time: {(end - start) * 1000:.4f} ms")
    
    for i, prompt in enumerate(prompts):
        output_tokens = [x[i] for x in outputs]
        output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
        if i == 0 or i == nprompts - 1:
            print(f"{prompt}|{output_text}")
