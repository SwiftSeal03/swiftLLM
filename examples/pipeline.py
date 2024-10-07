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
    parser.add_argument(
        "--profile-result-path",
        help="Path to folder of profiling results",
        type=str,
        default="/home/ubuntu/swiftLLM/profile_results/"
    )
    model_path = parser.parse_args().model_path
    library_path = parser.parse_args().library_path
    profile_result_path = parser.parse_args().profile_result_path

    engine_config = swiftllm.EngineConfig(
        model_path = model_path,
        use_dummy = False,

        block_size = 16,
        gpu_mem_utilization = 0.995,
        num_cpu_blocks = 4000,
        num_gpu_blocks = 1700,
        max_seqs_in_block_table = 1024,
        max_blocks_per_seq = 512,

        # The following are not used in the offline example
        max_batch_size = 512,
        max_prefill_tokens = 2048*16,
        max_tokens_in_batch = 2048*16,

        library_path=library_path,
        profile_result_path=profile_result_path,

        monitor_performance=False,
    )

    start_time = time.perf_counter()

    # Initialize the model
    # For instructions on how to initialize the model, see comments in swiftllm/worker/model.py
    model = swiftllm.LlamaModel(engine_config)
    model.load_weights()
    num_blocks = 1700
    # num_blocks = swiftllm.ModelProfiler(model).profile_num_blocks()
    print("Number of blocks:", num_blocks)
    model.init_kvcache_and_swap(num_blocks)

    model_creation_time = time.perf_counter() - start_time
    print(f"Model creation time: {model_creation_time:.2f} seconds")

    ngpu_prompts = 40
    ncpu_prompts = 40
    nprompts = ncpu_prompts + ngpu_prompts
    with open("/home/ubuntu/swiftLLM/examples/example.txt", "r") as f:
        prompt = ' '.join(f.readlines())
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    outputs = []

    # Prompt phase
    input_ids = tokenizer(prompt)['input_ids']
    reqs = [None] * nprompts
    if ncpu_prompts:
        batch = swiftllm.SubBatch()
        for i in range(ngpu_prompts // 2, nprompts // 2):
            reqs[i] = swiftllm.create_request(input_ids, i)
            batch.add_pref(reqs[i], is_gpu=False)
        model.forward(batch)
        batch = swiftllm.SubBatch()
        for i in range(nprompts // 2 + ngpu_prompts // 2, nprompts):
            reqs[i] = swiftllm.create_request(input_ids, i)
            batch.add_pref(reqs[i], is_gpu=False)
        model.forward(batch)
    if ngpu_prompts:
        batch = swiftllm.SubBatch()
        for i in list(range(ngpu_prompts // 2)) + list(range(nprompts // 2, nprompts // 2 + ngpu_prompts // 2)):
            reqs[i] = swiftllm.create_request(input_ids, i)
            batch.add_pref(reqs[i], is_gpu=True)
        model.forward(batch)

    # model.turn_on_perf_monitor()
    for _ in range(10):
        batches = [swiftllm.SubBatch() for _ in range(2)]
        for i in range(ngpu_prompts // 2):
            batches[0].add_gdec(reqs[i])
        for i in range(ngpu_prompts // 2, nprompts // 2):
            batches[1].add_cdec(reqs[i])
        for i in range(nprompts // 2, nprompts // 2 + ngpu_prompts // 2):
            batches[1].add_gdec(reqs[i])
        for i in range(nprompts // 2 + ngpu_prompts // 2, nprompts):
            batches[0].add_cdec(reqs[i])
        reqs.append(swiftllm.create_request(input_ids, len(reqs)))
        batches[0].add_pref(reqs[-1], is_gpu=True)
        reqs.append(swiftllm.create_request(input_ids, len(reqs)))
        batches[1].add_pref(reqs[-1], is_gpu=True)
        start = time.perf_counter()
        model.forward_pipeline(batches)
        end = time.perf_counter()
        print(f"E2E decoding time: {(end - start) * 1000:.4f} ms")
    
    for i in range(nprompts):
        if i == 0 or i == nprompts - 1:
            output_text = tokenizer.decode(reqs[i].output_token_ids, skip_special_tokens=True)
            print(f"{prompt}|{output_text}")

    # res = model.flush_perf_results_and_turn_off_perf_monitor()
    # print(res)
