"""
Offline example of using the swiftllm model executor directly for inferencing without using the engine.

Explicity uses pipeline mode.
"""
import os
import time
import argparse
from transformers import AutoTokenizer

import swiftllm


if __name__ == '__main__':
    home = os.path.expanduser("~")
    tp = 2
    nparam = 70
    parser = argparse.ArgumentParser()
    parser.description = """
        An example script to demonstrate how to use the swiftllm model executor directly for inferencing without using the engine
    """
    parser.add_argument(
        "--model-path",
        help="Path to the model. Note: please download the model weights from HuggingFace in advance and specify the path here.",
        type=str,
        default=f"{home}/weights/Llama-2-{nparam}b-hf"
    )
    parser.add_argument(
        "--library-path",
        help="Path to the shared library",
        type=str,
        default=f"{home}/pacpu/build/libpacpu-llama2_{nparam}b-tp{tp}.so"
    )
    parser.add_argument(
        "--profile-result-path",
        help="Path to folder of profiling results",
        type=str,
        default=f"{home}/swiftLLM/profile_results/"
    )
    model_path = parser.parse_args().model_path
    library_path = parser.parse_args().library_path
    profile_result_path = parser.parse_args().profile_result_path

    engine_config = swiftllm.EngineConfig(
        model_path = model_path,
        use_dummy = False,

        block_size = 16,
        gpu_mem_utilization = 0.995,
        num_gpu_blocks = 1300,
        num_cpu_blocks = 500,
        max_seqs_in_block_table = 1024,
        max_blocks_per_seq = 512,

        # The following are not used in the offline example
        max_batch_size = 512,
        max_prefill_tokens = 20000,
        max_tokens_in_batch = 20000,

        library_path=library_path,
        profile_result_path=profile_result_path,

        extra_layer_for_cprf=True,

        tensor_parallel_degree=tp
    )

    start_time = time.perf_counter()

    # Initialize the model
    # For instructions on how to initialize the model, see comments in swiftllm/worker/model.py
    engine = swiftllm.Engine(engine_config)
    engine.initialize()
    print(f"Engine creation time: {time.perf_counter() - start_time:.2f} seconds")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    ngpu_prompts = 10
    ncpu_prompts = 10
    nprompts = ncpu_prompts + ngpu_prompts
    with open(f"{home}/swiftLLM/examples/example.txt", "r") as f:
        prompt = ''.join(f.readlines())

    # Prompt phase
    input_ids = tokenizer(prompt)['input_ids']
    reqs = [None] * nprompts
    gpu_req_ids = list(range(ngpu_prompts // 2)) + list(range(nprompts // 2, nprompts // 2 + ngpu_prompts // 2))
    gpu_reqs = []
    if ngpu_prompts:
        batch = swiftllm.SubBatch()
        for i in gpu_req_ids:
            reqs[i] = swiftllm.create_request(input_ids, i)
            batch.add_pref(reqs[i], is_gpu=True)
        gpu_reqs = [reqs[i] for i in gpu_req_ids]
        engine.step([batch])

    if ncpu_prompts:
        batch = swiftllm.SubBatch()
        for i in range(ngpu_prompts // 2, nprompts // 2):
            reqs[i] = swiftllm.create_request(input_ids, i)
            batch.add_pref(reqs[i], is_gpu=False)
        engine.step([batch])

        batch = swiftllm.SubBatch()
        for i in range(nprompts // 2 + ngpu_prompts // 2, nprompts):
            reqs[i] = swiftllm.create_request(input_ids, i)
            batch.add_pref(reqs[i], is_gpu=False)
        engine.step([batch])

    print("Prompt phase done")

    # engine.executor.turn_on_perf_monitor()
    for iteration in range(16):
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
        reqs.append(swiftllm.create_request(input_ids, len(reqs)))
        # batches[0].add_pref(reqs[-2], is_gpu=True)
        # batches[1].add_pref(reqs[-1], is_gpu=True)

        start = time.perf_counter()
        engine.step(batches)
        end = time.perf_counter()
        print(f"Iteration {iteration:3} E2E time: {(end - start) * 1000:.4f} ms")
    
    for i in range(nprompts):
        if i in (0, nprompts // 2 - 1, nprompts - 1):
            output_text = tokenizer.decode(reqs[i].output_token_ids, skip_special_tokens=True)
            print(f"{prompt}|{output_text}")
            print(reqs[i].output_token_ids)

    # res = engine.executor.flush_perf_results_and_turn_off_perf_monitor()
    # print(res)
