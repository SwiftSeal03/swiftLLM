"""
Offline example of using the swiftllm model executor directly for inferencing without using the engine.

Explicity uses pipeline mode.
"""
import os
import time
import argparse
from transformers import AutoTokenizer

import swiftllm
from swiftllm.structs import Request

# pylint: disable=missing-function-docstring
def forward_wrapper(model: swiftllm.LlamaModel, batch: swiftllm.SubBatch):
    batch.set_model_forward_args(model.model_config)
    output_tokens = model.forward(batch)
    Request.update_output(batch.all_reqs, output_tokens)

def main():
    home = os.path.expanduser("~")
    parser = argparse.ArgumentParser()
    parser.description = """
        An example script to demonstrate how to use the swiftllm model executor directly for inferencing without using the engine
    """
    parser.add_argument(
        "--model-path",
        help="Path to the model. Note: please download the model weights from HuggingFace in advance and specify the path here.",
        type=str,
        default=f"{home}/weights/Llama-2-7b-hf"
    )
    parser.add_argument(
        "--library-path",
        help="Path to the shared library",
        type=str,
        default=f"{home}/pacpu/build/libpacpu.so"
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
        num_gpu_blocks = 1700,
        num_cpu_blocks = 500,
        max_seqs_in_block_table = 1024,
        max_blocks_per_seq = 512,

        # The following are not used in the offline example
        max_batch_size = 512,
        max_prefill_tokens = 2048*16,
        max_tokens_in_batch = 2048*16,

        library_path=library_path,
        profile_result_path=profile_result_path,

        extra_layer_for_cprf=True
    )

    start_time = time.perf_counter()

    # Initialize the model
    # For instructions on how to initialize the model, see comments in swiftllm/worker/model.py
    model = swiftllm.LlamaModel(engine_config)
    model.load_weights()
    # num_blocks = swiftllm.ModelProfiler(model).profile_num_blocks()
    model.init_kvcache_and_swap()

    model_creation_time = time.perf_counter() - start_time
    print(f"Model creation time: {model_creation_time:.2f} seconds")

    ngpu_prompts = 10
    ncpu_prompts = 0
    nprompts = ncpu_prompts + ngpu_prompts
    with open(f"{home}/swiftLLM/examples/example.txt", "r") as f:
        prompt = ' '.join(f.readlines())
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Prompt phase
    input_ids = tokenizer(prompt)['input_ids']
    reqs = [None] * nprompts
    if ncpu_prompts:
        batch = swiftllm.SubBatch()
        for i in range(ngpu_prompts // 2, nprompts // 2):
            reqs[i] = swiftllm.create_request(input_ids, i)
            batch.add_pref(reqs[i], is_gpu=False)
        forward_wrapper(model, batch)

        batch = swiftllm.SubBatch()
        for i in range(nprompts // 2 + ngpu_prompts // 2, nprompts):
            reqs[i] = swiftllm.create_request(input_ids, i)
            batch.add_pref(reqs[i], is_gpu=False)
        forward_wrapper(model, batch)

    if ngpu_prompts:
        batch = swiftllm.SubBatch()
        for i in list(range(ngpu_prompts // 2)) + list(range(nprompts // 2, nprompts // 2 + ngpu_prompts // 2)):
            reqs[i] = swiftllm.create_request(input_ids, i)
            batch.add_pref(reqs[i], is_gpu=True)
        forward_wrapper(model, batch)
    print("Prompt phase done")

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
        reqs.append(swiftllm.create_request(input_ids, len(reqs)))
        batches[0].add_pref(reqs[-2], is_gpu=True)
        batches[1].add_pref(reqs[-1], is_gpu=True)

        for batch in batches:
            batch.set_model_forward_args(model.model_config)
        start = time.perf_counter()
        output_tokens = model.forward_pipeline(batches)
        end = time.perf_counter()
        Request.update_output(sum([b.all_reqs for b in batches], []), output_tokens)

        print(f"E2E decoding time: {(end - start) * 1000:.4f} ms")
    
    for i in range(nprompts):
        if i == 0 or i == nprompts - 1:
            output_text = tokenizer.decode(reqs[i].output_token_ids, skip_special_tokens=True)
            print(f"{prompt}|{output_text}")

    # res = model.flush_perf_results_and_turn_off_perf_monitor()
    # print(res)

if __name__ == '__main__':
    main()
