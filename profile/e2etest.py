"""
This script runs a series of tests to measure the performance of the model on a given test case.
"""

import asyncio
import argparse
import logging
import json
import time
from transformers import AutoTokenizer
import numpy as np

import swiftllm

logger = logging.getLogger(__name__)
logging.basicConfig(filename="e2e.log", level=logging.INFO)

data_prefix = "../data/d1008"
engine = None
tokenizer = None


last_output_token_ids = None
async def send_request_and_wait_non_streaming_mode(prompt: str, output_len: int):
    raw_request = swiftllm.RawRequest(prompt, output_len)
    start = time.perf_counter()
    (_, output_token_ids) = await engine.add_request_and_wait(raw_request)
    end = time.perf_counter()
    # print(f"Output: {tokenizer.decode(output_token_ids)}")
    # global last_output_token_ids
    # if last_output_token_ids and last_output_token_ids != output_token_ids:
    #     raise ValueError("Output token ids mismatch")
    # last_output_token_ids = output_token_ids
    return start, end

async def run_latency_test(
    nrequests: int,
    prompt: str,
    output_len: int,
    req_rate: float,
    gpu_only: bool,
):
    promptlen = len(tokenizer.encode(prompt))
    logger.info("Latency test: input_len=%d, output_len=%d, nrequests=%d, req_rate=%.2f, gpu_only=%s", promptlen, output_len, nrequests, req_rate, gpu_only)
    engine.engine_config.always_use_gpu = gpu_only
    tasks = []
    for i in range(nrequests):
        task = asyncio.create_task(send_request_and_wait_non_streaming_mode(prompt, output_len))
        tasks.append(task)
        await asyncio.sleep(1/req_rate)
    times = await asyncio.gather(*tasks)
    times = [end - start for start, end in times]
    average_completion_time = sum(times) / len(times)
    logger.info("Average completion time: %.3f s", average_completion_time)

async def run_throughput_test(
    prompts: list[str],
    output_lens: list[int],
    gpu_only: bool,
    data_file: str
):
    engine.engine_config.always_use_gpu = gpu_only
    
    tasks = []
    start = time.perf_counter()
    n = len(prompts)
    engine.itr_end_times = []
    engine.ntoken_of_itr = []
    for i, prompt, output_len in zip(range(n), prompts, output_lens):
        task = asyncio.create_task(send_request_and_wait_non_streaming_mode(prompt, output_len))
        tasks.append(task)
    req_times = await asyncio.gather(*tasks)
    req_end_times = sorted([end for _, end in req_times])
    itr_end_times = [time - start for time in engine.itr_end_times]
    with open(data_file, "w") as f:
        deltas = [req_end_times[i] - req_end_times[i-1] for i in range(1, len(req_end_times))]
        json.dump({
            "req_end_deltas": deltas,
            "itr_end_times": itr_end_times,
            "ntoken_of_itr": engine.ntoken_of_itr
        }, f, indent=4)
    # Omit first 10% and last 30% of requests to leave out warm-up and cool-down periods
    req_end_times = req_end_times[n // 10: - n // 10 * 3]
    throughput = (len(req_end_times) - 1) / (req_end_times[-1] - req_end_times[0])
    logger.info("Throughput: %.3f req/s", throughput)

async def run_mock_throughput_test(
    nrequests: int,
    prompt: str,
    output_len: int,
    gpu_only: bool
):
    promptlen = len(tokenizer.encode(prompt))
    data_file = f"{data_prefix}/{nrequests}-{promptlen}-{output_len}-{gpu_only}.json"
    logger.info("Mock throughput test: input_len=%d, output_len=%d, nrequests=%d, gpu_only=%s", promptlen, output_len, nrequests, gpu_only)
    await run_throughput_test([prompt] * nrequests, [output_len] * nrequests, gpu_only, data_file)

async def run_real_throughput_test(
    dataset_name: str,
    input_str_file: str,
    output_len_file: str,
    gpu_only: bool
):
    with open(input_str_file, "r") as f:
        prompts = f.readlines()
    output_lens = np.load(output_len_file).tolist()

    assert len(prompts) == len(output_lens), "Length mismatch between input_str_file and output_len_file"
    data_file = f"{data_prefix}/{dataset_name}-{gpu_only}.json"
    logger.info("Real throughput test: dataset_name=%s, gpu_only=%s", dataset_name, gpu_only)
    await run_throughput_test(prompts, output_lens, gpu_only, data_file)


async def main():
    parser = argparse.ArgumentParser()
    parser.description = """
        An example script to demonstrate how to use the swiftllm engine (both streaming and non-streaming mode)
    """
    parser.add_argument(
        "--model-path",
        help="Path to the model. Note: please download the model weights from HuggingFace in advance and specify the path here.",
        type=str,
        default="/home/ubuntu/weights/Llama-3-8B-Instruct-Gradient-1048k"
    )
    parser.add_argument(
        "--streaming",
        help="Use streaming mode",
        action="store_true"
    )
    args = parser.parse_args()
    model_path = args.model_path
    # is_streaming_mode = args.streaming

    engine_config = swiftllm.EngineConfig(
        model_path = model_path,
        use_dummy = False,
        
        block_size = 16,
        gpu_mem_utilization = 0.995,
        num_gpu_blocks = 1500,
        num_cpu_blocks = 15000,
        max_seqs_in_block_table = 2048,
        max_blocks_per_seq = 2048,

        max_batch_size = 512,
        max_prefill_tokens = 20000,
        max_tokens_in_batch = 20000,

        library_path="/home/ubuntu/pacpu/build/libpacpu.so",
        profile_result_path="/home/ubuntu/swiftLLM/profile_results/",

        # always_use_gpu=True
        extra_layer_for_cprf=True
    )

    global engine, tokenizer
    engine = swiftllm.Engine(engine_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    await engine.initialize()
    asyncio.create_task(engine.start_all_event_loops())

    with open("example.txt", "r") as f:
        text = f.readlines()
        prompts = [' '.join(text[:i]) for i in range(1, len(text)+1, 2)]
    
    print([len(tokenizer.encode(prompt)) for prompt in prompts])

    logger.info("Warming up...")
    await run_mock_throughput_test(300, prompts[9], 200, False)

    for gpu_only in [False, True]:
        await run_real_throughput_test(
            "arxiv-summarization-test", 
            "/home/ubuntu/arxiv-dataset/test_input_prompts.txt", 
            "/home/ubuntu/arxiv-dataset/test_output_tokens.npy", 
            gpu_only
        )

    # for prompt in prompts:
    #     if len(tokenizer.encode(prompt)) == 521:
    #         for outlen in range(100, 1000, 200):
    #             await run_mock_throughput_test(2000, prompt, outlen, False)
    #             await run_mock_throughput_test(2000, prompt, outlen, True)
    # for prompt in prompts:
    #     if len(tokenizer.encode(prompt)) == 99:
    #         for rate in [0.8 + i * 0.2 for i in range(6)]:
    #             await run_latency_test(1000, prompt, 40, rate, True, engine, tokenizer)
    #             await run_latency_test(1000, prompt, 40, rate, False, engine, tokenizer)
    

if __name__ == "__main__":
    asyncio.run(main())
