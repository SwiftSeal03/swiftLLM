import asyncio
import argparse
import logging
import json
import time
import os
from transformers import AutoTokenizer

import swiftllm

logger = logging.getLogger(__name__)
logging.basicConfig(filename="e2e.log", level=logging.INFO)

async def send_request_and_wait_non_streaming_mode(engine: swiftllm.Engine, tokenizer: AutoTokenizer, prompt: str, output_len: int):
    raw_request = swiftllm.RawRequest(prompt, output_len)
    start = time.perf_counter()
    (_, output_token_ids) = await engine.add_request_and_wait(raw_request)
    end = time.perf_counter()
    print(f"Output: {tokenizer.decode(output_token_ids)}")
    return start, end

async def run_latency_test(
    nrequests: int,
    prompt: str,
    output_len: int,
    req_rate: float,
    gpu_only: bool,
    engine: swiftllm.Engine,
    tokenizer: AutoTokenizer
):
    promptlen = len(tokenizer.encode(prompt))
    logger.info(f"Latency test: input_len={promptlen}, output_len={output_len}, nrequests={nrequests}, req_rate={req_rate}, gpu_only={gpu_only}")
    engine.engine_config.always_use_gpu = gpu_only
    tasks = []
    for i in range(nrequests):
        task = asyncio.create_task(send_request_and_wait_non_streaming_mode(engine, tokenizer, prompt, output_len))
        tasks.append(task)
        await asyncio.sleep(1/req_rate)
    times = await asyncio.gather(*tasks)
    times = [end - start for start, end in times]
    average_completion_time = sum(times) / len(times)
    logger.info(f"Average completion time: {average_completion_time:.3f}s")

async def run_throughput_test(
    nrequests: int,
    prompt: str,
    output_len: int,
    gpu_only: bool,
    engine: swiftllm.Engine,
    tokenizer: AutoTokenizer
):
    promptlen = len(tokenizer.encode(prompt))
    data_file = f"../data/d0926/{nrequests}-{promptlen}-{output_len}-{gpu_only}.json"
    
    logger.info(f"Throughput test: input_len={promptlen}, output_len={output_len}, nrequests={nrequests}, gpu_only={gpu_only}")
    engine.engine_config.always_use_gpu = gpu_only
    tasks = []
    for i in range(nrequests):
        task = asyncio.create_task(send_request_and_wait_non_streaming_mode(engine, tokenizer, prompt, output_len))
        tasks.append(task)
    times = await asyncio.gather(*tasks)
    times = [end for _, end in times]
    with open(data_file, "w") as f:
        deltas = [times[i] - times[i-1] for i in range(1, len(times))]
        json.dump(deltas, f, indent=4)
    times.sort()
    # Omit first 10% and last 30% of requests to leave out warm-up and cool-down periods
    times = times[nrequests//10: -nrequests//10*3]
    throughput = (len(times) - 1) / (times[-1] - times[0])
    logger.info(f"Throughput: {throughput:.3f} reqs/s")

async def warm_up(
    prompt: str,
    engine: swiftllm.Engine, 
    tokenizer: AutoTokenizer
):
    logger.info("Warming up...")
    await run_throughput_test(300, prompt, 100, False, engine, tokenizer)
    logger.info("Warm up complete")


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
    is_streaming_mode = args.streaming

    engine_config = swiftllm.EngineConfig(
        model_path = model_path,
        use_dummy = False,
        
        block_size = 16,
        gpu_mem_utilization = 0.99,
        num_cpu_blocks = 15000,
        max_seqs_in_block_table = 1024,
        max_blocks_per_seq = 512,

        max_batch_size = 512,
        max_prefill_tokens = 1000,
        max_tokens_in_batch = 3000,

        library_path="/home/ubuntu/pacpu/build/libpacpu.so",
        profile_result_path="/home/ubuntu/swiftLLM/profile_results/",

        # always_use_gpu=True
    )

    engine = swiftllm.Engine(engine_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    await engine.initialize()
    asyncio.create_task(engine.start_all_event_loops())

    with open("example.txt", "r") as f:
        text = f.readlines()
        prompts = [' '.join(text[:i]) for i in range(1, len(text)+1, 2)]
    
    print([len(tokenizer.encode(prompt)) for prompt in prompts])

    await warm_up(prompts[6], engine, tokenizer)

    for prompt in prompts:
        if len(tokenizer.encode(prompt)) == 98:
            for outlen in range(500, 1500, 500):
                await run_throughput_test(1000, prompt, outlen, False, engine, tokenizer)
                # await run_throughput_test(1000, prompt, outlen, True, engine, tokenizer)
    # for rate in [8, 10]:
    #     await run_latency_test(1200, prompt, 40, rate, True, engine, tokenizer)
    #     await run_latency_test(1200, prompt, 40, rate, False, engine, tokenizer)
    

if __name__ == "__main__":
    asyncio.run(main())
