import asyncio
import argparse
import logging
import time
from transformers import AutoTokenizer

import swiftllm

logger = logging.getLogger(__name__)
logging.basicConfig(filename="e2e.log", level=logging.INFO)

async def send_request_and_wait_non_streaming_mode(engine: swiftllm.Engine, tokenizer: AutoTokenizer, prompt: str, output_len: int):
    raw_request = swiftllm.RawRequest(prompt, output_len)
    start = time.perf_counter()
    (_, output_token_ids) = await engine.add_request_and_wait(raw_request)
    end = time.perf_counter()
    return end - start
    # print(f"Prompt: {prompt}")
    # print(f"Output: {tokenizer.decode(output_token_ids)}")

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
    logger.info(f"Throughput test: input_len={promptlen}, output_len={output_len}, nrequests={nrequests}, gpu_only={gpu_only}")
    engine.engine_config.always_use_gpu = gpu_only
    tasks = []
    start = time.perf_counter()
    for i in range(nrequests):
        task = asyncio.create_task(send_request_and_wait_non_streaming_mode(engine, tokenizer, prompt, output_len))
        tasks.append(task)
    await asyncio.gather(*tasks)
    end = time.perf_counter()
    logger.info(f"Throughput: {nrequests / (end - start):.3f} reqs/s")

async def warm_up(
    prompt: str,
    engine: swiftllm.Engine, 
    tokenizer: AutoTokenizer
):
    logger.info("Warming up...")
    await run_throughput_test(400, prompt, 200, False, engine, tokenizer)
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
        num_cpu_blocks = 10000,
        max_seqs_in_block_table = 768,
        max_blocks_per_seq = 512,

        max_batch_size = 512,
        max_tokens_in_batch = 3072,

        library_path="/home/ubuntu/pacpu/build/libpacpu.so",
        profile_result_path="/home/ubuntu/swiftLLM/profile_results/",

        # always_use_gpu=True
    )

    engine = swiftllm.Engine(engine_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    await engine.initialize()
    asyncio.create_task(engine.start_all_event_loops())

    with open("example.txt", "r") as f:
        prompt = ' '.join(f.readlines()[:3])

    await warm_up(prompt, engine, tokenizer)

    for outlen in range(30, 81, 10):
        await run_throughput_test(2500, prompt, outlen, True, engine, tokenizer)
        await run_throughput_test(2500, prompt, outlen, False, engine, tokenizer)
    # for rate in [8, 10]:
    #     await run_latency_test(1200, prompt, 40, rate, True, engine, tokenizer)
    #     await run_latency_test(1200, prompt, 40, rate, False, engine, tokenizer)
    

if __name__ == "__main__":
    asyncio.run(main())
