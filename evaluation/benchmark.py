import os
import asyncio
import time
import logging
import json
import random

import numpy as np
from tqdm import tqdm

# pylint: disable=import-error
from api_client import request_completions
from server import start_server, stop_server

home = os.path.expanduser("~")
input_dir = f"{home}/neo-data/tokens"

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f"{home}/swiftLLM/evaluation/bench.log", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

res_dir = f"{home}/swiftLLM/evaluation/results"
os.makedirs(res_dir, exist_ok=True)

api_url = "http://localhost:8000/v1/completions"
server_name = None

with open(f"{home}/swiftLLM/evaluation/config.json") as cf:
    config = json.load(cf)


async def request_completions_task(prompt: list[int], output_len: int):
    start = time.perf_counter()
    await request_completions(api_url, prompt, output_len)
    end = time.perf_counter()
    return start, end


async def run_test(
    prompts: list[list[int]],
    output_lens: list[int],
    res_prefix: str,
    rate: float = -1 # -1 means throughput test
):
    if rate > 0:
        res_file = f"{res_prefix}-lat-{str(rate).replace('.', '_')}.json"
    else:
        res_file = f"{res_prefix}-tp.json"

    if os.path.exists(res_file):
        logger.info("Test result file already exists: %s", res_file)
        with open(res_file, "r") as f:
            data = json.load(f)
            times = [(d["start"], d["end"]) for d in data]
    else:
        logger.info("Running test, saving results to %s", res_file)
        
        tasks = []
        np.random.seed(0)
        gaps = np.random.exponential(1 / rate, len(prompts)).tolist() if rate > 0 else [0] * len(prompts)
        for prompt, output_len in tqdm(zip(prompts, output_lens)):
            task = asyncio.create_task(request_completions_task(prompt, output_len))
            tasks.append(task)
            if rate > 0:
                await asyncio.sleep(gaps.pop(0))
        times = await asyncio.gather(*tasks)
        with open(res_file, "w") as f:
            json.dump([{
                "input_len": len(prompt),
                "output_len": output_len,
                "start": start,
                "end": end
            } for (start, end), prompt, output_len in zip(times, prompts, output_lens)], f, indent=4)

    if rate > 0:
        comp_times = [end - start for start, end in times]
        pertok_times = [comp_time / (len(prompt) + output_len) for comp_time, prompt, output_len in zip(comp_times, prompts, output_lens)]
        average_completion_time = sum(comp_times) / len(comp_times)
        average_pertok_time = sum(pertok_times) / len(pertok_times)
        logger.info("Average completion time: %.3f s", average_completion_time)
        logger.info("Average per-token completion time: %.3f s", average_pertok_time)
    else:
        n = len(prompts)
        req_end_times = sorted([end for _, end in times])
        req_end_times = req_end_times[n // 10: n - n // 10 * 3 + 1]
        throughput = (len(req_end_times) - 1) / (req_end_times[-1] - req_end_times[0])
        logger.info("Throughput: %.3f req/s", throughput)


def _get_rand_array(n: int, avg_val: int, ratio: float):
    """
    Get a random array with average value `avg_val`,

    all values are uniformly distributed in the range of [avg_val * (1 - ratio), avg_val * (1 + ratio)]
    """
    delta = int(avg_val * ratio)
    return [avg_val + random.randint(-delta, delta) for _ in range(n)]


def prepare_mock_test(
    nreqs: int,
    input_len: int,
    output_len: int
) -> tuple[list[list[int]], list[int], str]:
    input_lens = _get_rand_array(nreqs, input_len, 0.1)
    output_lens = _get_rand_array(nreqs, output_len, 0.1)
    prompts = [[10] * input_len for input_len in input_lens]
    res_file = f"{res_dir}/{server_name}-{nreqs}-{input_len}-{output_len}"
    return prompts, output_lens, res_file


def prepare_real_test(
    dataset_name: str
) -> tuple[list[list[int]], list[int], str]:
    input_file = f"{home}/neo-data/tokens/{dataset_name}-{config['model']}.json"
    with open(input_file, "r") as f:
        datas = json.load(f)[:2000]
        prompts = [data["prompt"] for data in datas]
        output_lens = [data["max_tokens"] for data in datas]
        
    res_file = f"{res_dir}/{server_name}-{dataset_name}"
    return prompts, output_lens, res_file


async def one_round(name: str):
    global server_name
    server_name = name
    start_server(server_name)
    # if name == "ours":
    #     await run_test(*prepare_real_test("arxiv"), 0.18)
    #     await run_test(*prepare_real_test("arxiv"), 0.20)
    #     await run_test(*prepare_real_test("arxiv"), 0.22)
    # await run_test(*prepare_real_test("azure_code"))
    await run_test(*prepare_real_test("azure_code"), 2.6)
    await run_test(*prepare_real_test("azure_code"), 2.8)
    await run_test(*prepare_real_test("azure_code"), 2.9)
    # await run_test(*prepare_mock_test(2000, 2000, 50))
    # await run_test(*prepare_mock_test(2000, 2000, 100))
    # await run_test(*prepare_mock_test(2000, 2000, 200))
    # await run_test(*prepare_mock_test(2000, 1000, 50))
    # await run_test(*prepare_mock_test(2000, 1000, 100))
    # await run_test(*prepare_mock_test(2000, 1000, 200))
    # await run_test(*prepare_mock_test(2000, 500, 50))
    # await run_test(*prepare_mock_test(2000, 500, 100))
    # await run_test(*prepare_mock_test(2000, 500, 200))
    stop_server()
    await asyncio.sleep(10)


async def main():
    await one_round("ours")
    await one_round("vllm")
    # await one_round("base")


if __name__ == "__main__":
    asyncio.run(main())
    