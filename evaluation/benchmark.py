import os
import asyncio
import time
import logging
import json
import random

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

with open(f"{home}/swiftLLM/evaluation/config.json") as f:
    config = json.load(f)


async def request_completions_task(prompt: list[int], output_len: int):
    start = time.perf_counter()
    await request_completions(api_url, prompt, output_len)
    end = time.perf_counter()
    return start, end


async def run_latency_test(
    prompts: list[list[int]],
    output_lens: list[int],
    res_prefix: str,
    rate: float
):
    res_file = f"{res_prefix}-lat-{str(rate).replace('.', '_')}.json"
    
    logger.info("Running latency test, saving results to %s", res_file)
    
    tasks = []
    for prompt, output_len in zip(prompts, output_lens):
        task = asyncio.create_task(request_completions_task(prompt, output_len))
        tasks.append(task)
        await asyncio.sleep(1 / rate)
    times = await asyncio.gather(*tasks)
    with open(res_file, "w") as f:
        json.dump([{
            "input_len": len(prompt),
            "output_len": output_len,
            "start": start,
            "end": end
        } for (start, end), prompt, output_len in zip(times, prompts, output_lens)], f, indent=4)

    comp_times = [end - start for start, end in times]
    pertok_times = [comp_time / (len(prompt) + output_len) for comp_time, prompt, output_len in zip(comp_times, prompts, output_lens)]
    average_completion_time = sum(times) / len(times)
    average_pertok_time = sum(pertok_times) / len(pertok_times)
    logger.info("Average completion time: %.3f s", average_completion_time)
    logger.info("Average per-token completion time: %.3f s", average_pertok_time)


async def run_throughput_test(
  prompts: list[list[int]],
  output_lens: list[int],
  res_prefix: str
):
    res_file = f"{res_prefix}-tp.json"
    logger.info("Running throughput test, saving results to %s", res_file)
    
    tasks = []
    for prompt, output_len in zip(prompts, output_lens):
        task = asyncio.create_task(request_completions_task(prompt, output_len))
        tasks.append(task)
    req_times = await asyncio.gather(*tasks)
    req_end_times = sorted([end for _, end in req_times])
    with open(res_file, "w") as f:
        deltas = [req_end_times[i] - req_end_times[i-1] for i in range(1, len(req_end_times))]
        json.dump({
            "req_end_deltas": deltas
        }, f, indent=4)
    
    # Omit first 10% and last 30% of requests to leave out warm-up and cool-down periods
    n = len(prompts)
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
        datas = json.load(f)[:500]
        prompts = [data["prompt"] for data in datas]
        output_lens = [data["max_tokens"] for data in datas]
        
    res_file = f"{res_dir}/{server_name}-{dataset_name}"
    return prompts, output_lens, res_file


async def main():
    global server_name
    server_name = "vllm"
    # start_server(server_name)
    # await run_mock_throughput_test(1, [10] * 19900, 10)
    # await run_throughput_test(*prepare_real_test("arxiv"))
    await run_latency_test(*prepare_real_test("arxiv"), 0.18)
    await run_latency_test(*prepare_real_test("arxiv"), 0.20)
    await run_latency_test(*prepare_real_test("arxiv"), 0.22)
    # stop_server()


if __name__ == "__main__":
    asyncio.run(main())
    