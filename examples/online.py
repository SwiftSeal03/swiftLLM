import asyncio
import argparse
import time
from transformers import AutoTokenizer

import swiftllm

async def send_request_and_wait_streaming_mode(engine: swiftllm.Engine, tokenizer: AutoTokenizer, prompt: str, output_len: int):
    raw_request = swiftllm.RawRequest(prompt, output_len)
    output_token_ids = []
    token_latencies = []
    last_time = time.perf_counter()

    async for step_output in engine.add_request_and_stream(raw_request):
        output_token_ids.append(step_output.token_id)
        token_latencies.append(time.perf_counter() - last_time)
        last_time = time.perf_counter()

    print("---------------------------------")
    print(f"Prompt: {prompt}")
    print(f"Output: {tokenizer.decode(output_token_ids)}")
    print(f"Token latencies (ms): {[round(t*1000, 1) for t in token_latencies]}")

async def send_request_and_wait_non_streaming_mode(engine: swiftllm.Engine, tokenizer: AutoTokenizer, prompt: str, output_len: int):
    raw_request = swiftllm.RawRequest(prompt, output_len)
    (_, output_token_ids) = await engine.add_request_and_wait(raw_request)
    print("---------------------------------")
    print(f"Prompt: {prompt}")
    print(f"Output: {tokenizer.decode(output_token_ids)}")

async def main():
    parser = argparse.ArgumentParser()
    parser.description = """
        An example script to demonstrate how to use the swiftllm engine (both streaming and non-streaming mode)
    """
    parser.add_argument(
        "--model-path",
        help="Path to the model. Note: please download the model weights from HuggingFace in advance and specify the path here.",
        type=str,
        required=True
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
        num_cpu_blocks = 4096,
        max_seqs_in_block_table = 384,
        max_blocks_per_seq = 256,

        max_batch_size = 256,
        max_tokens_in_batch = 2048*16,

        library_path="/home/ubuntu/pacpu/build/libpacpu.so",
        profile_result_path="/home/ubuntu/swiftLLM/profile_results/",
    )

    prompt_and_output_lens = [
        ("Life blooms like a flower, far away", 10),
        ("one two three four five", 50),
        ("A B C D E F G H I J K L M N O P Q R S T U V", 5),
        ("To be or not to be,", 15),
    ]

    engine = swiftllm.Engine(engine_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    await engine.initialize()
    asyncio.create_task(engine.start_all_event_loops())

    tasks = []
    for prompt, output_len in prompt_and_output_lens:
        if is_streaming_mode:
            task = asyncio.create_task(send_request_and_wait_streaming_mode(engine, tokenizer, prompt, output_len))
        else:
            task = asyncio.create_task(send_request_and_wait_non_streaming_mode(engine, tokenizer, prompt, output_len))
        tasks.append(task)
        await asyncio.sleep(0.1)
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
