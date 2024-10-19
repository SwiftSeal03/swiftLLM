import os
import json
import asyncio
from fastapi import HTTPException
import aiohttp
from transformers import AutoTokenizer

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

home = os.path.expanduser("~")
with open(f"{home}/swiftLLM/evaluation/config.json") as f:
    config = json.load(f)

model_path = f"/home/aleria/weights/{config['model']}/"
tokenizer = AutoTokenizer.from_pretrained(model_path)

async def request_completions(
    api_url: str,
    prompt: str | list[int],
    output_len: int
):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:        
        payload = {
            "model": model_path,
            "prompt": prompt,
            "max_tokens": output_len,
            "temperature": 0.0,
            "ignore_eos": True
        }

        async with session.post(url=api_url, json=payload) as response:
            if response.status != 200:
                raise HTTPException(status_code=response.status, detail=await response.text())
            data = json.loads(await response.text())

    return data['choices'][0]['text']

async def main():
    url = "http://localhost:8000/v1/completions"
    prompts = [
        "The United States was founded in",
        "The capital of France is",
        "The capital of Italy is",
    ]
    output_len = 100
    tasks = [asyncio.create_task(request_completions(url, prompt, output_len)) for prompt in prompts]
    
    results = await asyncio.gather(*tasks)
    for prompt, result in zip(prompts, results):
        print(f"Prompt: {prompt}\nResult: {result}\n")

    prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]
    tasks = [asyncio.create_task(request_completions(url, prompt, output_len)) for prompt in prompt_tokens]
    results = await asyncio.gather(*tasks)
    for prompt, result in zip(prompts, results):
        print(f"Prompt tokens: {prompt}\nResult: {result}\n")

if __name__ == "__main__":
    asyncio.run(main())
