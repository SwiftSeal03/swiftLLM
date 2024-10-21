import os
import sys
import json
import subprocess
import time
import logging
import argparse

home = os.path.expanduser("~")
logger = logging.getLogger(__name__)
logging.basicConfig(filename=f"{home}/swiftLLM/evaluation/bench.log", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

server_proc = None

def start_server(name: str):
    """
    Start the server
    """
    # pylint: disable=global-statement
    global server_proc
    with open(f"{home}/swiftLLM/evaluation/config.json") as f:
        config = json.load(f)

    with open(f"{home}/swiftLLM/evaluation/server.log", "w") as f:
        if name == "vllm":
            server_proc = subprocess.Popen(
                [
                    "numactl", "-N", "0", "-m", "0",
                    "vllm", "serve", f"{home}/weights/{config['model']}/", "--port", "8000",
                    "--block-size", str(config["block_size"]),
                    "--max-model-len", str(config["max_model_len"]),
                    "--max-num-seqs", str(config["max_num_seqs"]),
                    "--max-num-batched-tokens", str(config["num_gpu_blocks_override"] * config["block_size"]),
                    "--tensor-parallel-size", str(config["tensor_parallel_size"]),
                    # "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
                    "--num-gpu-blocks-override", str(config["num_gpu_blocks_override"]),
                    "--swap-space", str(config["swap_space"] / config["tensor_parallel_size"]),
                    "--enforce-eager",
                    "--disable-sliding-window",
                    "--disable-async-output-proc",
                    "--disable-custom-all-reduce",
                    "--disable-frontend-multiprocessing",
                    "--tokenizer-pool-size", "1",
                    "--enable-chunked-prefill",
                    "--preemption-mode", "swap",
                    "--dtype", "float16"
                ], 
                env=os.environ | {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
                stdout=f,
                stderr=f
            )
        elif name == "ours" or name == "base":
            nl = config['num_layers']
            if name == "base":
                cmd=["--always-use-gpu"]
                num_gpu_blocks_override = config["num_gpu_blocks_override"]
                swap_space = config["swap_space"] // 8
            else:
                cmd=["--extra-layer-for-cprf"]
                num_gpu_blocks_override = config["num_gpu_blocks_override"] * nl // (nl + 1)
                swap_space = config["swap_space"]

            cmd = [
                "numactl", "-N", "0", "-m", "0",
                sys.executable, "-m", "swiftllm.server.api_server",
                "--port", "8000",
                "--model-path", f"{home}/weights/{config['model']}/",
                "--block-size", str(config["block_size"]),
                "--max-blocks-per-seq", str((config["max_num_batched_tokens"] - 1) // config["block_size"] + 1),
                "--max-seqs-in-block-table", str(config["max_num_seqs"]),
                "--max-batch-size", str(config["max_num_seqs"]),
                "--max-tokens-in-batch", str(config["max_num_batched_tokens"]),
                "--tensor-parallel-degree", str(config["tensor_parallel_size"]),
                # "--gpu-mem-utilization", str(config["gpu_memory_utilization"]),
                "--num-gpu-blocks-override", str(num_gpu_blocks_override),
                "--swap-space", str(swap_space),
                "--library-path", f"{home}/pacpu/build/{config['library']}",
                "--profile-result-path", f"{home}/swiftLLM/profile_results/",
            ] + cmd

            server_proc = subprocess.Popen(
                cmd, 
                stdout=f,
                stderr=f
            )
        else:
            raise ValueError(f"Unknown server name: {name}")
        
        pid = server_proc.pid
        logger.info("Server started with pid %d", pid)
        for i in range(24 if name == "vllm" else 14):
            time.sleep(5)
            logger.info("Server starting, %d s passed ...", (i * 5))
        logger.info("Server started")


def stop_server():
    """
    Stop the server
    """
    assert server_proc is not None, "Server not started"
    server_proc.terminate()
    logger.info("Server stopped")


if __name__ == "__main__":
    # parse server name from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("server_name", type=str, help="Server name")
    args = parser.parse_args()

    start_server(args.server_name)
    input("Press Enter to stop the server ...")
    stop_server()

