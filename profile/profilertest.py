"""
This script is used to test the performance of the model on a given test case.
"""

import time
import json

from swiftllm.worker.model import LlamaModel, ModelPerfResult
from swiftllm.worker.profiler import ModelProfiler
from swiftllm.engine_config import EngineConfig
from swiftllm.structs import BatchMetaData

def init_model():
    global model, profiler
    engine_config = EngineConfig(
        model_path = "/home/ubuntu/weights/Llama-3-8B-Instruct-Gradient-1048k",
        use_dummy = False,
        
        block_size = 16,
        gpu_mem_utilization = 0.995,
        num_gpu_blocks = 1700,
        num_cpu_blocks = 15000,
        max_seqs_in_block_table = 2048,
        max_blocks_per_seq = 1024,

        # The following are not used in the offline example
        max_batch_size = 512,
        max_prefill_tokens = 10000,
        max_tokens_in_batch = 20000,

        library_path="/home/ubuntu/pacpu/build/libpacpu.so",
        profile_result_path="/home/ubuntu/swiftLLM/profile_results/",
    )

    start_time = time.perf_counter()

    # Initialize the model
    # For instructions on how to initialize the model, see comments in swiftllm/worker/model.py
    model = LlamaModel(engine_config)
    model.load_weights()
    profiler = ModelProfiler(model)
    model.init_kvcache_and_swap()

    model_creation_time = time.perf_counter() - start_time
    print(f"Model creation time: {model_creation_time:.2f} seconds")

    profiler.init_profile_tables()

def run_test_case(
    pref_lens: list[list[int]],
    gdec_lens: list[list[int]],
    cdec_lens: list[list[int]]
):
    nbatches = len(pref_lens)
    batchmds = [BatchMetaData(profiler.pp) for _ in range(nbatches)]

    for i in range(nbatches):
        for l in pref_lens[i]:
            batchmds[i].add_pref(l)

        for l in gdec_lens[i]:
            batchmds[i].add_gdec(l)

        for l in cdec_lens[i]:
            batchmds[i].add_cdec(l)

    real_res = ModelPerfResult.mean_all(profiler._run_test_case(
        pref_lens, gdec_lens, cdec_lens
    ))

    print([batchmd.s for batchmd in batchmds])

    print("Predicted result:")
    print(json.dumps({
        "avg_linr_time": [batchmds[i].linr_T for i in range(nbatches)],
        "avg_pref_time": [batchmds[i].pref_T for i in range(nbatches)],
        "avg_gdec_time": [batchmds[i].gdec_T for i in range(nbatches)],
        "avg_cdec_time": [batchmds[i].cdec_T for i in range(nbatches)],
        "avg_lnch_time": [batchmds[i].lnch_T for i in range(nbatches)]
    }, indent=2))

    print("Real result:")
    print(json.dumps(real_res, indent=2))


if __name__ == "__main__":
    init_model()
    # run_test_case(
    #   pref_lens=[[526, 526, 526, 526],[]],
    #   gdec_lens=[[694, 693, 692, 691, 690, 689, 688, 687, 686, 685, 684, 683, 682, 681, 680, 679, 678, 677, 676, 675, 674, 673, 672, 671, 670, 669, 668, 667, 666, 665, 664, 663, 662, 661, 660, 659], []],
    #   cdec_lens=[
    #     [549, 548, 548, 547, 547, 546, 546, 545, 545, 544, 543, 543, 542, 542, 541, 541, 540, 540, 539, 539, 538, 538, 537, 537], 
    #     [542, 658, 559, 657, 575, 656, 590, 655, 605, 654, 619, 653, 632, 652, 649, 642, 642, 642, 642, 642, 642, 642, 642, 642, 641, 640, 639, 638, 637, 636, 635, 634, 633, 631, 630, 629, 628, 627, 626, 625, 624, 623, 622, 621, 620, 618, 617, 616, 615, 614, 613, 612, 611, 610, 609, 608, 607, 606, 604, 603, 602, 601, 600, 600, 599, 599, 598, 598, 597, 597, 596, 596, 595, 595, 594, 594, 593, 593, 592, 592, 591, 591, 590, 589, 589, 588, 588, 587, 587, 586, 586, 585, 585, 584, 584, 583, 583, 582, 582, 581, 581, 580, 580, 579, 579, 578, 578, 577, 577, 576, 576, 575, 574, 574, 573, 573, 572, 572, 572, 571, 571, 571, 570, 570, 570, 569, 569, 569, 568, 568, 568, 567, 567, 567, 566, 566, 566, 565, 565, 565, 564, 564, 564, 563, 563, 563, 562, 562, 562, 561, 561, 561, 560, 560, 560, 559, 559, 558, 558, 558, 557, 557, 557, 556, 556, 556, 556, 555, 555, 555, 555, 554, 554, 554, 554, 553, 553, 553, 553, 552, 552, 552, 552, 551, 551, 551, 551, 550, 550, 550, 550, 549, 549, 549, 548, 548, 547, 547, 546, 546, 545, 545, 544, 544, 544, 543, 543, 542, 541, 541, 540, 540, 539, 539, 538, 538, 537, 537, 536]
    #   ]
    # )
    run_test_case(
        pref_lens=[[128]],
        gdec_lens=[[128] * 10],
        cdec_lens=[[128] * 10]
    )
