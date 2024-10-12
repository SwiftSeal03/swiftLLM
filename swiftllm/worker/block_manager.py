"""
BlockManager and Swapper

Contains initalization and transition logics for the KV cache.
"""

import torch
import swiftllm_c
from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.structs import Request

class BlockManager:
    """
    BlockManager - Manage the block table and free blocks on CPU / GPU

    This manager records the mapping from (sequence ID, block index) to block 
    ID (which we call `block_table`), and provides methods to allocate and free
    blocks.

    All tables (the block table, the `seq_num_blks`, and the free block
    list) are all maintained on the GPU, so that we can leverage custom Triton
    kernels for fast operations.

    We may split KV cache along layer dimension, so there may be multi sets of free blocks.
    However, the sequences share same namespace of sequence IDs, so we only need one set of
    block table and `seq_num_blks`.
    """

    def __init__(
        self, 
        device_name: str, 
        num_blocks: int, 
        max_seqs_in_block_table: int, 
        max_blocks_per_seq: int, 
        block_size: int,
        nsplits: int=1
    ):
        self.device_name = device_name
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.block_table_width = max_blocks_per_seq

        # seq_id |-> number of blocks allocated for this sequence
        self.seq_num_blks = torch.zeros(
            (max_seqs_in_block_table,),
            dtype=torch.int32,
            device='cpu'
        )
        # (seq_id, block_index) |-> block_id
        self.block_table = torch.empty(
            (max_seqs_in_block_table, max_blocks_per_seq),
            dtype=torch.int32,
            device=device_name
        )
        # block_id |-> whether this block is free or not
        self.num_free_blocks = [num_blocks] * nsplits
        self.is_block_free = [torch.ones(
            (num_blocks,),
            dtype=torch.bool,
            device='cpu'
        ) for _ in range(nsplits)]
    
    def _get_new_blk_ids(self, num_blocks: int, split_id: int=0) -> torch.Tensor:
        """
        Check the free block table and return the block IDs of the newly allocated blocks
        """
        if num_blocks == 0:
            return torch.empty(0, dtype=torch.int32)

        is_block_free = self.is_block_free[split_id]
        if num_blocks > self.num_free_blocks[split_id]:
            raise RuntimeError(
                f"No enough free blocks available on {self.device_name} ({self.num_blocks} in total, "
                f"{self.num_free_blocks[split_id]} free, {num_blocks} requested)"
            )
            
        selected_blocks = torch.nonzero(is_block_free)[:num_blocks].view(-1).to(dtype=torch.int32)
        self.num_free_blocks[split_id] -= num_blocks
        is_block_free[selected_blocks] = False
        return selected_blocks
    
    def alloc(self, reqs: list[Request], split_point: int=0) -> list[int]:
        """
        Allocate blocks for sequences, making sure that seq #i has at least 
        ceil(target_lengths[i] / block_size) blocks allocated.

        Those after split_point will be allocated in the first split, and the rest will be allocated in the second split.

        Return new blocks allocated for the sequences. (useful for swapping)
        """
        if not reqs:
            return []

        seq_ids = Request.get_ids(reqs)
        seq_lens = torch.tensor(Request.get_lens(reqs), dtype=torch.int32)
        tgt_num_blks = (seq_lens - 1) // self.block_size + 1
        seq_num_blks = self.seq_num_blks[seq_ids]

        assert all(seq_num_blks <= tgt_num_blks), \
            f"""(On {self.device_name}) Logic error: Some sequences have more blocks already allocated than needed.
                seq_ids: {seq_ids}, target_lens: {seq_lens}, target_num_blocks: {tgt_num_blks},
                seq_num_blks: {seq_num_blks}"""
        
        new_num_blks = tgt_num_blks - seq_num_blks
        new_blk_ids0 = self._get_new_blk_ids(torch.sum(new_num_blks[split_point:]), 0)
        new_blk_ids1 = self._get_new_blk_ids(torch.sum(new_num_blks[:split_point]), 1)
        new_blk_ids = torch.cat([new_blk_ids1, new_blk_ids0])

        seq_num_blks_list = seq_num_blks.tolist()
        new_num_blks_list = new_num_blks.tolist()
        blk_poss = [seq_ids[i] * self.block_table_width + j + seq_num_blks_list[i] for i, n in enumerate(new_num_blks_list) for j in range(n)]
        self.block_table.view(-1)[blk_poss] = new_blk_ids.to(device=self.device_name) # possibly on GPU
        self.seq_num_blks[seq_ids] = tgt_num_blks
        return new_blk_ids.tolist()

    def free(self, reqs: list[Request], split_id: int=0) -> list[int]:
        """
        Gather the block IDs allocated for the specified sequences and mark them as free

        Useful fow swapping in/out
        """
        if not reqs:
            return []
        
        seq_ids = Request.get_ids(reqs)
        seq_num_blks_list = self.seq_num_blks[seq_ids].tolist()
        blk_poss = [seq_ids[i] * self.block_table_width + j for i, n in enumerate(seq_num_blks_list) for j in range(n)]
        blk_ids = self.block_table.view(-1)[blk_poss] # possibly on GPU
        self.num_free_blocks[split_id] += len(blk_ids)
        self.is_block_free[split_id][blk_ids] = True
        self.seq_num_blks[seq_ids] = 0
        return blk_ids.tolist()

class Swapper:
    """
    Swapper - Manage the swapping of sequences in and out of the model

    This manager is responsible for swapping sequences in and out of the model.
    It maintains the block manager, and provides methods to swap sequences in
    and out.
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        model_config: LlamaModelConfig
    ):
        self.engine_config = engine_config
        self.model_config = model_config

        # Initialize KV cache
        kvcache_shape = (
            model_config.num_layers + engine_config.extra_layer_for_cprf,
            engine_config.num_gpu_blocks,
            model_config.num_kv_heads,
            engine_config.block_size,
            model_config.head_dim
        )
        # Here we use torch.zeros instead of torch.empty, since that torch.empty
        # has the possibility to contain NaNs, which will cause the model to output NaNs.
        self.k_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")
        self.v_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")

        # Initialize KV swap space
        kvswap_shape = (
            model_config.num_layers,
            engine_config.num_cpu_blocks,
            model_config.num_kv_heads,
            engine_config.block_size,
            model_config.head_dim
        )
        self.k_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu", pin_memory=True)
        self.v_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu", pin_memory=True)

        # Initialize CPU QKV buffer
        q_cpu_shape = (engine_config.max_batch_size, model_config.num_q_heads, model_config.head_dim)
        o_cpu_shape = (engine_config.max_batch_size, model_config.num_q_heads * model_config.head_dim)
        kv_cpu_shape = (engine_config.max_batch_size, model_config.num_kv_heads, model_config.head_dim)
        self.q_cpu = torch.zeros(q_cpu_shape, dtype=torch.float16, device="cpu", pin_memory=True)
        self.k_cpu = torch.zeros(kv_cpu_shape, dtype=torch.float16, device="cpu", pin_memory=True)
        self.v_cpu = torch.zeros(kv_cpu_shape, dtype=torch.float16, device="cpu", pin_memory=True)
        # We store float32 tensors for the output, but convert them to float16 after copying back to GPU
        self.o_cpu = torch.zeros(o_cpu_shape, dtype=torch.float32, device="cpu", pin_memory=True)

        # Initialize block managers
        self.gpu_block_manager = BlockManager(
            "cuda",
            engine_config.num_gpu_blocks,
            engine_config.max_seqs_in_block_table,
            engine_config.max_blocks_per_seq,
            engine_config.block_size,
            2 if engine_config.extra_layer_for_cprf else 1
        )
        self.cpu_block_manager = BlockManager(
            "cpu",
            engine_config.num_cpu_blocks,
            # 1000,
            engine_config.max_seqs_in_block_table,
            engine_config.max_blocks_per_seq,
            engine_config.block_size
        )

    def _initiate_swap(
        self,
        reqs: list[Request],
        is_swap_in: bool,
        use_itm: bool = False # Only true when swapping out from intermediate cache to CPU
    ) -> tuple[list[int], list[int]]:
        """
        Do all the set-up work for swapping in/out sequences.
        Returns src and dst block ids.
        """
        if not reqs:
            return [], []
        src_block_manager = self.cpu_block_manager if is_swap_in else self.gpu_block_manager
        dst_block_manager = self.gpu_block_manager if is_swap_in else self.cpu_block_manager
        src_blk_ids = src_block_manager.free(reqs, 1 if use_itm else 0)
        dst_blk_ids = dst_block_manager.alloc(reqs)
        return src_blk_ids, dst_blk_ids
    
    def _swap(
        self,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        is_swap_in: bool,
        gpu_layer: int,
        cpu_layer: int
    ):
        swiftllm_c.swap_blocks(
            src_block_ids,
            dst_block_ids,
            is_swap_in,
            gpu_layer,
            cpu_layer,

            self.k_cache, self.v_cache,
            self.k_swap, self.v_swap
        )
        
    @torch.inference_mode()
    def initiate_swap_in(self, reqs: list[Request]):
        """
        Do all the set-up work for swapping in sequences.
        Returns src and dst block ids.
        """
        return self._initiate_swap(reqs, True)

    @torch.inference_mode()
    def initiate_swap_out(self, reqs: list[Request], use_itm: bool = False):
        """
        Do all the set-up work for swapping out sequences.
        Returns src and dst block ids.
        """
        return self._initiate_swap(reqs, False, use_itm)

    @torch.inference_mode()
    def swap_in_blocks(
        self,
        cur_layer: int,
        src_block_ids: list[int],
        dst_block_ids: list[int]
    ):
        """
        Swap in (move blocks from CPU to GPU) the specified sequences.
        """
        self._swap(src_block_ids, dst_block_ids, True, cur_layer, cur_layer)
    
    @torch.inference_mode()
    def swap_out_blocks(
        self,
        gpu_layer: int,
        cpu_layer: int,
        src_block_ids: list[int],
        dst_block_ids: list[int]
    ):
        """
        Swap out (move blocks from GPU to CPU) the specified sequences.
        """
        print("Swapping out {} blocks from GPU to CPU".format(len(src_block_ids)))
        self._swap(src_block_ids, dst_block_ids, False, gpu_layer, cpu_layer)
