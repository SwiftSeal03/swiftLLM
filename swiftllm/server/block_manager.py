"""
Block manager classes on the control plane.

They are used to manage the allocated and free blocks on both CPU and GPU, but actual 
model computations doesn't involve these classes.
"""

import torch
from swiftllm.engine_config import EngineConfig
from swiftllm.structs import Request, SubBatch


class DeviceBlockManager:
    """
    DeviceBlockManager - Manage the allocated & free blocks on one device (CPU / GPU)

    We may split KV cache along layer dimension, so there may be multiple free blocks tables.

    However, the sequences share same namespace of sequence IDs, so we only need one set of
    block table and `seq_num_blks`.
    """

    @torch.inference_mode()
    def __init__(
        self, 
        device_name: str, 
        engine_config: EngineConfig
    ):
        self.device_name = device_name
        self.num_blocks = engine_config.num_gpu_blocks if device_name == 'cuda' else engine_config.num_cpu_blocks
        self.block_size = engine_config.block_size
        self.block_table_width = engine_config.max_blocks_per_seq
        nsplits = 1 + engine_config.extra_layer_for_cprf

        # seq_id |-> number of blocks allocated for this sequence
        self.seq_num_blks = torch.zeros(
            (engine_config.max_seqs_in_block_table,),
            dtype=torch.int32,
            device='cpu'
        )
        # (seq_id, block_index) |-> block_id
        self.block_table = torch.empty(
            (engine_config.max_seqs_in_block_table, engine_config.max_blocks_per_seq),
            dtype=torch.int32,
            device='cpu'
        )
        # block_id |-> whether this block is free or not
        self.num_free_blocks = [self.num_blocks] * nsplits
        self.is_block_free = [torch.ones(
            (self.num_blocks,),
            dtype=torch.bool,
            device='cpu'
        ) for _ in range(nsplits)]

    
    @torch.inference_mode()
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
    
    
    @torch.inference_mode()
    def alloc(self, reqs: list[Request], split_point: int=0) -> tuple[list[int], list[int]]:
        """
        Allocate blocks for sequences, making sure that every request have enough blocks allocated for all its tokens.

        Those after split_point will be allocated in the first split, and the rest will be allocated in the second split.

        Return new mapping from block virtual IDs to block physical IDs.
        """
        if not reqs:
            return [], []

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
        new_blk_pids = torch.cat([new_blk_ids1, new_blk_ids0])

        seq_num_blks_list = seq_num_blks.tolist()
        new_num_blks_list = new_num_blks.tolist()
        new_blk_vids = [seq_ids[i] * self.block_table_width + j + seq_num_blks_list[i] for i, n in enumerate(new_num_blks_list) for j in range(n)]
        self.block_table.view(-1)[new_blk_vids] = new_blk_pids
        self.seq_num_blks[seq_ids] = tgt_num_blks
        return new_blk_vids, new_blk_pids.tolist()
    

    @torch.inference_mode()
    def free(self, reqs: list[Request], split_id: int=0) -> list[int]:
        """
        Free the blocks allocated for the specified sequences.

        Return the block physical IDs that are freed.
        """
        if not reqs:
            return []
        
        seq_ids = Request.get_ids(reqs)
        seq_num_blks_list = self.seq_num_blks[seq_ids].tolist()
        blk_vids = [seq_ids[i] * self.block_table_width + j for i, n in enumerate(seq_num_blks_list) for j in range(n)]
        blk_pids = self.block_table.view(-1)[blk_vids] # possibly on GPU
        self.num_free_blocks[split_id] += len(blk_pids)
        self.is_block_free[split_id][blk_pids] = True
        self.seq_num_blks[seq_ids] = 0
        return blk_pids.tolist()
    


class BlockManager:
    """
    BlockManager - Manage the allocated & free blocks on both CPU and GPU
    """

    def __init__(
        self, 
        engine_config: EngineConfig
    ):
        self.extra_layer_for_cprf = engine_config.extra_layer_for_cprf
        self.gpu_block_manager = DeviceBlockManager("cuda", engine_config)
        self.cpu_block_manager = DeviceBlockManager("cpu", engine_config)
    

    def alloc_blocks_for_batch(self, batch: SubBatch) -> tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]:
        """
        Allocate blocks for a batch of sequences.

        Return new block VIDs to block PIDs mappings on both CPU and GPU. 
        """
        return (
            self.gpu_block_manager.alloc(batch.all_reqs[:batch.num_prgds], split_point=batch.num_cprfs * self.extra_layer_for_cprf),
            self.cpu_block_manager.alloc(batch.all_reqs[batch.num_prgds:])
        )
    

    def free_blocks_of_requests(self, reqs: list[Request]) -> tuple[list[int], list[int]]:
        """
        Free the blocks allocated for the specified requests.
        """
        return self.gpu_block_manager.free(reqs), self.cpu_block_manager.free(reqs)


    def initiate_swap(
        self,
        reqs: list[Request],
        is_swap_out: bool,
        use_itm: bool = False # Only true when swapping out from intermediate cache to CPU
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Do all the set-up work for swapping in/out sequences.
        Returns a triple of src block PIDs, dst block VIDs and dst block PIDs.
        """
        assert is_swap_out or not use_itm, "Cannot swap in to intermediate space"

        if not reqs:
            return [], [], []
        
        src_block_manager = self.gpu_block_manager if is_swap_out else self.cpu_block_manager
        dst_block_manager = self.cpu_block_manager if is_swap_out else self.gpu_block_manager
        src_blk_pids = src_block_manager.free(reqs, int(use_itm))
        dst_blk_vids, dst_blk_pids = dst_block_manager.alloc(reqs)
        return src_blk_pids, dst_blk_vids, dst_blk_pids
    
