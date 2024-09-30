import torch
import swiftllm_c
from swiftllm.engine_config import EngineConfig

from .kernels.block_mgmt import set_block_table_and_num_seq_alloc_blocks, unset_block_table_and_num_seq_alloc_blocks, gather_allocated_blocks_and_unset

class BlockManager:
    """
    BlockManager - Manage the block table and free blocks on CPU / GPU

    This manager records the mapping from (sequence ID, block index) to block 
    ID (which we call `block_table`), and provides methods to allocate and free
    blocks.

    All tables (the block table, the `num_seq_allocated_blocks`, and the free block
    list) are all maintained on the GPU, so that we can leverage custom Triton
    kernels for fast operations.
    """

    def __init__(self, device_name: str, num_blocks: int, max_seqs_in_block_table: int, max_blocks_per_seq: int, block_size: int):
        self.device_name = device_name
        self.num_free_blocks = num_blocks
        self.num_blocks = num_blocks
        self.block_size = block_size

        # seq_id |-> number of blocks allocated for this sequence
        self.num_seq_allocated_blocks = torch.zeros(
            (max_seqs_in_block_table,),
            dtype=torch.int32,
            device=device_name
        )
        # (seq_id, block_index) |-> block_id
        self.block_table = torch.empty(
            (max_seqs_in_block_table, max_blocks_per_seq),
            dtype=torch.int32,
            device=device_name
        )
        # block_id |-> whether this block is free or not
        self.is_block_free = torch.ones(
            (num_blocks,),
            dtype=torch.bool,
            device=device_name
        )
    
    def _allocate_blocks(self, num_blocks: int) -> torch.Tensor:
        """
        Allocate the requested number of blocks, update relevant status, and
        return the block IDs.
        """
        if num_blocks > self.num_free_blocks:
            raise RuntimeError(f"No enough free blocks available on {self.device_name} ({self.num_blocks} in total, {self.num_free_blocks} free, {num_blocks} requested)")
        selected_blocks = torch.nonzero(self.is_block_free)[:num_blocks].view(-1)
        self.num_free_blocks -= num_blocks
        self.is_block_free[selected_blocks] = False
        return selected_blocks
    
    def _free_blocks(self, block_ids: torch.Tensor):
        """
        Free the specified blocks, and update relevant status.
        """
        self.num_free_blocks += len(block_ids)
        self.is_block_free[block_ids] = True
    
    def allocate_blocks_for_seqs(self, seq_ids: torch.Tensor, target_lens: torch.Tensor) -> torch.Tensor:
        """
        Allocate blocks for sequences, making sure that seq #i has at least 
        ceil(target_lengths[i] / block_size) blocks allocated.

        Return new blocks allocated for the sequences. (useful for swapping)
        """
        target_num_blocks = (target_lens + (self.block_size-1)) // self.block_size
        assert (self.num_seq_allocated_blocks[seq_ids] <= target_num_blocks).all(), \
            f"""(On {self.device_name}) Logic error: Some sequences have more blocks already allocated than needed.
                seq_ids: {seq_ids}, target_lens: {target_lens}, target_num_blocks: {target_num_blocks},
                self.num_seq_allocated_blocks[seq_ids]: {self.num_seq_allocated_blocks[seq_ids]}"""
        block_needed = target_num_blocks - self.num_seq_allocated_blocks[seq_ids]
        new_blocks = self._allocate_blocks(torch.sum(block_needed))

        set_block_table_and_num_seq_alloc_blocks(self.num_seq_allocated_blocks, self.block_table, new_blocks, seq_ids, block_needed)

        return new_blocks
        
    def free_blocks_for_seqs(self, seq_ids: torch.Tensor):
        """
        Free blocks for sequences.
        """
        self.num_free_blocks += torch.sum(self.num_seq_allocated_blocks[seq_ids])
        unset_block_table_and_num_seq_alloc_blocks(self.num_seq_allocated_blocks, self.block_table, seq_ids, self.is_block_free)

    def gather_allocated_blocks_and_free(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """
        Gather the block IDs allocated for the specified sequences and mark them as free

        Useful fow swapping in/out
        """
        gathered_block_ids = gather_allocated_blocks_and_unset(self.num_seq_allocated_blocks, self.block_table, seq_ids, self.is_block_free)
        self.num_free_blocks += len(gathered_block_ids)
        return gathered_block_ids

    def get_num_allocated_blocks(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """
        Get the number of blocks allocated for the specified sequences
        Useful for swapping
        """
        return self.num_seq_allocated_blocks[seq_ids]

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
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_swap: torch.Tensor,
        v_swap: torch.Tensor,
        gpu_block_manager: BlockManager,
        cpu_block_manager: BlockManager
    ):
        assert gpu_block_manager.device_name == "cuda"
        assert cpu_block_manager.device_name == "cpu"
        self.engine_config = engine_config
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.k_swap = k_swap
        self.v_swap = v_swap
        self.gpu_block_manager = gpu_block_manager
        self.cpu_block_manager = cpu_block_manager
        self.num_layers = k_cache.size(0)

    def _initiate_swap(
        self,
        seq_ids_list: list[int],
        is_swap_in: bool
    ) -> tuple[list[int], list[int]]:
        """
        Do all the set-up work for swapping in/out sequences.
        Returns src and dst block ids.
        """
        if not seq_ids_list:
            return [], []
        src_block_manager = self.cpu_block_manager if is_swap_in else self.gpu_block_manager
        dst_block_manager = self.gpu_block_manager if is_swap_in else self.cpu_block_manager
        src_seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device=src_block_manager.device_name)
        dst_seq_ids = src_seq_ids.to(dst_block_manager.device_name)
        src_seq_lengths = src_block_manager.get_num_allocated_blocks(src_seq_ids) * self.engine_config.block_size
        dst_seq_lengths = src_seq_lengths.to(dst_block_manager.device_name)
        src_block_ids = src_block_manager.gather_allocated_blocks_and_free(src_seq_ids)
        dst_block_ids = dst_block_manager.allocate_blocks_for_seqs(dst_seq_ids, dst_seq_lengths)
        return src_block_ids.tolist(), dst_block_ids.tolist()
    
    def _swap(
        self,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        is_swap_in: bool,
        cur_layer: int
    ):
        swiftllm_c.swap_blocks(
            src_block_ids,
            dst_block_ids,
            is_swap_in,
            cur_layer,

            self.k_cache, self.v_cache,
            self.k_swap, self.v_swap
        )
        
    @torch.inference_mode()
    def initiate_swap_in(
        self,
        seq_ids_list: list[int]
    ):
        """
        Do all the set-up work for swapping in sequences.
        Returns src and dst block ids.
        """
        return self._initiate_swap(seq_ids_list, True)

    @torch.inference_mode()
    def initiate_swap_out(
        self,
        seq_ids_list: list[int]
    ):
        """
        Do all the set-up work for swapping out sequences.
        Returns src and dst block ids.
        """
        return self._initiate_swap(seq_ids_list, False)


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
        self._swap(src_block_ids, dst_block_ids, True, cur_layer)
    
    @torch.inference_mode()
    def swap_out_blocks(
        self,
        cur_layer: int,
        src_block_ids: list[int],
        dst_block_ids: list[int]
    ):
        """
        Swap out (move blocks from GPU to CPU) the specified sequences.
        """
        self._swap(src_block_ids, dst_block_ids, False, cur_layer)
    
    