import dataclasses
import argparse

@dataclasses.dataclass
class EngineConfig:
    """
    Configuration for the SwiftLLM engine.
    """
    
    # Model loading parameters
    model_path: str
    use_dummy: bool

    # PagedAttention-related parameters
    block_size: int
    gpu_mem_utilization: float
    num_cpu_blocks: int
    max_seqs_in_block_table: int
    max_blocks_per_seq: int

    # Scheduling-related parameters
    max_batch_size: int
    max_prefill_tokens: int
    max_tokens_in_batch: int

    # External library path
    library_path: str = None
    profile_result_path: str = None

    # Switches
    ignore_kvcache: bool = False      # Should be turned off when profiling blocks
    monitor_performance: bool = False # Can be altered while running
    always_use_gpu: bool = False      # Can be altered while running

    @property
    def max_seq_len(self) -> int:
        return self.block_size * self.max_blocks_per_seq
    
    @property
    def max_tokens_on_cpu(self) -> int:
        return self.block_size * self.num_cpu_blocks

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        """
        Add CLI arguments for the engine configuration
        """
        parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="Path to the model directory (currently SwiftLLM does not support downloading from HuggingFace, so please download in advance)",
        )
        parser.add_argument(
            "--use-dummy",
            action="store_true",
            help="Use dummy weights (mainly for profiling)",
        )

        parser.add_argument(
            "--block-size",
            type=int,
            default=16,
            help="Block size for PagedAttention",
        )
        parser.add_argument(
            "--gpu-mem-utilization",
            type=float,
            default=0.99,
            help="Fraction of GPU memory to be used",
        )
        parser.add_argument(
            "--num-cpu-blocks",
            type=int,
            default=10000,
            help="Number of CPU blocks",
        )
        parser.add_argument(
            "--max-seqs-in-block-table",
            type=int,
            default=768,
            help="Maximum number of sequences in the block table",
        )
        parser.add_argument(
            "--max-blocks-per-seq",
            type=int,
            default=512,
            help="Maximum number of blocks per sequence",
        )

        parser.add_argument(
            "--max-batch-size",
            type=int,
            default=512,
            help="Maximum batch size",
        )
        parser.add_argument(
            "--max-tokens-in-batch",
            type=int,
            default=3072,
            help="Maximum number of tokens in a batch",
        )

        parser.add_argument(
            "--library-path",
            type=str,
            help="Path to the external library",
        )

        parser.add_argument(
            "--monitor-performance",
            action="store_true",
            help="Monitor performance",
        )
        