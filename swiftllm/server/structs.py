import asyncio
import dataclasses

@dataclasses.dataclass
class StepOutput:
    """
    The output of one decoding step
    """
    token_id: int
    request: "Request"


class RawRequest:
    """
    A request issued by user
    """
    prompt: str
    output_len: int

    def __init__(self, prompt: str, output_len: int):
        self.prompt = prompt
        self.output_len = output_len


class Request:
    """
    A (queuing, processing, or finished) request in the system
    """   

    prompt_token_ids: list[int]     # Prompt token ids, generated by the tokenizer upon request arrival
    prompt_len: int     # len(prompt_token_ids)
    output_len: int     # Output length

    output_q: asyncio.Queue[StepOutput] # Queue initialized when the raw request enters the
                                        # engine, and to be set upon a new token being generated

    request_id: int     # Request ID, within range [0, max_seqs_in_block_table).
                        # Generated before being prefilled, and used as the index
                        # into the block table
    output_token_ids: list[int]     # Output token ids

    def __init__(self, raw_request: RawRequest, prompt_token_ids: list[int], output_q: asyncio.Queue[StepOutput]):
        self.prompt_token_ids = prompt_token_ids
        self.prompt_len = len(prompt_token_ids)
        self.output_len = raw_request.output_len
        self.output_q = output_q
        self.request_id = -1
        self.output_token_ids = []
    
    def is_finished(self) -> bool:
        return len(self.output_token_ids) == self.output_len
    
    def get_cur_output_len(self) -> int:
        return len(self.output_token_ids)

    def is_prefill_stage(self) -> bool:
        return not self.output_token_ids
    