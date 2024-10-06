import asyncio
import dataclasses
import torch
import itertools
from swiftllm.perfpredictor import PerfPredictor, ZeroPerfPredictor

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
    cur_output_len: int     # Current output length

    output_q: asyncio.Queue[StepOutput] # Queue initialized when the raw request enters the
                                        # engine, and to be set upon a new token being generated
                                        # Mainly for streaming the output back to the user
    finished_event: asyncio.Event       # Event to be set when the request is finished
                                        # Mainly for the non-streaming case

    request_id: int     # Request ID, within range [0, max_seqs_in_block_table).
                        # Generated before being prefilled, and used as the index
                        # into the block table
    output_token_ids: list[int]     # Output token ids'

    @property
    def seq_len(self) -> int:
        return self.prompt_len + self.cur_output_len

    def __init__(self, raw_request: RawRequest):
        # A request is __init__-ed when entering `untokenized_raw_requests`, and
        # its `prompt_token_ids` and `prompt_len` will be set upon tokenization
        self.prompt_token_ids = []
        self.prompt_len = 0
        self.output_len = raw_request.output_len
        self.cur_output_len = 0
        self.output_q = asyncio.Queue()
        self.finished_event = asyncio.Event()
        self.request_id = -1
        self.output_token_ids = []
    
    def is_finished(self) -> bool:
        return self.cur_output_len == self.output_len

    def is_prefill_stage(self) -> bool:
        return not self.output_token_ids

def create_request(prompt_token_ids: list[int], req_id: int, output_len: int=1e9) -> Request:
    ret = Request(RawRequest("", output_len))
    ret.prompt_token_ids = prompt_token_ids
    ret.prompt_len = len(prompt_token_ids)
    ret.request_id = req_id
    return ret

class BatchMetaData:
    def __init__(self, predictor: PerfPredictor):
        self.x = 0
        self.s = 0
        self.n_g = 0
        self.x_c = 0
        self.n_c = 0

        self.predictor = predictor
        self.pref_T = 0
        self.gdec_T = 0
        self.lnch_T = predictor.get_lnch_T()

    def add_pref(self, prompt_len):
        self.x += 1
        self.s += prompt_len
        self.pref_T += self.predictor.get_pref_T(prompt_len)
    
    def pop_pref(self, prompt_len):
        self.x -= 1
        self.s -= prompt_len
        self.pref_T -= self.predictor.get_pref_T(prompt_len)

    def add_gdec(self, seq_len):
        self.x += 1
        self.s += 1
        self.n_g += seq_len
        self.gdec_T = self.predictor.get_gdec_T(self.n_g)

    def add_cdec(self, seq_len):
        self.x += 1
        self.s += 1
        self.x_c += 1
        self.n_c += seq_len

    def pop_cdec(self, seq_len):
        self.x -= 1
        self.s -= 1
        self.x_c -= 1
        self.n_c -= seq_len

    @property
    def linr_T(self) -> float:
        return self.predictor.get_linr_T(self.s)
    
    @property
    def cdec_T(self) -> float:
        return self.predictor.get_cdec_T(self.x_c, self.n_c)
    
    @property
    def gpu_time(self) -> float:
        return self.linr_T + self.pref_T + self.gdec_T
    
    @property
    def cpu_time(self) -> float:
        return self.cdec_T + self.lnch_T

class SubBatch:
    def __init__(self, predictor: PerfPredictor=ZeroPerfPredictor()):
        self.gprf_reqs = []
        self.cprf_reqs = []
        self.gdec_reqs = []
        self.cdec_reqs = []
        self.metadata = BatchMetaData(predictor)  
    
    def __len__(self):
        return self.metadata.x

    def add_pref(self, req: Request, is_gpu: bool):
        if is_gpu:
            self.gprf_reqs.append(req)
        else:
            self.cprf_reqs.append(req)
        self.metadata.add_pref(req.prompt_len)

    def pop_pref(self) -> Request:
        is_gpu = not self.cprf_reqs
        req = self.gprf_reqs.pop() if is_gpu else self.cprf_reqs.pop()
        self.metadata.pop_pref(req.prompt_len)
        return req, is_gpu
        
    def add_gdec(self, req: Request):
        self.gdec_reqs.append(req)
        self.metadata.add_gdec(req.seq_len)

    def add_cdec(self, req: Request):
        self.cdec_reqs.append(req)
        self.metadata.add_cdec(req.seq_len)

    def pop_cdec(self):
        req = self.cdec_reqs.pop()
        self.metadata.pop_cdec(req.seq_len)

    def get_num_prefs(self) -> int:
        return len(self.gprf_reqs) + len(self.cprf_reqs)

    def set_model_forward_args(self):
        self.pref_reqs = self.gprf_reqs + self.cprf_reqs
        self.deco_reqs = self.gdec_reqs + self.cdec_reqs
        self.all_reqs = self.pref_reqs + self.deco_reqs

        self.num_prefs = len(self.pref_reqs)
        self.num_cprfs = len(self.cprf_reqs)
        self.num_gdecs = len(self.gdec_reqs)
        self.num_cdecs = len(self.cdec_reqs)

        self.pref_seq_ids_list = [req.request_id for req in self.pref_reqs]
        self.gdec_seq_ids_list = [req.request_id for req in self.gdec_reqs]
        self.cdec_seq_ids_list = [req.request_id for req in self.cdec_reqs]

        self.pref_seq_lens_list = [req.seq_len for req in self.pref_reqs]
        self.gdec_seq_lens_list = [req.seq_len for req in self.gdec_reqs]
        self.cdec_seq_lens_list = [req.seq_len for req in self.cdec_reqs]

        self.prgd_seq_ids = torch.tensor(self.pref_seq_ids_list + self.gdec_seq_ids_list, dtype=torch.int32, device='cuda')
        self.cdec_seq_ids = torch.tensor(self.cdec_seq_ids_list, dtype=torch.int32, device='cpu')
        
        self.prgd_seq_lens = torch.tensor(self.pref_seq_lens_list + self.gdec_seq_lens_list, dtype=torch.int32, device='cuda')
        self.cdec_seq_lens = torch.tensor(self.cdec_seq_lens_list, dtype=torch.int32, device='cpu')

        self.sum_pref_toks = sum(self.pref_seq_lens_list)
        self.max_pref_toks = max(self.pref_seq_lens_list, default=0)

        pref_st_locs_we = [0] + list(itertools.accumulate(self.pref_seq_lens_list))
        self.pref_st_locs_we = torch.tensor(pref_st_locs_we, dtype=torch.int32, device='cuda')

        self.input_token_ids = sum([req.prompt_token_ids for req in self.pref_reqs], []) + \
            [req.output_token_ids[-1] for req in self.deco_reqs]

        # To be set by the model

    def update_output(self, output_toks: torch.Tensor):
        output_toks = output_toks.tolist()
        assert len(output_toks) == len(self.all_reqs), \
            f"len(output_toks) {len(output_toks)} != len(self.all_reqs) {len(self.all_reqs)}"
        for i, req in enumerate(self.all_reqs):
            req.cur_output_len += 1
            req.output_token_ids.append(output_toks[i])
            req.output_q.put_nowait(StepOutput(output_toks[i], req))


    def print_profile(self):
        print(f"gprf lens: {[req.prompt_len for req in self.gprf_reqs]}, cprf lens: {[req.prompt_len for req in self.cprf_reqs]}, \
gdec lens: {[req.seq_len for req in self.gdec_reqs]}, cdec lens: {[req.seq_len for req in self.cdec_reqs]}")
    