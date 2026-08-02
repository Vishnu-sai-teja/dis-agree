import time
from typing import List, Optional, Union
from server.utils import Counter
from server.config import ParallelConfig


class SamplingParams:

    _SAMPLING_EPS = 1e-5 # Negligence gap for the values - boundary decisions

    def __init__(self,
        n: int = 1, # Number of responses to return for this prompt
        best_of: Optional[int] = None, # Number of repsonses to return on teh basis of the beam search
        presence_penality: float = 0.0, # Penalise tokens on the basis of their presence in the generated - 
        frequency_penality: float = 0.0, # Presence penality + based on frequency 
        top_p: float = 1,
        top_k: int = -1, # -1 means - let the top_p and temperature do the talking
        use_beam_search: bool = False,
        stop: Union[List[str], str, None] = None, # List of strings to stop the generation of the output
        ignore_eos = False, # Ignore the EOS
        max_tokens: int = 64, # Max number of tokens for the outout sequence
        logprobs: int = None, # Return logprobs number of likely tokens 
        temperature: int = 0, # Creativity of the response
    ):
        self.n = n
        self.best_of = best_of
        self.presence_penality = presence_penality
        self.frequency_penality = frequency_penality
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.use_beam_search = use_beam_search
        self.max_tokens = max_tokens
        self.logprobs = logprobs
        self.ignore_eos = ignore_eos

        if stop is not None:
            self.stop = []
        elif isinstance(stop, str):
            self.stop.append(stop)
        else:
            self.stop = list(stop)
        
        self.verify_args()
        if self.use_beam_search:
            self.verify_beam_search() 
        if self.temperature < self._SAMPLING_EPS:
            self.verify_greedy_sampling()
    
    def verify_args(self):
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}.")
        if self.best_of < self.n:
            raise ValueError(
                f"best_of must be greater than or equal to n, "
                f"got n={self.n} and best_of={self.best_of}."
            )
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                "presence_penalty must be in [-2, 2], got " f"{self.presence_penalty}."
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                "frequency_penalty must be in [-2, 2], got "
                f"{self.frequency_penalty}."
            )
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(
                f"top_k must be -1 (disable), or at least 1, " f"got {self.top_k}."
            )
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {self.max_tokens}.")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(f"logprobs must be non-negative, got {self.logprobs}.")

    def verify_beam_search(self):
        # The self.best_of > 1
        # The temperature shouild be 0 - to run the bea search > eps
        # The top_p should be 1 - to run the beam search 
        # Top_k = -1 # disable top k in case of running the beamsearch 
        if self.best_of is None or self.best_of <= 1:
            raise ValueError(
                f"best_of must be greater than 1 for beam search, got {self.best_of}."
            )
        if self.temperature > self._SAMPLING_EPS: # Highest probable tokens - no randomness
            raise ValueError(
                f"temperature must be 0 for beam search, got {self.temperature}."
            )
        if self.top_p < 1.0 - self._SAMPLING_EPS:
            raise ValueError(
                f"top_p must be 1 for beam search, got {self.top_p}."
            )
        if self.top_k != -1:
            raise ValueError(
                f"top_k must be -1 (disabled) for beam search, got {self.top_k}."
            )

    def verify_greedy_sampling(self):
        # Temperature = 0 - only one possible answer - purely greedy sampling - no randomness
        if self.n > 1:
            raise ValueError(
                f"n must be 1 for greedy sampling, got {self.n}."
            )
        if self.top_p < 1.0 - self._SAMPLING_EPS:
            raise ValueError(
                f"top_p must be 1 for greedy sampling, got {self.top_p}."
            )
        if self.top_k != -1:
            raise ValueError(
                f"top_k must be -1 (disabled) for greedy sampling, got {self.top_k}."
            )

class Request:
    """
        Request : User prompt + output tokens + metadata
    """
    def __init__(self
        arrival_time: float, # Time of arrival of the request
        request_id: int,
        input_prompt: str, 
        prompt_token_ids: List[int], # Token ID's of the input ptompt
        sampling_params: SamplingParams = SamplingParams(),
        priority: int = 0 # Least priority be default
    ):

        # Constant states
        self.arrival_time = arrival_time
        self.request_id = request_id
        self.input_prompt = input_prompt
        self.prompt_tokens = prompt_tokens
        self.sampling_params = sampling_params
    
        # Dynamic States
        self.generated_tokens = []
        self.generated_token_ids = []
        self.is_running = False # generation running
        self.is_finished = False # generation fisnished
        self.priority = priority

        self.process_time = 0.0
        self.last_step_time = 0.0
    
    def reset_process_time(self):
        self.process_time = 0.0
    
    def add_process_time(self, running_time: float):
        self.process_time += running_time

    # Check if the request response is done
    def check_finish_condition(self):
        if len(self.generated_token_ids) >= self.sampling_params.max_tokens:
            self.is_finished = True
        
        if not self.sampling_params.ignore_eos:
            if len(self.generated_token_ids) and (
                self.generated_tokens[-1] in self.sampling_params.stop
            ):
                self.is_finished = True

    def add_generated_token(self, token_id: int, token: str):
        self.generated_token_ids.append(token_id)
        self.generated_tokens.append(token)
        self.check_finish_condition() # Check if the request is finished

    def is_prefill(self):
        return len(self.generated_tokens) == 0
    
    def get_input_prompt_length(self):
        return len(self.prompt_tokens)


    def get_latest_token_index(self):
        # For teh decoding - as the decoding needs teh latest token to compute the new QKV layer
        return 0 if self.is_prefill() else self.get_input_prompt_length() + len(self.generated_tokens) - 1
    
    def get_kv_cache_slots(self):
        # A slot is the request memory to store the one tokens tensors
        return len(self.prompt_tokens) + len(self.generated_tokens)


def create_request(
    prompt: str,
    prompt_token_ids,
    sampling_params: SamplingParams = SamplingParams(),
    request_counter: Counter = Counter(),
    tokenizer = None,
    priority: int = 0
    arrival_time: float = 0.0,
    request_id: int = None
):
    if request_id is None:
        request_id = next(request_counter)
    if prompt_token_ids is None and tokenizer is not None:
        prompt_token_ids = tokenizer.encode(prompt)
    if prompt is None and prompt_token_ids is not None and tokenizer is not None:
        prompt = tokenizer.decode(prompt_token_ids)
    return Request(
        arrival_time=arrival_time,
        request_id=request_id,
        input_prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        priority=priority
    )


class BatchRequests:
    def __init__(self, 
        requests: Optional[List[Request]] = None
    ):
        if requests is None:
            self.requests = []
        else:
            self.requests = requests
        self.start_time = None
    
    def __len__(self):
        return len(self.requests)

    def __str__(self):
        return f"BatchedRequests: {self.requests}"
    
    def __repr__(self):
        return f"BatchedRequests: {self.requests}"

    def get_request_ids(self):
        return [request.request_id for request in self.requests]

    def add_request(self, request: Request):
        """Check for duplicates and add a request""" 
        try:
            if request.request_id in self.get_request_ids():
                raise ValueError(f'Request with {request.request_id} already exists in the Batch')
            else:
                self.requests.append(request)
        except Exception as e:
            print(f"An Error Occuured in adding request to the barch : {e}")

    def pop_finished_requests(self):
        finished_requests = []
        unfinished_requests = []
        for request in self.requests:
            if request.is_finished:
                finished_requests.append(request)
            else:
                unfinished_requets.append(request)
        
        self.requests = unfinished_requests
        return finished_requests
    
    def batch_start_one_iteration(self, start_time = None):
        self.start_time = start_time if start_time else time.time()
        self.is_running = True
    
    def batch_update_one_iteration(self, 
        generated_tokens: List[str],
        generated_token_ids: List[int],
        end_time: float
    ):
        for request, generated_token, generated_token_ids in zip(self.requests, generated_tokens, generated_token_ids):
            request.last_step_time = end_time
            request.generated_tokens.append(generated_token)
            request.genreated_token_ids.append(generated_token_id)
        
        self.start_time = None
        self.is_running = True

    # Get states
    def get_num_input_tokens(self):
        return sum([request.get_input_prompt_length() for request in self.requests])

    def get_kv_cache_slots(self):
        return sum([request.get_kv_cache_slots() for request in self.requests])
    
    def get_input_tokens_batched(self):
        return [
            request.generated_token_ids for request in self.requests
        ]
    
    def get_first_token_indexes(self):
        return [
            self.get_latest_token_index() for request in self.requests
        ]
    
    def is_context_stage(self):
        return [
            request.is_prefill() for request in self.requests
        ]

    