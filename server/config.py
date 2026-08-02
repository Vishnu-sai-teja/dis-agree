# Define configs for each phase of 
# -- Model config, 

from transformers import AutoConfig

from utils import GB

class ModelConfig:
    def __init__(self,
        model,
        tokenizer,
        tokenizer_mode = "auto",
        seed = 1,
        trust_remote_code = False, 
        dtype = "fp16",
        use_dummpy_weights = False # Architecture is created - but real weights are not loaded
    ):
        self.model = model
        self.tokenizer = tokenizer if tokenizer else model
        self.tokenizer_mode = tokenizer_mode
        self.seed = seed
        self.dtype = dtype
        
        self.trust_remote_code = trust_remote_code
        self.use_dummpy_weights = use_dummpy_weights

        self.verify_args()

    def verify_args(self):
        assert self.dtype in ["fp16", "fp32"], 'Model dtype must be either f16 or f32'
    
    def get_hf_config(self):
        """Get the model config - technichal from HF"""
        try:
            config = AutoConfig.from_pretrained(
                self.model,
                trust_remote_code=self.trust_remote_code
            ) 
            return config
        except:
            raise ValueError(
                "Failed to load the model config from Hugging Face"
            )

class ParallelConfig:
    def __init__(self,
        tensor_parallel_size = 1,
        tensor_parallel_rank = 0,
        pipeline_parallel_size = 1,
        pipeline_parallel_rank = 0
    ):
        self.tensor_parallel_size = tensor_parallel_size
        self.tensor_parallel_rank = tensor_parallel_rank
        self.pipeline_parallel_size = pipeline_parallel_size
        self.pipeline_parallel_rank = pipeline_parallel_rank

        # Total GPU's 
        self.world_size = self.tensor_parallel_size * self.pipeline_parallel_size
        # Parallel config makes sense it 
        self.use_parallel = self.world_size > 1

class DisaggParallelConfig:
    def __init__(self, prefill: ParallelConfig, decode: ParallelConfig):
        self.prefill = context
        self.decode = decode 

    def get_num_workers(self):
        return self.prefill.world_size + self.prefill.world_size


class PrefillSchedulingConfig:
    def __init__(self, 
        policy: str,
        max_batch_size: int, # Max requersts in a batch
        max_tokens_per_batch: int # Max tokens in one batch
    ):
        assert policy in ["fcfs"], "Policies other than FCFS are not supported"
        self.policy = policy
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch


class DecodeSchedulingConfig:
    def __init__(self,
        policy: str, 
        model_name: str = None,
        max_batch_size: int,
        max_tokens_per_batch: int,
        waiting_block_prop_threadhold: float = 0.05 # Number of KV blocks that can be reserved by waiting requests
    ):

        # <!--
        # These are scheduling policies: rules for deciding which request gets GPU time next.

        # - FCFS: First Come, First Served. Older requests run first.
        # - SRPT: Shortest Remaining Processing Time. Shorter remaining jobs run first.
        # - MLFQ: Multi-Level Feedback Queue. Requests move between priority queues.
        # - SJ-MLFQ: Size-aware MLFQ that favors shorter jobs.

        # In this repository, only `fcfs` is currently implemented.
        # The other policies currently raise `NotImplementedError`.
        # -->
        assert policy in ["fcfs"], f'Decoding sheduler {policy} not supported'
        
        self.policy = policy
        self.model_name = model_name
        
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.waiting_block_prop_threadhold = waiting_block_prop_threadhold
        

class CacheConfig:
    def __init__(self,
            block_size: int,
            max_num_blocks_per_request: int, # Number of blocks that can be used per request
            gpu_mermory_utilisation: float = 0.9,  
            cpu_swap_space: int # Amount of CPU swap space that can be used
        ):
        self.block_size = block_size
        self.max_num_blocks_per_request = max_num_blocks_per_request
        self.gpu_mermory_utilisation = gpu_mermory_utilisation
        self.cpu_swap_space = cpu_swap_space * GB

    
    
        


