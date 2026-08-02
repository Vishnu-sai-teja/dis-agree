# dis-agree
This project is being created to implement a disaggregated serving system.

flowchart LR

    subgraph EDGE["Edge & API Plane"]
        CLIENT["Client"]
        API["FastAPI Server<br/>/generate"]
        STREAM["Streaming Response"]
        RESPONSE["JSON Response"]
    end

    subgraph REQUESTS["Request Management Plane"]
        PARAMS["SamplingParams"]
        REQUEST["Request"]
        BATCH["BatchedRequests"]
        EVENTS["Lifetime Events"]
    end

    subgraph ENGINE["Engine Orchestration Plane"]
        LLM["AsyncLLM / OfflineLLM"]
        CORE["LLMEngine"]
        BRIDGE["Bridge Queue"]
    end

    subgraph CONTEXT["Context / Prefill Plane"]
        CONTEXT_ENGINE["ContextStageLLMEngine"]
        CONTEXT_SCHED["Context Scheduler"]
        CONTEXT_BATCH["Context Batch"]
        CONTEXT_WORKERS["Context Ray Workers"]
    end

    subgraph DECODING["Decoding Plane"]
        DECODING_ENGINE["DecodingStageLLMEngine"]
        DECODING_SCHED["Decoding Scheduler"]
        DECODING_BATCH["Decoding Batch"]
        DECODING_WORKERS["Decoding Ray Workers"]
    end

    subgraph MEMORY["KV-Cache and Memory Plane"]
        CONTEXT_BM["Context BlockManager"]
        DECODING_BM["Decoding BlockManager"]
        GPU_CACHE["GPU KV Cache"]
        CPU_CACHE["CPU Swap Cache"]
        MIGRATION["KV-Cache Migration"]
        SWAPPING["KV-Cache Swap In / Out"]
    end

    subgraph MODEL["Model Execution Plane"]
        TOKENIZER["HuggingFace Tokenizer"]
        MODEL_OP["Model Operator"]
        FORWARD["GPU Forward Pass"]
        TOKENS["Generated Token IDs"]
    end

    subgraph INFRA["Distributed Infrastructure Plane"]
        RAY["Ray Runtime"]
        PLACEMENT["Placement Groups"]
        GPU["GPU Devices"]
        NCCL["NCCL Communication"]
    end

    %% Request entry and response
    CLIENT --> API
    API --> PARAMS
    API --> LLM
    API --> STREAM
    API --> RESPONSE

    %% Request creation and preparation
    LLM --> CORE
    CORE --> REQUEST
    REQUEST --> BATCH
    CORE --> EVENTS
    CORE --> TOKENIZER
    TOKENIZER --> REQUEST

    %% Engine orchestration
    CORE --> CONTEXT_ENGINE
    CORE --> DECODING_ENGINE
    CONTEXT_ENGINE --> BRIDGE
    BRIDGE --> DECODING_ENGINE

    %% Context / prefill execution
    CONTEXT_ENGINE --> CONTEXT_SCHED
    CONTEXT_SCHED --> CONTEXT_BATCH
    CONTEXT_BATCH --> CONTEXT_WORKERS
    CONTEXT_WORKERS --> CONTEXT_BM

    %% Decoding execution
    DECODING_ENGINE --> DECODING_SCHED
    DECODING_SCHED --> DECODING_BATCH
    DECODING_BATCH --> DECODING_WORKERS
    DECODING_WORKERS --> DECODING_BM

    %% KV-cache management
    CONTEXT_BM --> GPU_CACHE
    DECODING_BM --> GPU_CACHE
    DECODING_BM --> CPU_CACHE

    CONTEXT_BM --> MIGRATION
    MIGRATION --> DECODING_BM

    DECODING_SCHED --> SWAPPING
    SWAPPING --> CPU_CACHE
    SWAPPING --> GPU_CACHE

    %% Model execution
    CONTEXT_WORKERS --> MODEL_OP
    DECODING_WORKERS --> MODEL_OP
    MODEL_OP --> FORWARD
    FORWARD --> TOKENS
    TOKENS --> TOKENIZER

    %% Output path
    TOKENIZER --> STREAM
    TOKENIZER --> RESPONSE

    %% Distributed infrastructure
    CORE --> RAY
    RAY --> PLACEMENT
    PLACEMENT --> GPU

    CONTEXT_WORKERS --> NCCL
    DECODING_WORKERS --> NCCL
    NCCL --> GPU

    %% Red: core model and execution components
    classDef llm fill:#fde2e2,stroke:#b3261e,stroke-width:2px,color:#5c1410

    %% Orange: scheduling and model-adjacent components
    classDef maybemodel fill:#fff3dc,stroke:#a46a12,stroke-width:2px,color:#573700

    %% Green: deterministic infrastructure and control components
    classDef nomodel fill:#e8f3ec,stroke:#24754b,stroke-width:1px,color:#123b25

    %% Main execution entities
    class LLM,CORE,CONTEXT_ENGINE,DECODING_ENGINE,MODEL_OP,FORWARD llm

    %% Scheduling and cache-transfer entities
    class TOKENIZER,CONTEXT_SCHED,DECODING_SCHED,MIGRATION,SWAPPING maybemodel

    %% Supporting entities
    class CLIENT,API,STREAM,RESPONSE,PARAMS,REQUEST,BATCH,EVENTS,BRIDGE,CONTEXT_BATCH,CONTEXT_WORKERS,DECODING_BATCH,DECODING_WORKERS,CONTEXT_BM,DECODING_BM,GPU_CACHE,CPU_CACHE,TOKENS,RAY,PLACEMENT,GPU,NCCL nomodel

    %% Plane backgrounds and borders
    style EDGE fill:#eef5ff,stroke:#3973b9,stroke-width:2px
    style REQUESTS fill:#fff9e6,stroke:#c28a16,stroke-width:2px
    style ENGINE fill:#f4edff,stroke:#7651a8,stroke-width:2px
    style CONTEXT fill:#eef9f0,stroke:#3f8f55,stroke-width:2px
    style DECODING fill:#fff0f0,stroke:#b95757,stroke-width:2px
    style MEMORY fill:#eaf8fb,stroke:#27879b,stroke-width:2px
    style MODEL fill:#fff0f5,stroke:#ad4c70,stroke-width:2px
    style INFRA fill:#f1f3f5,stroke:#68717a,stroke-width:2px
