```mermaid
flowchart TD
    classDef qwen fill:#eef1ff,stroke:#5157d8,stroke-width:2px,color:#1f1f3d;
    classDef store fill:#fef6e4,stroke:#d98c3b,stroke-width:1.5px,color:#3d2206;

    RIMGS[Raw mission images<br/>挡风板安装检查 / BBU接地线检查]
        --> STAGEA[Stage A Inference<br/>qwen-3vl · scripts/stage_a_infer.sh]
    class STAGEA qwen

    STAGEA --> AJSON[[Stage A JSONL outputs<br/>挡风板安装检查_stage_a.jsonl<br/>BBU接地线检查_stage_a.jsonl]]
    class AJSON store

    AJSON --> INGEST[Stage B Ingest<br/>normalize groups · read mission guidance]

    INGEST -->|per mission| GUIDANCE[Mission Guidance File<br/>focus + guidance + experiences]
    class GUIDANCE store

    INGEST --> LOOP{For each group ticket}

    LOOP -->|reuse qwen-3vl| SAMPLER[Sampler
K rollouts · decode grid]
    class SAMPLER qwen

    SAMPLER --> TLOG[[Trajectories JSONL
response text · decode params]]
    class TLOG store

    TLOG --> SIGNALS[Deterministic Signals
label_match · confidence · trust]
    SIGNALS --> TLOG

    TLOG --> CRITIC[CriticEngine (LLM)
strict JSON per-candidate]
    class CRITIC qwen
    CRITIC --> TLOG

    subgraph BATCH[After batch completes]
        TLOG --> REFLECT[Reflection Engine (LLM)
propose operations (JSON only)]
        GUIDANCE --> REFLECT
        REFLECT -->|apply if eligible| GUIDANCE
    end

    TLOG --> SELECT[Selection Policy
GT alignment + trust tie-break]
    GUIDANCE --> SELECT

    SELECT --> EXPORT[[Final Verdict Export
JSONL · three-line format]]
    class EXPORT store

    SELECT --> METRICS[[Metrics Snapshot
pass@K · disagreement buckets]]
    class METRICS store
```

### Iterative Training-Free Cycle

1. **Baseline** — mission guidance for `挡风板安装检查` contains focus and initial experiences.
2. **Rollout** — sampler (K=4) generates verdicts for `QC-TEMP-20241206-0015502`.
3. **Signals + Critic** — minimal deterministic signals are computed; CriticEngine emits strict-JSON per-candidate `{summary, critique, ...}` with length caps and at most 6 candidates per group.
4. **Reflection** — after the batch, ReflectionEngine proposes structured operations; uplift gating applies only when `apply_if_delta` is set; holdout evaluation is deferred by default.
5. **Apply + Select** — if eligible, guidance updates are applied atomically; selection chooses GT-aligned verdicts with trust-weighted tie-breakers; exports three-line verdicts.
6. **Iterate** — subsequent batches use the updated guidance; snapshots persist, never-empty experiences enforced.