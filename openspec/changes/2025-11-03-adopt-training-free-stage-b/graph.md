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

    INGEST -->|per mission| GUIDANCE[Mission Guidance File<br/>focus + guidance + preferences]
    class GUIDANCE store

    INGEST --> SCFLAG[[Summary Confidence
high / low / needs_review]]
    INGEST --> LOOP{For each group ticket}

    LOOP -->|reuse qwen-3vl| SAMPLER[Sampler
K rollouts · decode grid]
    class SAMPLER qwen

    SAMPLER --> TLOG[[Trajectory Log JSONL
response text · decode params]]
    class TLOG store

    TLOG --> JUDGE[Deterministic Judge
format ✔ · label match · guidance consistency]

    JUDGE --> ADV[(Semantic Advantage
consistency_score – baseline)]
    JUDGE --> ANNO[[Annotated Trajectories
scores · summary_confidence · label_contradiction]]
    class ANNO store

    ADV --> SELECT[Selection Policy
rule override / top consistency]
    ANNO --> SELECT
    GUIDANCE --> SELECT
    SCFLAG --> SELECT

    SELECT --> EXPORT[[Final Verdict Export
JSONL / Parquet · three-line format]]
    class EXPORT store

    SELECT --> UPDATE[Update Mission Guidance
append or refine guidance/preferences · bump step]
    UPDATE --> GUIDANCE

    SELECT --> METRICS[[Metrics Snapshot
pass@K · disagreement buckets · recheck queue]]
    class METRICS store
```

### Iterative Training-Free Cycle

1. **Baseline** — mission guidance for `挡风板安装检查` only contains focus and a rule “挡风板缺失→不通过”.
2. **Rollout** — sampler (K=4) generates verdicts for `QC-TEMP-20241206-0015502`; judge finds one consistent candidate, others lack备注 coverage and inherit `summary_confidence=low`.
3. **Semantic advantage** — consistent candidate scores +0.35 over baseline, negative scores highlight missing evidence in the rest while flagging `needs_review` due to low summary confidence.
4. **Guidance update** — operator records a new prompt cue "判断前确认备注是否强调未安装", step increments and snapshot is stored; preference for low-confidence summaries emphasises conservative verdicts.
5. **Re-run** — rerolling yields two consistent candidates; selection chooses the higher score, exports a three-line verdict (verdict, rationale, confidence), and logs improved pass@K while clearing the recheck flag (no label contradiction).
6. **Next iteration** — continue across other tickets, only promoting durable guidance tweaks while immutable focus (from `STAGE_B_MISSION_FOCUS`) remains untouched.