# Tasks — add-stageb-guidance-scaffold-hypothesis-gating

- [x] Update `stage-b-training-free` spec deltas for scaffold + hypothesis gating.
- [x] Extend `initial_guidance` seeds to include mission-wise `S*` scaffold keys (at least `挡风板安装检查`), allowing `G0+` to be mutable while keeping `S*` immutable.
- [x] Update rollout prompt rendering to show `S*` as immutable scaffold and `G0+` as mutable guidance (no keyword-coupling to `G1`).
- [x] Extend reflection ops prompt + parser to support `hypotheses[]` output; enforce strict forbidden phrase filtering (including common “复核/不应直接/佐证” variants).
- [x] Add hypothesis pool persistence and deterministic promotion gate (threshold-based, per-mission).
- [x] Ensure learnability closure counts hypothesis evidence (`L == H ∪ E`) and retries uncovered learnables; preserve bounded budgets.
- [x] Add unit tests: scaffold immutability, hypothesis promotion threshold, no third-state leakage into guidance, and closure behavior.
- [x] Update runtime docs: Stage‑B artifacts + how to interpret hypothesis pool vs guidance.
