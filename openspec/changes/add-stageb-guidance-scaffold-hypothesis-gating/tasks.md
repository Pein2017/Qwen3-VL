# Tasks — add-stageb-guidance-scaffold-hypothesis-gating

- [ ] Update `stage-b-training-free` spec deltas for scaffold + hypothesis gating.
- [ ] Extend `initial_guidance` seeds to include mission-wise `S*` scaffold keys (at least `挡风板安装检查`), allowing `G0+` to be mutable while keeping `S*` immutable.
- [ ] Update rollout prompt rendering to show `S*` as immutable scaffold and `G0+` as mutable guidance (no keyword-coupling to `G1`).
- [ ] Extend reflection ops prompt + parser to support `hypotheses[]` output; enforce strict forbidden phrase filtering (including common “复核/不应直接/佐证” variants).
- [ ] Add hypothesis pool persistence and deterministic promotion gate (threshold-based, per-mission).
- [ ] Ensure learnability closure counts hypothesis evidence (`L == H ∪ E`) and retries uncovered learnables; preserve bounded budgets.
- [ ] Add unit tests: scaffold immutability, hypothesis promotion threshold, no third-state leakage into guidance, and closure behavior.
- [ ] Update runtime docs: Stage‑B artifacts + how to interpret hypothesis pool vs guidance.
