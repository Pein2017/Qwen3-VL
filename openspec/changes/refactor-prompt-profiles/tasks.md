# Tasks

- [ ] Add prompt profile dataclasses (training vs inference) and a composition API for summary prompts.
- [ ] Define domain knowledge packs as Python dataclasses for BBU and RRU, including schema hints, priors, and restrictions.
- [ ] Extend config loading to resolve `prompts.profile` and `prompts.domain` for summary mode; keep `prompts.system/user` overrides authoritative.
- [ ] Update Stage‑A inference prompt building to use the runtime profile and domain packs; add a CLI/config toggle for prompt profile/domain if needed.
- [ ] Update Stage‑B prompt building to inject domain knowledge as a read‑only block separate from guidance reflection.
- [ ] Add tests asserting training prompts exclude domain terms and runtime prompts include the correct domain pack.
- [ ] Update docs (`docs/training/REFERENCE.md`, `docs/runtime/STAGE_A_RUNTIME.md`, `docs/runtime/STAGE_A_STAGE_B.md`) to document prompt profiles and inference domain knowledge.
- [ ] Run `openspec validate refactor-prompt-profiles --strict` and address any issues.
