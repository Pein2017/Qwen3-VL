## Title
Align Stage-B prompts to chat-style (ChatML) conversations

## Problem
- Stage-B rollout/critic/reflection prompts occasionally produce “echo” artifacts (model repeats user block or prompt text) and multi-turn continuations because requests are treated like plain text completions instead of bounded chat turns.
- Critic and reflection still feed a single concatenated string to the model without explicit assistant start markers; output can spill extra `user` text and violate line protocol.
- Need a consistent, chat-conversation framing similar to Stage-A inference to reduce prompt drift and parsing failures.

## Proposal
- Adopt explicit chat-style messages (`system` optional, `user` + `assistant`) for Stage-B critic and reflection generation, using tokenizer/processor `apply_chat_template(add_generation_prompt=True)` so generation begins inside the assistant turn.
- Keep existing line-based output protocols, but ensure prompts are wrapped in chat messages to bound generation and minimize prompt echo.
- Add minimal decode safeguards (eos/stop defaults unchanged) as needed to prevent multi-turn spillover.

## Success Criteria
- Critic and reflection generation built via chat-template API with explicit assistant turn.
- Rollout remains chat-based; no regressions to current line protocol.
- Validation: prompts render with `<im_start>user ... <im_end><im_start>assistant` (or model-native equivalent) and unit/path checks pass.
