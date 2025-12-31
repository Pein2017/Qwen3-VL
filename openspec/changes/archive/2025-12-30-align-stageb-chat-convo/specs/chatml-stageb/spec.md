## Capability
Stage-B prompts use chat-style conversations (ChatML) for critic and reflection

## ADDED Requirements
### Requirement: Critic prompts use chat messages with assistant turn
Critic prompts MUST be framed as chat messages and rendered with the chat template so the assistant turn is explicit.
#### Scenario: Critic prompts rendered via chat template
- Given Stage-B critic builds a prompt for a candidate
- When the prompt is rendered
- Then it MUST be assembled as chat messages (system optional, user content filled) and passed through `apply_chat_template(..., add_generation_prompt=True)` so generation starts in the assistant turn.

### Requirement: Reflection prompts use chat messages with assistant turn
Reflection prompts MUST wrap template + bundle as chat messages and render via chat template before generation.
#### Scenario: Reflection prompts rendered via chat template
- Given Stage-B reflection builds a prompt bundle
- When invoking the LLM to propose guidance updates
- Then it MUST wrap the template + bundle as chat messages and render via `apply_chat_template(..., add_generation_prompt=True)` so the model replies inside the assistant turn.

### Requirement: Rollout prompt remains chat-based
Existing rollout chat-based prompting MUST remain unchanged after critic/reflection updates.
#### Scenario: Rollout prompt compatibility
- Given Stage-B rollout already uses chat-template rendering
- When updating critic/reflection to chat-style
- Then rollout prompt construction MUST remain chat-based and function without regression.
