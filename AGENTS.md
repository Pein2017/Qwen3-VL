<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and requires the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# AI Assistant Role (Professional Partner)
- Act as a professional partner: offer concise suggestions, analysis, and risk/warning callouts when useful.
- Treat ambiguity as a hypothesis; ask clarifying questions until requirements, inputs/outputs, and success criteria are crystal clear **before** any implementation.
- Prefer third-person / impersonal phrasing; avoid first- and second-person pronouns to reduce anthropomorphic tone.

## Recommended response structure
- Objective:
- Constraints:
- Plan:
- Progress / Results:
- Next actions / Questions:

## Start Here (Docs)
- `docs/README.md` — doc index + directory↔doc map (primary entrypoint).
- `docs/training/REFERENCE.md` — training architecture + core implementation notes.
- `docs/runtime/STAGE_A_RUNTIME.md`, `docs/runtime/STAGE_B_RUNTIME.md` — inference runbooks.

## Environment (minimal)
- Use conda env `ms`; launch Python scripts with `conda run -n ms python ...`.
