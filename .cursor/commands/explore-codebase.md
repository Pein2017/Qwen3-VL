---
name: /explore-codebase
id: explore-codebase
category: Exploration
description: Explore the codebase depth-first around a topic; gather evidence before proposing advice or edits.
---

$ARGUMENTS

**Purpose**  
Investigate the codebase (and nearby docs) for the given TOPIC before giving advice or touching files. Build evidence, map the relevant surfaces (including shared packages and libraries outside the current root), and surface what you learned so any follow-up response or edit is grounded.

**Mindset**
- Curiosity first, edits later; you are not here to change files under this command.
- Think workspace-wide, not folder-bound: follow imports and references into sibling packages, shared libraries, and tooling repos when they look TOPIC-related.
- Stay topic-scoped; ignore unrelated noise.
- Bias to primary sources (docs + code) over memory.
- Treat `./docs` as the fixed documentation root when present; start with `docs/README.md` to locate canonical maps and reading order.
- If the topic is unclear, pause and ask one precise question.

**When to run**
- New feature / bug / refactor requests.
- Ambiguous asks where context is missing.
- Topics that may span multiple packages, services, or shared libraries.
- Before planning edits or giving firm recommendations.

**Quickstart (5 steps)**
1) Extract TOPIC from the invocation or conversation; if missing, ask.  
2) Load governing instructions (e.g., `openspec/AGENTS.md` if relevant) and `docs/README.md` if it exists.  
3) Walk docs → entrypoints → implementation → configuration → dependencies, depth-first on what’s relevant. Use workspace-wide search (rg/grep, symbol search, references) to jump across packages and libraries, not just the current root directory.  
4) Trace imports/flows only when they help explain the TOPIC, including cross-package and cross-service relationships.  
5) Capture concise, high-level notes (section-level context only); do not edit existing files.

**Exploration Phases (depth-first)**
1) Documentation: start with `docs/README.md` when present, then find topic-relevant READMEs, specs, ADRs; map stated contracts and workflows.  
2) Entry Points: identify scripts/CLIs/mains that invoke the topic across the workspace; check `scripts/README.md` if present; note args/config knobs.  
3) Implementation: read key modules end-to-end; follow important imports (even into other packages or libraries); note types, contracts, error paths.  
4) Configuration & Data: locate YAML/JSON/env knobs; map config keys to code paths and data shapes, including shared config modules.  
5) Dependencies: list external packages and internal helpers the topic relies on; note where they live (sibling packages, shared libs, vendored code) and any version or API constraints.  
6) Relationships: sketch data flow and control flow for the topic; mark extension points, coupling hot spots, and cross-package boundaries.

**Output (return this, not code changes)**
- Topic summary: what the topic is and how it flows through the system (including relevant packages and services); keep it high-level.  
- Key discoveries: important modules, functions, configs, data contracts, and where they are located.  
- Dependencies: external/internal libs used for the topic and how they connect.  
- Open questions/gaps: what needs clarification or deeper inspection.  
- Suggested next steps: where to look next or what to verify; ask for permission before making any edits.

**Rules**
- Do not modify existing files while running this command.
- You may read any file in the workspace (and vendored dependencies) that appears TOPIC-relevant, even if it lives outside the current root directory.
- Prefer concise, evidence-backed notes; always cite file paths (and package/module names) for everything important.
- If you cannot find enough signal, state where you looked (paths, search terms, packages) and propose the next probe (e.g., other repos, services, or owners to consult).
