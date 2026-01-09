---
name: git-hygiene
description: Turn big uncommitted piles into small logical commits, then push to a remote branch safely (no secrets, no force-push).
---

# Git Hygiene: Commit + Push in Time

## When to use
Use this skill whenever:
- A “non-trivial” change is complete (feature slice, refactor, bugfix, test suite update).
- You notice uncommitted changes are piling up.
- You’re about to context-switch, pull, rebase, or run risky experiments.

## Goals
- Keep the working tree small and readable.
- Produce a sequence of *logical commits* (each passes basic checks).
- Avoid “mega-commits”: commit in *logical/similar groups*, not “everything at once”.
- Push to a remote branch early (so work is backed up and reviewable).

## Hard safety rules
- **Never commit secrets** (tokens, private keys, passwords, `.env`, credentials). If suspected, stop and ask to rotate/remove.
- **No destructive commands** unless explicitly requested: no `reset --hard`, no `rebase`, no `push --force`, no history rewriting.
- Prefer pushing to a **feature branch**, not `main`/`master`.

## Workflow (do this in order)

### 0) Establish context
Run:
- `git status`
- `git branch --show-current`
- `git remote -v`

If the current branch has no upstream, plan to push with `git push -u origin HEAD`.

### 1) Summarize the change pile
Collect a quick map:
- `git diff --stat`
- `git diff --name-only`
- If needed: `git diff` (spot-check key areas)

Then propose a commit plan: 2–6 commits max, grouped by intent *and similarity*.
- Prefer separating: refactor vs feature vs tests vs docs vs formatting.
- Prefer separating: unrelated areas/modules.
- Prefer separating: “mechanical” changes (config/args/format) from behavioral changes.

Example plan:
1. “refactor: extract X”
2. “feat: implement Y”
3. “test: add coverage for Y”
4. “chore: docs/formatting”

### 2) For each planned commit: stage narrowly (don’t commit everything at once)
Prefer interactive staging:
- `git add -p` (recommended)
or stage by path:
- `git add path/to/files`

Verify:
- `git diff --cached` (confirm only intended changes are staged)

Rule of thumb: if a staged diff mixes multiple intents, split it into multiple commits.

### 3) Run the smallest meaningful checks
Run the fastest checks that reduce regret:
- lint/format (if present)
- unit tests for touched modules (or the smallest test target)
If checks are slow, at least run smoke tests or a minimal subset.

### 4) Write a high-signal commit message
Default format (imperative, scoped):
- `feat(<area>): ...`
- `fix(<area>): ...`
- `refactor(<area>): ...`
- `test(<area>): ...`
- `chore(<area>): ...`

Body (optional): why, constraints, follow-ups, known gaps.

#### Minimal messages for regular/trivial changes
For standard config changes or argument tweaks (mechanical/expected/no behavioral surprise), use *minimal* commit messages, e.g.:
- `chore: config`
- `chore: args`
- `chore: flags`
- `chore: defaults`
- `chore: formatting`

Commit:
- `git commit -m "type(scope): summary"`

### 5) Repeat until the worktree is clean-ish
Aim for:
- `git status` shows either clean or only intentional leftovers (WIP experiments).

### 6) Push to remote branch
If upstream exists:
- `git push`

If no upstream:
- `git push -u origin HEAD`

### 7) Final report
Provide:
- commit list (`git log --oneline -n <N>`)
- current status (`git status`)
- what remains uncommitted (if anything) and why

## Examples

### Example A: “I refactored + added a feature”
Plan:
1) `refactor(parser): split tokenizer`
2) `feat(parser): support new syntax`
3) `test(parser): add cases for new syntax`

Commands (typical):
- `git add -p`
- `git diff --cached`
- run targeted tests
- `git commit -m "..."`
- repeat
- `git push -u origin HEAD`

### Example B: “Too many changes, I’m lost”
Do:
- Create a commit plan based on `--name-only` and `--stat`
- Stage one directory at a time
- If truly tangled, isolate with interactive staging (`git add -p`) and leave leftovers for a follow-up commit

### Example C: “Mostly config/args tweaks”
Plan:
1) `chore: config`
2) `chore: args`

Commands:
- stage only config files first, commit with minimal message
- stage arg/flag/default changes next, commit with minimal message
