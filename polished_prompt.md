```markdown
# Review Spec Proposal: Add Irrelevant Summary Source for Summary Mode Training

## Goal
Review and assess the spec proposal in `openspec/changes/add-irrelevant-summary-source` for adding an irrelevant image dataset to summary mode training. The proposal aims to mitigate overfitting where the model predicts BBU-related content on irrelevant images, losing generalization ability. The spec has not been implemented yet.

## Context
- **Proposal location**: `openspec/changes/add-irrelevant-summary-source/`
- **Problem**: Summary SFT is overfitting to BBU scene priors; model hallucinates BBU content on irrelevant images
- **Solution approach**: Add a lightweight "irrelevant" source dataset that always yields summary `无关图片` to regularize hallucinations
- **Scope**: Summary-mode only; negative/irrelevant images; no geometry or objects required
- **Related docs**: 
  - `docs/data/DATA_AND_DATASETS.md` (summary mode format, fusion configs)
  - `docs/data/DATA_JSONL_CONTRACT.md` (JSONL schema requirements)
  - `openspec/AGENTS.md` (OpenSpec workflow and validation)
  - `openspec/project.md` (project conventions)

## Review Scope
Review the following files in the proposal:
1. **`proposal.md`** - Problem statement, solution approach, scope, risks, validation plan
2. **`design.md`** - Technical design decisions (dataset semantics, integration, validation)
3. **`tasks.md`** - Implementation checklist
4. **`specs/multi-dataset-fusion/spec.md`** - Spec deltas (ADDED requirements with scenarios)

## Review Criteria

### 1. OpenSpec Compliance
- Run `openspec validate add-irrelevant-summary-source --strict` and verify all issues are resolved
- Verify spec deltas use correct format: `## ADDED|MODIFIED|REMOVED Requirements` headers
- Ensure each requirement has at least one `#### Scenario:` with WHEN/THEN/AND structure
- Check that change-id is unique and follows kebab-case verb-led naming

### 2. Technical Soundness
- **Summary-only records**: Verify that empty `objects` with `summary: "无关图片"` aligns with `docs/data/DATA_JSONL_CONTRACT.md` (summary is optional, but when present must be non-empty)
- **Fusion integration**: Confirm the `irrelevant` wrapper design (domain=source, aug off, curriculum off) is consistent with existing fusion patterns in `src/datasets/builders/` and `configs/fusion/`
- **Helper script**: Validate that the JSONL generation helper handles EXIF-aware dimension extraction (per `data_conversion/utils/file_ops.py::FileOperations.get_image_dimensions`) and relative path resolution
- **Ratio knob**: Assess whether the proposed small ratio (0.01–0.03) is appropriate and won't cause oversampling issues with a tiny pool

### 3. Alignment with Repo Patterns
- **Summary mode**: Verify alignment with existing summary mode configs (`configs/summary.yaml`, `configs/dlora/summary.yaml`) and prompts (`src/config/prompts.py::build_summary_system_prompt`)
- **Fusion config**: Check that the proposal follows patterns from `configs/fusion/` and `scripts/fuse_datasets.py`
- **Validation**: Ensure the helper script output can be validated using `scripts/validate_dense_jsonl_contract.py` or similar tools

### 4. Risk Assessment
- Evaluate the identified risks in `proposal.md`:
  - Oversampling a tiny pool skewing training
  - Summary-only records with empty objects staying compatible with fusion validation
- Assess whether mitigation strategies (small default ratio, source-domain flagging) are sufficient

### 5. Completeness
- Verify all implementation tasks in `tasks.md` are actionable and in correct order
- Check that design decisions in `design.md` cover all integration points
- Ensure spec deltas cover all three requirements (wrapper, summary-only records, helper)

## Expected Output
Provide a structured review with:
1. **OpenSpec validation status**: Pass/fail with any remaining issues
2. **Technical assessment**: Strengths, concerns, and recommendations for each review criterion
3. **Approval recommendation**: Approve, approve with conditions, or request revisions (with specific items)
4. **Implementation readiness**: Whether the proposal is ready for implementation or needs clarification

## Validation Steps
1. Run `openspec validate add-irrelevant-summary-source --strict` and capture output
2. Review proposal files against OpenSpec conventions in `openspec/AGENTS.md`
3. Cross-reference technical design with relevant code/docs:
   - `src/datasets/builders/` (fusion dataset wrappers)
   - `src/config/prompts.py` (summary prompt template)
   - `docs/data/DATA_AND_DATASETS.md` (summary mode requirements)
   - `data_conversion/utils/file_ops.py` (image dimension extraction)
4. Check for conflicts with active changes: `openspec list`
5. Verify alignment with project conventions in `openspec/project.md`

## Constraints
- Do not modify any files; this is a review-only task
- Stay focused on the proposal's scope (summary-mode only, irrelevant images)
- Reference real paths/configs from the codebase; do not invent details
- If critical information is missing and blocking assessment, ask one concise clarifying question
```
