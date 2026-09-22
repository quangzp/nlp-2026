# AGENTS.md

Read `PROJECT.md` first — it is the authoritative, mandatory project context (written in Vietnamese): goals, constraints, dataset schema, and working rules.

## Non-negotiable workflow rules (PROJECT.md §0)
- Never modify files, install packages, or restructure the repo without presenting a plan/diff and getting explicit user confirmation first.
- Ask instead of assuming when requirements are unclear; prefer small, reviewable steps over large changes.

## Repo state
- Documentation-only so far: no code, tests, CI, or dependency manifests exist yet.
- Undecided — confirm with the user before coding: modeling approach (PROJECT.md §4), train/val/test split strategy and seed, whether the LLM few-shot comparison is in scope.

## Hard constraints
- Evaluation must report detection precision/recall/F1, correction accuracy on detected positions, and over-correction rate separately — never a single overall accuracy number (PROJECT.md §6).
- Anti-data-leakage rules are mandatory (PROJECT.md §5): split by sentence before computing any global statistics; augmentation on train only; dedupe/near-dedupe before splitting; fixed seed; never fit anything on dev/test labels.

## VSEC dataset gotchas
- Single `train` split only (`nguyenthanhasia/vsec-vietnamese-spell-correction` on Hugging Face) — you must create your own split.
- `has_errors` is always true (100% of sentences contain errors) → over-correction risk; no fully-clean sentences to learn "change nothing".
- `position` / `error_positions` are 0-based syllable indices, NOT character indices.
- `syllable_annotations` contains edge cases (split syllables, merged words, stray leading characters) — errors are not always 1-to-1 syllable substitutions.

## Conventions
- Project documentation is written in Vietnamese; keep new docs consistent.
