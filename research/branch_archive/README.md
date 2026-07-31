# Retired research branch archive

This directory preserves branch-only research code after the project was
consolidated onto `main` on 2026-07-31.

## What is preserved

- `branches.yaml` records all 15 retired branches and their exact Git identity.
- Eight branches had commits that were intentionally not merged into active
  source because they were rejected, comparison-only, profiling-only, or
  otherwise non-canonical.
- Their 13 branch-only commits are stored under `patches/` as full-index,
  binary-capable `git format-patch` records.
- Each unmerged branch tip is also retained by an annotated
  `research-archive/...` Git tag.
- The other seven branch tips are already reachable from `main`, so no
  duplicate patch series or archive tag is required.

Archived code is research evidence. It is not accepted production source and
must not be used as a canonical baseline without a new review and experiment.

## Reconstruct an archived branch

Use the `merge_base`, `patches`, and `archive_tag` fields in `branches.yaml`.
The tag is the exact original tree and is the preferred read-only reference:

```powershell
git switch --detach research-archive/codex/v4174-exact-top2-forward
```

To replay the commit series into a temporary branch:

```powershell
git switch --detach 300a8966f9261904fa1a7e685b8e4e648f89a1f2
git switch -c research-replay
git am research/branch_archive/patches/codex__v4174-exact-top2-forward/*.patch
```

Do not replay archived patches directly onto `main`.
