# V2 Dataset Assembly Contract

Phase 0.8 defines the assembly layer for a future V2 dataset. It does not train, evaluate, or select a model. The purpose is to make later dataset growth reproducible, auditable, and leakage-safe before any new experiment consumes the data.

## Assembly Is Separate From Training

The V2 builder combines approved manifests into deterministic candidate outputs. A later training run must explicitly choose a V2 manifest version. Building a manifest is not permission to retrain automatically, and it does not imply a new model score.

## Immutable TrashNet Splits

The existing TrashNet manifest remains the baseline. Its `train`, `validation`, and `test` assignments are preserved exactly. Reviewed feedback and future public-source candidates default to `train` only in this phase.

This keeps validation and test comparable to the original POC and prevents new, inspected, or weakly mapped examples from silently entering model-selection splits.

Raw source manifests are immutable historical records. V2 assembly may exclude source rows during curation, but it must not relabel, move, or rewrite the original source manifest or its split assignments.

Byte-identical TrashNet images with contradictory material labels are quarantined rather than automatically relabelled. All occurrences in a conflicting-label SHA-256 group are excluded from V2 classification output, and `v2_trashnet_quarantine_report.csv` records each excluded occurrence as part of the V2 provenance record.

Enrichment sources must not reuse immutable TrashNet validation or test identities. The builder checks matching non-empty SHA-256 values, matching image IDs, and matching relative paths when the source dataset context is the same.

## Two Datasets, Two Label Spaces

The classification dataset uses only the six material labels:

```txt
cardboard
glass
metal
paper
plastic
trash
```

The review-gate dataset uses the human routing suitability label:

```txt
auto_route_eligible=true|false
```

These datasets must not be merged blindly. A corrected material label is useful for classifier training. A rejected multi-object or unsupported image is useful for review-gate training. `needs_review` remains an operational workflow decision, not a seventh TrashNet class.

## Provenance And Mapping

Every V2 row preserves where it came from:

- `source_kind`: `trashnet`, `feedback`, or `public_dataset`.
- `source_dataset_id`: the upstream dataset or feedback source identifier.
- `source_split`: the source split, such as `train`, `validation`, `test`, `workflow_feedback`, or `public_dataset`.
- `source_item_id`: the source item identifier when available.
- `parent_feedback_id`: the feedback row that produced a candidate row.
- `original_label`: the unmodified source label for mapped public classifier rows.
- `mapping_rule_id`: the explicit approved mapping rule used for public classifier rows.
- `source_commit` and `sha256`: preserved when the source manifest provides them or when a local image root is explicitly supplied.

Feedback candidate provenance is immutable. The V2 builder enforces these `source` / `source_split` pairs directly: `external_diagnostic` / `external_diagnostic_v1`, `workflow_feedback` / `workflow_feedback`, `manual_capture` / `manual_capture`, and `public_dataset` / `public_dataset`.

Public-source labels are never mapped automatically. Each accepted public classifier row must reference an approved mapping rule that maps a specific `source_dataset_id` and `source_label` to one of the six supported material labels.

When non-empty SHA-256 values are available, the classification manifest rejects duplicate hashes across TrashNet, feedback, and public-source classifier rows. Blank hashes are allowed when source images are not locally available.

## External Diagnostic Promotion

`external_diagnostic_v1` began as an inspected diagnostic set. By default, V2 assembly refuses to include promoted rows from that set, even when they arrive through Phase 0.7 candidate CSVs.

If `--allow-promoted-external-diagnostic` is used, the V2 output includes a leakage notice and reports the promoted row count. After promotion, `external_diagnostic_v1` is permanently retired from future comparative or final evaluation. Future model selection requires a new untouched external holdout.
