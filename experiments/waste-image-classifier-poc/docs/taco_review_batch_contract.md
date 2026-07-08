# TACO Review Batch Contract

Phase 0.8.4 builds a small local review/download batch from a Phase 0.8.3 `taco_download_plan.csv`. It does not download images, approve rows for training, create classifier or gate candidate CSVs, or invoke V2 assembly.

## Review Queue, Not Training Data

The batch is a manual review queue. `eligible_for_review` means the row has enough provenance to inspect; it does not mean the row is approved for training, cleared for commercial use, or eligible for V2. Downloading an image also does not grant approval. A human must inspect the image, provenance, and intended use before any later candidate approval.

## Selection Policy

All eligible `classifier_and_gate_candidate` rows are retained. These are rare in the current TACO plan and useful to inspect first.

Gate-only rows are intentionally limited:

```txt
multiple_objects: 24
ambiguous_scene: 14
unsupported_material: 9
```

The defaults produce a compact review target while preserving coverage across common review reasons. When a bucket has fewer eligible rows than requested, the builder includes all available rows in that bucket and records the shortage. It never fills a shortage from another bucket.

## Deterministic Diversity

Selection is deterministic for the same input CSV, seed, and limits. The gate-only sampler groups rows by canonical `mapped_labels`, prefers distinct `source_labels` combinations within each group, and ranks rows using a stable SHA-256 of:

```txt
seed + source_item_id + plan_id
```

Rows are selected round-robin across `mapped_labels` groups to improve visual and semantic variety. The goal is not class balancing or training-data selection; it is a small, reproducible review batch.

## Review Fields

The output `taco_review_batch.csv` marks every row as:

```txt
review_status=pending
review_decision=
review_notes=
```

No output column grants training approval. Later review may leave a row rejected, promote it only for gate training, or promote it as a classifier candidate, but that approval happens outside this batch builder.
