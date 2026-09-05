from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path, PurePosixPath

from .clip_candidate import DEFAULT_HF_MODEL_ID
from .images import PREPROCESSING_VERSION, load_rgb_image
from .manifests import find_trashnet_image_root
from .utils import ensure_dir, read_json, sha256_file, utc_now_iso, write_json
from .v2_dataset import CLASSIFICATION_OUTPUT_COLUMNS, GATE_OUTPUT_COLUMNS

GATE_MODEL_TYPE = "clip_vit_b32_logistic_regression_gate"
GATE_MODEL_FORMAT_VERSION = "phase_0_9_v1"
DEFAULT_GATE_THRESHOLD = 0.5
DEFAULT_SEED = 42

GATE_MANIFEST_COLUMNS = [
    "gate_example_id",
    "relative_path",
    "resolved_image_path",
    "auto_route_eligible",
    "source_kind",
    "source_dataset_id",
    "source_item_id",
    "source_split",
    "review_reason",
    "split",
    "notes",
]

GATE_PREDICTION_COLUMNS = [
    "image_id",
    "relative_path",
    "expected_auto_route_eligible",
    "predicted_auto_route_eligible",
    "gate_probability_auto_route_eligible",
    "threshold",
    "gate_decision",
    "evaluation_label",
]


class GateExperimentError(ValueError):
    pass


def _read_csv(path: str | Path, *, expected_columns: list[str] | None = None, required_columns: list[str] | None = None) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if expected_columns is not None and columns != expected_columns:
            raise GateExperimentError(f"{path}: expected exact columns {expected_columns}, found {columns}")
        missing = [column for column in (required_columns or []) if column not in columns]
        if missing:
            raise GateExperimentError(f"{path}: missing required columns {missing}")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    return columns, rows


def _write_csv(path: str | Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{column: row.get(column, "") for column in columns} for row in rows])


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _safe_relative_path(value: str) -> bool:
    if not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and path.parts not in {(), (".",)}


def _resolve_local_path(value: str | Path, root: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(root) / path
    return path.resolve()


def resolve_trashnet_image_root(
    *,
    poc_root: str | Path,
    trashnet_image_root: str | Path | None,
    metadata_path: str | Path = "data/raw/trashnet_source_metadata.json",
) -> Path:
    poc_root = Path(poc_root).resolve()
    candidates: list[Path] = []
    if trashnet_image_root is not None:
        candidates.append(_resolve_local_path(trashnet_image_root, poc_root))
    else:
        metadata = _resolve_local_path(metadata_path, poc_root)
        if metadata.is_file():
            payload = read_json(metadata)
            detected = payload.get("source_directory_detected")
            if detected:
                candidates.append(_resolve_local_path(detected, poc_root))
        candidates.append(poc_root / "data" / "raw" / "trashnet-source")

    errors: list[str] = []
    for candidate in candidates:
        try:
            return find_trashnet_image_root(candidate)
        except FileNotFoundError as exc:
            errors.append(str(exc))
    raise GateExperimentError(
        "Could not resolve the TrashNet image root. Pass --trashnet-image-root pointing to "
        "the extracted source repository or its dataset-resized directory. "
        + " ".join(errors)
    )


def _root_for_gate_row(
    row: dict[str, str],
    *,
    feedback_image_root: str | Path | None,
    public_image_root: str | Path | None,
    poc_root: str | Path,
) -> Path:
    if row["source_kind"] == "feedback":
        root = feedback_image_root
        option = "--feedback-image-root"
    elif row["source_kind"] == "public_dataset":
        root = public_image_root
        option = "--public-image-root"
    else:
        raise GateExperimentError(
            f"gate_row_id={row['gate_row_id']}: unsupported source_kind={row['source_kind']!r}; "
            "expected feedback or public_dataset"
        )
    if root is None:
        raise GateExperimentError(f"gate_row_id={row['gate_row_id']}: {option} is required for selected {row['source_kind']} rows")
    return _resolve_local_path(root, poc_root)


def build_gate_manifest(
    *,
    v2_classification_manifest: str | Path,
    v2_gate_manifest: str | Path,
    output_dir: str | Path,
    negative_source_filter: str,
    trashnet_image_root: str | Path | None,
    feedback_image_root: str | Path | None,
    public_image_root: str | Path | None,
    poc_root: str | Path,
    trashnet_metadata: str | Path = "data/raw/trashnet_source_metadata.json",
) -> dict:
    if negative_source_filter not in {"feedback", "all"}:
        raise GateExperimentError("negative_source_filter must be one of: feedback, all")

    _, classification_rows = _read_csv(v2_classification_manifest, expected_columns=CLASSIFICATION_OUTPUT_COLUMNS)
    _, gate_rows = _read_csv(v2_gate_manifest, expected_columns=GATE_OUTPUT_COLUMNS)
    trashnet_root = resolve_trashnet_image_root(
        poc_root=poc_root,
        trashnet_image_root=trashnet_image_root,
        metadata_path=trashnet_metadata,
    )

    output_rows: list[dict[str, str]] = []
    missing_paths: list[Path] = []
    selected_gate_rows = 0
    skipped_eligible_gate_rows = 0

    for row in classification_rows:
        if row["source_kind"] != "trashnet" or row["split"] != "train":
            continue
        relative_path = row["relative_path"]
        if not _safe_relative_path(relative_path):
            raise GateExperimentError(f"dataset_row_id={row['dataset_row_id']}: unsafe relative_path={relative_path!r}")
        image_path = (trashnet_root / relative_path).resolve()
        if not image_path.is_file():
            missing_paths.append(image_path)
        output_rows.append(
            {
                "gate_example_id": _stable_id("gate", "trashnet", row["dataset_row_id"]),
                "relative_path": relative_path,
                "resolved_image_path": str(image_path),
                "auto_route_eligible": "true",
                "source_kind": "trashnet",
                "source_dataset_id": row["source_dataset_id"],
                "source_item_id": row["source_item_id"] or row["image_id"],
                "source_split": row["source_split"],
                "review_reason": "none",
                "split": "train",
                "notes": "Clean TrashNet train example used as a positive gate example.",
            }
        )

    for row in gate_rows:
        if negative_source_filter == "feedback" and row["source_kind"] != "feedback":
            continue
        if row["source_kind"] not in {"feedback", "public_dataset"}:
            continue
        if row["auto_route_eligible"] == "true":
            skipped_eligible_gate_rows += 1
            continue
        if row["auto_route_eligible"] != "false":
            raise GateExperimentError(
                f"gate_row_id={row['gate_row_id']}: auto_route_eligible must be lowercase true or false"
            )
        selected_gate_rows += 1
        relative_path = row["relative_path"]
        if not _safe_relative_path(relative_path):
            raise GateExperimentError(f"gate_row_id={row['gate_row_id']}: unsafe relative_path={relative_path!r}")
        image_root = _root_for_gate_row(
            row,
            feedback_image_root=feedback_image_root,
            public_image_root=public_image_root,
            poc_root=poc_root,
        )
        image_path = (image_root / relative_path).resolve()
        if not image_path.is_file():
            missing_paths.append(image_path)
        output_rows.append(
            {
                "gate_example_id": _stable_id("gate", row["source_kind"], row["gate_row_id"]),
                "relative_path": relative_path,
                "resolved_image_path": str(image_path),
                "auto_route_eligible": "false",
                "source_kind": row["source_kind"],
                "source_dataset_id": row["source_dataset_id"],
                "source_item_id": row["source_item_id"] or row["image_id"],
                "source_split": row["source_split"],
                "review_reason": row["review_reason"],
                "split": "train",
                "notes": "Reviewed gate example used as a negative gate example.",
            }
        )

    if missing_paths:
        listed = "\n".join(f"- {path}" for path in sorted(set(missing_paths), key=str))
        raise GateExperimentError(f"Gate manifest images are missing:\n{listed}")

    output_rows.sort(key=lambda row: (row["auto_route_eligible"] != "true", row["source_kind"], row["source_item_id"], row["gate_example_id"]))
    ids = [row["gate_example_id"] for row in output_rows]
    duplicate_ids = sorted(value for value, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        raise GateExperimentError(f"Duplicate gate_example_id values: {duplicate_ids}")
    resolved_paths = [row["resolved_image_path"] for row in output_rows]
    duplicate_paths = sorted(value for value, count in Counter(resolved_paths).items() if count > 1)
    if duplicate_paths:
        raise GateExperimentError(f"The same resolved image appears more than once in the gate manifest: {duplicate_paths}")
    if not output_rows:
        raise GateExperimentError("No gate examples matched the requested inputs and source filter")

    output_dir = ensure_dir(output_dir)
    manifest_path = output_dir / "gate_manifest.csv"
    _write_csv(manifest_path, GATE_MANIFEST_COLUMNS, output_rows)
    report = {
        "gate_manifest_path": str(manifest_path),
        "gate_manifest_sha256": sha256_file(manifest_path),
        "negative_source_filter": negative_source_filter,
        "rows": len(output_rows),
        "positive_rows": sum(row["auto_route_eligible"] == "true" for row in output_rows),
        "negative_rows": sum(row["auto_route_eligible"] == "false" for row in output_rows),
        "selected_gate_rows": selected_gate_rows,
        "skipped_auto_route_eligible_gate_rows": skipped_eligible_gate_rows,
        "source_kind_counts": dict(Counter(row["source_kind"] for row in output_rows)),
        "trashnet_image_root": str(trashnet_root),
    }
    write_json(output_dir / "gate_manifest_report.json", report)
    return report


def load_gate_manifest(path: str | Path) -> list[dict[str, str]]:
    _, rows = _read_csv(path, expected_columns=GATE_MANIFEST_COLUMNS)
    ids = Counter(row["gate_example_id"] for row in rows)
    duplicates = sorted(value for value, count in ids.items() if not value or count > 1)
    if duplicates:
        raise GateExperimentError(f"Gate manifest has blank or duplicate gate_example_id values: {duplicates}")
    for index, row in enumerate(rows, start=2):
        if row["auto_route_eligible"] not in {"true", "false"}:
            raise GateExperimentError(f"Gate manifest row {index}: auto_route_eligible must be lowercase true or false")
        if not Path(row["resolved_image_path"]).is_file():
            raise GateExperimentError(f"Gate manifest row {index}: image is missing: {row['resolved_image_path']}")
    return rows


def _deterministic_positive_sample(rows: list[dict[str, str]], max_positive_examples: int | None, seed: int) -> list[dict[str, str]]:
    positives = [row for row in rows if row["auto_route_eligible"] == "true"]
    negatives = [row for row in rows if row["auto_route_eligible"] == "false"]
    if max_positive_examples is not None:
        if max_positive_examples <= 0:
            raise GateExperimentError("--max-positive-examples must be positive")
        positives = sorted(
            positives,
            key=lambda row: hashlib.sha256(f"{seed}\x1f{row['gate_example_id']}".encode("utf-8")).hexdigest(),
        )[:max_positive_examples]
    selected = positives + negatives
    return sorted(selected, key=lambda row: row["gate_example_id"])


def load_clip_embedding_components(hf_model_id: str):
    from transformers import CLIPImageProcessor, CLIPModel

    model = CLIPModel.from_pretrained(hf_model_id, use_safetensors=True)
    processor = CLIPImageProcessor.from_pretrained(hf_model_id)
    return model, processor


def extract_clip_embeddings(
    image_paths: list[str | Path],
    *,
    hf_model_id: str = DEFAULT_HF_MODEL_ID,
    device: str = "auto",
    batch_size: int = 16,
    component_loader=load_clip_embedding_components,
) -> tuple[object, str]:
    if batch_size <= 0:
        raise GateExperimentError("--batch-size must be positive")
    import numpy as np
    import torch

    from .device import resolve_device

    selected_device = resolve_device(device, torch_module=torch)
    model, processor = component_loader(hf_model_id)
    model.requires_grad_(False)
    model.eval()
    model.to(selected_device)
    chunks = []
    for start in range(0, len(image_paths), batch_size):
        images = [load_rgb_image(path) for path in image_paths[start : start + batch_size]]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(selected_device)
        with torch.no_grad():
            features = model.get_image_features(pixel_values=pixel_values)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        chunks.append(features.detach().cpu().numpy().astype(np.float32))
    if not chunks:
        raise GateExperimentError("Cannot extract embeddings from an empty image list")
    return np.concatenate(chunks, axis=0), str(selected_device)


def _load_or_extract_embeddings(
    rows: list[dict[str, str]],
    *,
    cache_path: str | Path,
    hf_model_id: str,
    device: str,
    batch_size: int,
) -> tuple[object, str, bool]:
    import numpy as np

    cache_path = Path(cache_path)
    ids = [row["gate_example_id"] for row in rows]
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_ids = cached["gate_example_ids"].astype(str).tolist()
            cached_model_id = str(cached["hf_model_id"].item())
            if cached_ids == ids and cached_model_id == hf_model_id:
                return cached["embeddings"], str(cached["selected_device"].item()), True
    embeddings, selected_device = extract_clip_embeddings(
        [row["resolved_image_path"] for row in rows],
        hf_model_id=hf_model_id,
        device=device,
        batch_size=batch_size,
    )
    ensure_dir(cache_path.parent)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        gate_example_ids=np.asarray(ids),
        hf_model_id=np.asarray(hf_model_id),
        selected_device=np.asarray(selected_device),
    )
    return embeddings, selected_device, False


def train_gate_model(
    *,
    gate_manifest: str | Path,
    output_dir: str | Path,
    max_positive_examples: int | None = 300,
    seed: int = DEFAULT_SEED,
    hf_model_id: str = DEFAULT_HF_MODEL_ID,
    device: str = "auto",
    batch_size: int = 16,
    threshold: float = DEFAULT_GATE_THRESHOLD,
) -> dict:
    if not 0.0 <= threshold <= 1.0:
        raise GateExperimentError("--threshold must be in [0, 1]")
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    rows = _deterministic_positive_sample(load_gate_manifest(gate_manifest), max_positive_examples, seed)
    targets = np.asarray([1 if row["auto_route_eligible"] == "true" else 0 for row in rows], dtype=np.int64)
    if set(targets.tolist()) != {0, 1}:
        raise GateExperimentError("Gate training requires at least one positive and one negative example")

    output_dir = ensure_dir(output_dir)
    embeddings, selected_device, cache_reused = _load_or_extract_embeddings(
        rows,
        cache_path=output_dir / "clip_embeddings.npz",
        hf_model_id=hf_model_id,
        device=device,
        batch_size=batch_size,
    )
    classifier = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
    classifier.fit(embeddings, targets)
    bundle = {
        "model_type": GATE_MODEL_TYPE,
        "format_version": GATE_MODEL_FORMAT_VERSION,
        "hf_model_id": hf_model_id,
        "threshold": threshold,
        "seed": seed,
        "preprocessing_version": PREPROCESSING_VERSION,
        "classifier": classifier,
    }
    model_path = output_dir / "gate_model.joblib"
    joblib.dump(bundle, model_path)
    manifest_used = output_dir / "gate_manifest_used.csv"
    _write_csv(manifest_used, GATE_MANIFEST_COLUMNS, rows)
    probabilities = classifier.predict_proba(embeddings)[:, list(classifier.classes_).index(1)]
    training_predictions = []
    for row, probability in zip(rows, probabilities):
        training_predictions.append(
            {
                **row,
                "gate_probability_auto_route_eligible": f"{float(probability):.10f}",
                "predicted_auto_route_eligible": str(bool(probability >= threshold)).lower(),
            }
        )
    _write_csv(
        output_dir / "gate_predictions_train.csv",
        GATE_MANIFEST_COLUMNS + ["gate_probability_auto_route_eligible", "predicted_auto_route_eligible"],
        training_predictions,
    )
    report = {
        "model_type": GATE_MODEL_TYPE,
        "format_version": GATE_MODEL_FORMAT_VERSION,
        "hf_model_id": hf_model_id,
        "rows": len(rows),
        "positive_rows": int(targets.sum()),
        "negative_rows": int((targets == 0).sum()),
        "class_weight": "balanced",
        "max_positive_examples": max_positive_examples,
        "seed": seed,
        "threshold": threshold,
        "selected_device": selected_device,
        "embedding_cache_reused": cache_reused,
        "gate_manifest_sha256": sha256_file(gate_manifest),
        "gate_manifest_used_sha256": sha256_file(manifest_used),
        "gate_model_sha256": sha256_file(model_path),
        "preprocessing_version": PREPROCESSING_VERSION,
        "timestamp": utc_now_iso(),
    }
    write_json(output_dir / "gate_training_report.json", report)
    return report


def load_gate_model(path: str | Path) -> dict:
    import joblib

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("model_type") != GATE_MODEL_TYPE:
        raise GateExperimentError(f"{path}: not a {GATE_MODEL_TYPE} model bundle")
    if bundle.get("format_version") != GATE_MODEL_FORMAT_VERSION:
        raise GateExperimentError(
            f"{path}: unsupported gate model format {bundle.get('format_version')!r}; expected {GATE_MODEL_FORMAT_VERSION}"
        )
    if "classifier" not in bundle or not bundle.get("hf_model_id"):
        raise GateExperimentError(f"{path}: gate model bundle is missing classifier or hf_model_id")
    return bundle


def _bool_text(value: str, *, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise GateExperimentError(f"{label} must be true or false")


def evaluate_gate_model(
    *,
    gate_model: str | Path,
    external_manifest: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    expected_auto_route_eligible: bool,
    evaluation_label: str,
    device: str = "auto",
    batch_size: int = 16,
    threshold: float | None = None,
) -> dict:
    import numpy as np

    bundle = load_gate_model(gate_model)
    selected_threshold = float(bundle["threshold"] if threshold is None else threshold)
    if not 0.0 <= selected_threshold <= 1.0:
        raise GateExperimentError("Evaluation threshold must be in [0, 1]")
    _, rows = _read_csv(external_manifest, required_columns=["image_id", "relative_path"])
    if not rows:
        raise GateExperimentError("External manifest contains no rows")
    image_root = Path(image_root).resolve()
    image_paths: list[Path] = []
    missing_paths: list[Path] = []
    ids = Counter(row["image_id"] for row in rows)
    duplicates = sorted(value for value, count in ids.items() if not value or count > 1)
    if duplicates:
        raise GateExperimentError(f"External manifest has blank or duplicate image_id values: {duplicates}")
    for index, row in enumerate(rows, start=2):
        if not _safe_relative_path(row["relative_path"]):
            raise GateExperimentError(f"External manifest row {index}: unsafe relative_path={row['relative_path']!r}")
        image_path = (image_root / row["relative_path"]).resolve()
        image_paths.append(image_path)
        if not image_path.is_file():
            missing_paths.append(image_path)
    if missing_paths:
        listed = "\n".join(f"- {path}" for path in sorted(set(missing_paths), key=str))
        raise GateExperimentError(f"External evaluation images are missing:\n{listed}")

    output_dir = ensure_dir(output_dir)
    cache_rows = [
        {
            "gate_example_id": row["image_id"],
            "resolved_image_path": str(path),
        }
        for row, path in zip(rows, image_paths)
    ]
    embeddings, selected_device, cache_reused = _load_or_extract_embeddings(
        cache_rows,
        cache_path=output_dir / "clip_embeddings.npz",
        hf_model_id=bundle["hf_model_id"],
        device=device,
        batch_size=batch_size,
    )
    classifier = bundle["classifier"]
    probabilities = classifier.predict_proba(embeddings)[:, list(classifier.classes_).index(1)]
    predictions: list[dict[str, object]] = []
    for row, probability in zip(rows, probabilities):
        predicted = bool(probability >= selected_threshold)
        predictions.append(
            {
                "image_id": row["image_id"],
                "relative_path": row["relative_path"],
                "expected_auto_route_eligible": str(expected_auto_route_eligible).lower(),
                "predicted_auto_route_eligible": str(predicted).lower(),
                "gate_probability_auto_route_eligible": f"{float(probability):.10f}",
                "threshold": selected_threshold,
                "gate_decision": "allow_classifier" if predicted else "needs_review",
                "evaluation_label": evaluation_label,
            }
        )
    prediction_path = output_dir / "gate_predictions.csv"
    _write_csv(prediction_path, GATE_PREDICTION_COLUMNS, predictions)

    expected = np.full(len(predictions), expected_auto_route_eligible, dtype=bool)
    predicted = np.asarray([row["predicted_auto_route_eligible"] == "true" for row in predictions], dtype=bool)
    negative_mask = ~expected
    positive_mask = expected
    correctly_blocked = int((~predicted & negative_mask).sum())
    missed_negative = int((predicted & negative_mask).sum())
    false_positive = int((~predicted & positive_mask).sum())
    expected_negative_count = int(negative_mask.sum())
    expected_positive_count = int(positive_mask.sum())
    report = {
        "evaluation_label": evaluation_label,
        "rows": len(predictions),
        "expected_negative_count": expected_negative_count,
        "expected_positive_count": expected_positive_count,
        "predicted_negative_count": int((~predicted).sum()),
        "correctly_blocked_count": correctly_blocked,
        "missed_negative_count": missed_negative,
        "negative_recall": None if expected_negative_count == 0 else correctly_blocked / expected_negative_count,
        "false_positive_count": false_positive,
        "false_positive_rate_if_known": None if expected_positive_count == 0 else false_positive / expected_positive_count,
        "threshold": selected_threshold,
        "selected_device": selected_device,
        "embedding_cache_reused": cache_reused,
        "gate_model_sha256": sha256_file(gate_model),
        "external_manifest_sha256": sha256_file(external_manifest),
        "gate_predictions_sha256": sha256_file(prediction_path),
        "timestamp": utc_now_iso(),
    }
    write_json(output_dir / "gate_evaluation_report.json", report)
    return report


def _baseline_is_auto_route(row: dict[str, str]) -> bool:
    action = (row.get("recommended_action") or row.get("routing_decision") or row.get("route") or "").strip()
    if action == "needs_review":
        return False
    if action == "auto_route":
        return True
    return bool(action)


def _is_unsafe_auto_route(row: dict[str, str], is_auto_route: bool) -> bool:
    if not is_auto_route:
        return False
    if row.get("expected_routing") == "needs_review":
        return True
    expected_label = row.get("expected_label", "")
    return bool(expected_label and row.get("predicted_label") != expected_label)


def combine_policy_rows(
    baseline_rows: list[dict[str, str]],
    gate_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict]:
    baseline_ids = Counter(row.get("image_id", "") for row in baseline_rows)
    gate_ids = Counter(row.get("image_id", "") for row in gate_rows)
    duplicate_baseline = sorted(value for value, count in baseline_ids.items() if not value or count > 1)
    duplicate_gate = sorted(value for value, count in gate_ids.items() if not value or count > 1)
    if duplicate_baseline:
        raise GateExperimentError(f"Baseline predictions have blank or duplicate image_id values: {duplicate_baseline}")
    if duplicate_gate:
        raise GateExperimentError(f"Gate predictions have blank or duplicate image_id values: {duplicate_gate}")
    if set(baseline_ids) != set(gate_ids):
        missing_gate = sorted(set(baseline_ids) - set(gate_ids))
        missing_baseline = sorted(set(gate_ids) - set(baseline_ids))
        raise GateExperimentError(
            f"Prediction image_id sets do not match; missing gate rows={missing_gate}, missing baseline rows={missing_baseline}"
        )

    gate_by_id = {row["image_id"]: row for row in gate_rows}
    combined_rows: list[dict[str, object]] = []
    for baseline in baseline_rows:
        gate = gate_by_id[baseline["image_id"]]
        if baseline.get("relative_path") and gate.get("relative_path") and baseline["relative_path"] != gate["relative_path"]:
            raise GateExperimentError(
                f"image_id={baseline['image_id']}: relative_path mismatch between baseline "
                f"{baseline['relative_path']!r} and gate {gate['relative_path']!r}"
            )
        gate_allows = _bool_text(
            gate.get("predicted_auto_route_eligible", ""),
            label=f"image_id={baseline['image_id']} predicted_auto_route_eligible",
        )
        baseline_auto = _baseline_is_auto_route(baseline)
        combined_auto = baseline_auto and gate_allows
        combined_rows.append(
            {
                **baseline,
                "baseline_policy_route": "auto_route" if baseline_auto else "needs_review",
                "gate_predicted_auto_route_eligible": str(gate_allows).lower(),
                "gate_probability_auto_route_eligible": gate.get("gate_probability_auto_route_eligible", ""),
                "gate_decision": gate.get("gate_decision", ""),
                "combined_policy_route": "auto_route" if combined_auto else "needs_review",
                "combined_is_unsafe_auto_route": str(_is_unsafe_auto_route(baseline, combined_auto)).lower(),
            }
        )

    baseline_auto_count = sum(_baseline_is_auto_route(row) for row in baseline_rows)
    combined_auto_count = sum(row["combined_policy_route"] == "auto_route" for row in combined_rows)
    baseline_unsafe = sum(_is_unsafe_auto_route(row, _baseline_is_auto_route(row)) for row in baseline_rows)
    combined_unsafe = sum(row["combined_is_unsafe_auto_route"] == "true" for row in combined_rows)
    expected_review = [row for row in baseline_rows if row.get("expected_routing") == "needs_review"]
    review_before = sum(not _baseline_is_auto_route(row) for row in expected_review)
    combined_by_id = {row["image_id"]: row for row in combined_rows}
    review_after = sum(combined_by_id[row["image_id"]]["combined_policy_route"] == "needs_review" for row in expected_review)
    report = {
        "rows": len(baseline_rows),
        "expected_review_count": len(expected_review),
        "baseline_auto_route_count": baseline_auto_count,
        "baseline_needs_review_count": len(baseline_rows) - baseline_auto_count,
        "baseline_unsafe_auto_route_count": baseline_unsafe,
        "combined_auto_route_count": combined_auto_count,
        "combined_needs_review_count": len(combined_rows) - combined_auto_count,
        "combined_unsafe_auto_route_count": combined_unsafe,
        "unsafe_auto_route_delta": combined_unsafe - baseline_unsafe,
        "review_recall_before": None if not expected_review else review_before / len(expected_review),
        "review_recall_after": None if not expected_review else review_after / len(expected_review),
    }
    return combined_rows, report


def evaluate_combined_policy(
    *,
    v1_external_predictions: str | Path,
    gate_predictions: str | Path,
    output_dir: str | Path,
) -> dict:
    baseline_columns, baseline_rows = _read_csv(
        v1_external_predictions,
        required_columns=["image_id", "relative_path", "expected_label", "expected_routing", "predicted_label"],
    )
    _, gate_rows = _read_csv(gate_predictions, expected_columns=GATE_PREDICTION_COLUMNS)
    combined_rows, report = combine_policy_rows(baseline_rows, gate_rows)
    extra_columns = [
        "baseline_policy_route",
        "gate_predicted_auto_route_eligible",
        "gate_probability_auto_route_eligible",
        "gate_decision",
        "combined_policy_route",
        "combined_is_unsafe_auto_route",
    ]
    output_dir = ensure_dir(output_dir)
    predictions_path = output_dir / "combined_policy_predictions.csv"
    _write_csv(predictions_path, baseline_columns + extra_columns, combined_rows)
    report.update(
        {
            "v1_external_predictions_sha256": sha256_file(v1_external_predictions),
            "gate_predictions_sha256": sha256_file(gate_predictions),
            "combined_policy_predictions_sha256": sha256_file(predictions_path),
            "timestamp": utc_now_iso(),
        }
    )
    write_json(output_dir / "combined_policy_report.json", report)
    return report
