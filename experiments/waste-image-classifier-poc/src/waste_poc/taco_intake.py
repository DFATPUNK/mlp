from __future__ import annotations

import html
import csv
import json
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from .utils import CLASS_NAMES, ensure_dir, write_json, write_text
from .v2_dataset import LABEL_MAPPING_COLUMNS, PUBLIC_CLASSIFIER_COLUMNS, PUBLIC_GATE_COLUMNS

SOURCE_DATASET_ID = "taco"
SOURCE_URL_FIELDS = ["flickr_url", "flickr_640_url", "url", "coco_url"]
BLOCKED_LICENSE_PATTERNS = [
    re.compile(r"(^|[^A-Z0-9])NC([^A-Z0-9]|$)", re.IGNORECASE),
    re.compile(r"(^|[^A-Z0-9])ND([^A-Z0-9]|$)", re.IGNORECASE),
    re.compile(r"non[- ]commercial", re.IGNORECASE),
    re.compile(r"no derivatives", re.IGNORECASE),
]

CATEGORY_INVENTORY_COLUMNS = [
    "source_dataset_id",
    "source_category_id",
    "source_label",
    "annotation_count",
    "image_count",
    "mapped_label",
    "mapping_status",
    "mapping_rule_id",
]

EXCLUDED_ROWS_COLUMNS = [
    "exclusion_id",
    "reason",
    "source_dataset_id",
    "source_item_id",
    "relative_path",
    "source_labels",
    "license_status",
    "source_url",
    "source_license",
    "source_license_reference",
]


class TacoIntakeError(ValueError):
    def __init__(self, messages: list[str]):
        self.messages = messages
        super().__init__("\n".join(messages))


def _stable_id(prefix: str, *parts: str) -> str:
    import hashlib

    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _safe_relative_path(value: str) -> bool:
    if not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and path.parts not in {(), (".",)}


def _write_csv(path: str | Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{column: row.get(column, "") for column in columns} for row in rows])


def _read_csv_exact(path: str | Path, columns: list[str], name: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise TacoIntakeError([f"{name}: expected exact columns {columns}, found {reader.fieldnames or []}"])
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise TacoIntakeError([f"annotations: invalid JSON: {exc}"]) from exc
    if not isinstance(payload, dict):
        raise TacoIntakeError(["annotations: expected a JSON object"])
    return payload


def _require_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise TacoIntakeError([f"annotations: required field {key!r} must be a list"])
    if not all(isinstance(item, dict) for item in value):
        raise TacoIntakeError([f"annotations: every item in {key!r} must be an object"])
    return value


def _string_id(value: Any) -> str:
    return str(value).strip()


def _category_label(category: dict[str, Any]) -> str:
    return str(category.get("name", "")).strip()


def _source_url(image: dict[str, Any]) -> str:
    for field in SOURCE_URL_FIELDS:
        value = str(image.get(field, "") or "").strip()
        if value:
            return value
    return ""


def _license_text_is_blocked(*values: str) -> bool:
    text = " ".join(value for value in values if value)
    return any(pattern.search(text) for pattern in BLOCKED_LICENSE_PATTERNS)


def _license_lookup(licenses: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_string_id(license_item.get("id")): license_item for license_item in licenses if _string_id(license_item.get("id"))}


def _license_reference(license_item: dict[str, Any] | None) -> str:
    if not license_item:
        return ""
    for field in ["url", "reference", "license_url"]:
        value = str(license_item.get(field, "") or "").strip()
        if value:
            return value
    return ""


def _source_license(license_item: dict[str, Any] | None, image_license_id: str) -> str:
    if not license_item:
        return ""
    name = str(license_item.get("name", "") or "").strip()
    return name or image_license_id


def _source_attribution(image: dict[str, Any], source_url: str) -> str:
    for field in ["attribution", "author", "creator", "photographer", "flickr_user_name"]:
        value = str(image.get(field, "") or "").strip()
        if value:
            return value
    image_id = _string_id(image.get("id"))
    return f"TACO image {image_id}; source {source_url}" if image_id and source_url else ""


def _resolve_license(image: dict[str, Any], licenses_by_id: dict[str, dict[str, Any]]) -> tuple[str, str, str, str, str]:
    source_url = _source_url(image)
    image_license_id = _string_id(image.get("license"))
    license_item = licenses_by_id.get(image_license_id)
    source_license = _source_license(license_item, image_license_id)
    source_license_reference = _license_reference(license_item)
    source_attribution = _source_attribution(image, source_url)
    if not source_url:
        return "blocked", source_url, source_license, source_license_reference, source_attribution
    if not image_license_id or not source_license or not source_license_reference:
        return "blocked", source_url, source_license, source_license_reference, source_attribution
    if not source_attribution:
        return "blocked", source_url, source_license, source_license_reference, source_attribution
    if _license_text_is_blocked(source_license, source_license_reference):
        return "blocked", source_url, source_license, source_license_reference, source_attribution
    return "eligible_for_review", source_url, source_license, source_license_reference, source_attribution


def _mapping_by_label(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    mappings: dict[str, dict[str, str]] = {}
    messages: list[str] = []
    for row in rows:
        if row["source_dataset_id"] != SOURCE_DATASET_ID:
            continue
        key = row["source_label"]
        if key in mappings:
            messages.append(f"label mapping: duplicate TACO source_label {key!r}")
        mappings[key] = row
    if messages:
        raise TacoIntakeError(messages)
    return mappings


def _object_area_ratio(annotation: dict[str, Any], image: dict[str, Any]) -> float:
    try:
        width = float(image["width"])
        height = float(image["height"])
        bbox = annotation["bbox"]
        bbox_width = float(bbox[2])
        bbox_height = float(bbox[3])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise TacoIntakeError([f"annotation {_string_id(annotation.get('id'))}: bbox and image width/height are required and must be numeric"]) from exc
    if width <= 0 or height <= 0 or bbox_width < 0 or bbox_height < 0:
        raise TacoIntakeError([f"annotation {_string_id(annotation.get('id'))}: bbox and image dimensions must be positive"])
    return min(1.0, (bbox_width * bbox_height) / (width * height))


def _format_ratio(value: float) -> str:
    return f"{value:.6f}"


def _source_labels(annotations: list[dict[str, Any]], categories_by_id: dict[str, dict[str, Any]]) -> list[str]:
    labels = []
    for annotation in annotations:
        category = categories_by_id.get(_string_id(annotation.get("category_id")), {})
        labels.append(_category_label(category))
    return sorted(label for label in labels if label)


def _local_image_exists(image_root: Path, relative_path: str) -> bool:
    return _safe_relative_path(relative_path) and (image_root / relative_path).is_file()


def _image_src(output_dir: Path, image_root: Path, relative_path: str) -> str:
    if not _safe_relative_path(relative_path):
        return ""
    try:
        return Path(image_root / relative_path).resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        import os

        return os.path.relpath(image_root / relative_path, output_dir)


def _classifier_row(
    *,
    image: dict[str, Any],
    annotation: dict[str, Any],
    relative_path: str,
    source_label: str,
    mapping: dict[str, str],
    object_area_ratio: float,
    license_fields: tuple[str, str, str, str, str],
) -> dict[str, str]:
    license_status, source_url, source_license, source_license_reference, source_attribution = license_fields
    image_id = _string_id(image["id"])
    annotation_id = _string_id(annotation["id"])
    return {
        "public_row_id": _stable_id("taco_cls", image_id, annotation_id),
        "source_dataset_id": SOURCE_DATASET_ID,
        "source_item_id": image_id,
        "relative_path": relative_path,
        "source_url": source_url,
        "source_license": source_license,
        "source_license_reference": source_license_reference,
        "source_attribution": source_attribution,
        "license_status": license_status,
        "source_annotation_id": annotation_id,
        "object_area_ratio": _format_ratio(object_area_ratio),
        "source_label": source_label,
        "mapped_label": mapping["mapped_label"],
        "mapping_rule_id": mapping["mapping_rule_id"],
        "annotation_type": "single_object",
        "object_count": "1",
        "approved_for_classifier_training": "false",
        "annotation_notes": "draft generated by TACO intake; requires human review",
    }


def _gate_row(
    *,
    image: dict[str, Any],
    annotations: list[dict[str, Any]],
    relative_path: str,
    auto_route_eligible: str,
    review_reason: str,
    annotation_type: str,
    object_area_ratio: float,
    license_fields: tuple[str, str, str, str, str],
    source_labels: list[str] | None = None,
    mapped_label: str = "",
) -> dict[str, str]:
    license_status, source_url, source_license, source_license_reference, source_attribution = license_fields
    image_id = _string_id(image["id"])
    annotation_ids = ";".join(_string_id(annotation.get("id")) for annotation in annotations)
    return {
        "public_row_id": _stable_id("taco_gate", image_id, annotation_ids, review_reason),
        "source_dataset_id": SOURCE_DATASET_ID,
        "source_item_id": image_id,
        "relative_path": relative_path,
        "source_url": source_url,
        "source_license": source_license,
        "source_license_reference": source_license_reference,
        "source_attribution": source_attribution,
        "license_status": license_status,
        "source_annotation_id": annotation_ids,
        "object_area_ratio": _format_ratio(object_area_ratio),
        "auto_route_eligible": auto_route_eligible,
        "review_reason": review_reason,
        "annotation_type": annotation_type,
        "object_count": str(len(annotations)),
        "approved_for_gate_training": "false",
        "annotation_notes": "draft generated by TACO intake; requires human review",
        "source_label": ";".join(source_labels or []),
        "mapped_label": mapped_label,
    }


def _excluded_row(
    *,
    reason: str,
    image: dict[str, Any],
    relative_path: str,
    source_labels: list[str],
    license_fields: tuple[str, str, str, str, str],
) -> dict[str, str]:
    license_status, source_url, source_license, source_license_reference, _ = license_fields
    image_id = _string_id(image.get("id"))
    return {
        "exclusion_id": _stable_id("taco_excl", image_id, reason, relative_path),
        "reason": reason,
        "source_dataset_id": SOURCE_DATASET_ID,
        "source_item_id": image_id,
        "relative_path": relative_path,
        "source_labels": ";".join(source_labels),
        "license_status": license_status,
        "source_url": source_url,
        "source_license": source_license,
        "source_license_reference": source_license_reference,
    }


def _category_inventory(
    categories: list[dict[str, Any]],
    annotations_by_category: dict[str, list[dict[str, Any]]],
    mappings: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    rows = []
    for category in sorted(categories, key=lambda item: _category_label(item)):
        category_id = _string_id(category.get("id"))
        source_label = _category_label(category)
        mapping = mappings.get(source_label, {})
        category_annotations = annotations_by_category.get(category_id, [])
        rows.append(
            {
                "source_dataset_id": SOURCE_DATASET_ID,
                "source_category_id": category_id,
                "source_label": source_label,
                "annotation_count": str(len(category_annotations)),
                "image_count": str(len({_string_id(annotation.get("image_id")) for annotation in category_annotations})),
                "mapped_label": mapping.get("mapped_label", ""),
                "mapping_status": mapping.get("mapping_status", ""),
                "mapping_rule_id": mapping.get("mapping_rule_id", ""),
            }
        )
    return rows


def _gallery_html(output_dir: Path, image_root: Path, rows: list[dict[str, str]]) -> str:
    body = [
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\"><title>TACO Review Gallery</title>",
        "<style>body{font-family:sans-serif}article{border:1px solid #ccc;margin:1rem 0;padding:1rem}img{max-width:320px;max-height:240px;display:block}</style>",
        "</head><body><h1>TACO Review Gallery</h1>",
    ]
    for row in rows:
        src = html.escape(_image_src(output_dir, image_root, row["relative_path"]), quote=True)
        body.append("<article>")
        body.append(f"<img src=\"{src}\" alt=\"{html.escape(row['source_item_id'], quote=True)}\">")
        for label, key in [
            ("Source item ID", "source_item_id"),
            ("Source labels", "source_label"),
            ("Mapped label", "mapped_label"),
            ("Object count", "object_count"),
            ("Area ratio", "object_area_ratio"),
            ("Proposed gate value", "auto_route_eligible"),
            ("Review reason", "review_reason"),
            ("Licence status", "license_status"),
            ("Source URL", "source_url"),
        ]:
            body.append(f"<p><strong>{label}:</strong> {html.escape(row.get(key, ''), quote=True)}</p>")
        body.append("</article>")
    body.append("</body></html>\n")
    return "\n".join(body)


def prepare_taco_intake(
    *,
    annotations: str | Path,
    image_root: str | Path,
    label_mapping: str | Path,
    output_dir: str | Path,
    min_object_area_ratio: float,
) -> dict:
    if min_object_area_ratio < 0 or min_object_area_ratio > 1:
        raise TacoIntakeError(["--min-object-area-ratio must be in [0, 1]"])

    payload = _load_json(annotations)
    images = _require_list(payload, "images")
    annotations_rows = _require_list(payload, "annotations")
    categories = _require_list(payload, "categories")
    licenses = _require_list(payload, "licenses")
    mapping_rows = _read_csv_exact(label_mapping, LABEL_MAPPING_COLUMNS, "label mapping")
    mappings = _mapping_by_label(mapping_rows)

    images_by_id = {_string_id(image.get("id")): image for image in images if _string_id(image.get("id"))}
    categories_by_id = {_string_id(category.get("id")): category for category in categories if _string_id(category.get("id"))}
    if len(images_by_id) != len(images):
        raise TacoIntakeError(["annotations: every image requires a unique id"])
    if len(categories_by_id) != len(categories):
        raise TacoIntakeError(["annotations: every category requires a unique id"])

    annotations_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    annotations_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations_rows:
        image_id = _string_id(annotation.get("image_id"))
        category_id = _string_id(annotation.get("category_id"))
        if image_id not in images_by_id:
            raise TacoIntakeError([f"annotation {_string_id(annotation.get('id'))}: unknown image_id {image_id!r}"])
        if category_id not in categories_by_id:
            raise TacoIntakeError([f"annotation {_string_id(annotation.get('id'))}: unknown category_id {category_id!r}"])
        annotations_by_image[image_id].append(annotation)
        annotations_by_category[category_id].append(annotation)

    output_dir = ensure_dir(output_dir)
    image_root = Path(image_root)
    licenses_by_id = _license_lookup(licenses)
    classifier_rows: list[dict[str, str]] = []
    gate_rows: list[dict[str, str]] = []
    excluded_rows: list[dict[str, str]] = []

    images_with_single_annotation = 0
    images_with_multiple_annotations = 0
    images_with_eligible_licence = 0
    images_blocked_by_licence = 0

    for image in sorted(images, key=lambda item: _string_id(item.get("id"))):
        image_id = _string_id(image.get("id"))
        relative_path = str(image.get("file_name", "") or "").strip()
        if not relative_path:
            raise TacoIntakeError([f"image {image_id}: file_name is required"])
        image_annotations = sorted(annotations_by_image.get(image_id, []), key=lambda item: _string_id(item.get("id")))
        source_labels = _source_labels(image_annotations, categories_by_id)
        license_fields = _resolve_license(image, licenses_by_id)
        license_status = license_fields[0]
        if len(image_annotations) == 1:
            images_with_single_annotation += 1
        elif len(image_annotations) > 1:
            images_with_multiple_annotations += 1
        if license_status == "eligible_for_review":
            images_with_eligible_licence += 1
        else:
            images_blocked_by_licence += 1
        if license_status != "eligible_for_review":
            excluded_rows.append(_excluded_row(reason="blocked_license", image=image, relative_path=relative_path, source_labels=source_labels, license_fields=license_fields))
            continue
        if not _local_image_exists(image_root, relative_path):
            excluded_rows.append(_excluded_row(reason="missing_local_image", image=image, relative_path=relative_path, source_labels=source_labels, license_fields=license_fields))
            continue
        if not image_annotations:
            excluded_rows.append(_excluded_row(reason="no_annotations", image=image, relative_path=relative_path, source_labels=source_labels, license_fields=license_fields))
            continue

        ratios = [_object_area_ratio(annotation, image) for annotation in image_annotations]
        max_ratio = max(ratios, default=0.0)
        if len(image_annotations) > 1:
            gate_rows.append(
                _gate_row(
                    image=image,
                    annotations=image_annotations,
                    relative_path=relative_path,
                    auto_route_eligible="false",
                    review_reason="multiple_objects",
                    annotation_type="scene",
                    object_area_ratio=max_ratio,
                    license_fields=license_fields,
                    source_labels=source_labels,
                )
            )
            continue

        annotation = image_annotations[0]
        category = categories_by_id[_string_id(annotation.get("category_id"))]
        source_label = _category_label(category)
        mapping = mappings.get(source_label)
        ratio = ratios[0]
        if mapping is None:
            excluded_rows.append(_excluded_row(reason="missing_mapping", image=image, relative_path=relative_path, source_labels=source_labels, license_fields=license_fields))
            continue
        if mapping["mapping_status"] == "approved" and mapping["mapped_label"] in CLASS_NAMES:
            if ratio >= min_object_area_ratio:
                classifier_rows.append(
                    _classifier_row(
                        image=image,
                        annotation=annotation,
                        relative_path=relative_path,
                        source_label=source_label,
                        mapping=mapping,
                        object_area_ratio=ratio,
                        license_fields=license_fields,
                    )
                )
                gate_rows.append(
                    _gate_row(
                        image=image,
                        annotations=image_annotations,
                        relative_path=relative_path,
                        auto_route_eligible="true",
                        review_reason="none",
                        annotation_type="single_object",
                        object_area_ratio=ratio,
                        license_fields=license_fields,
                        source_labels=source_labels,
                        mapped_label=mapping["mapped_label"],
                    )
                )
            else:
                gate_rows.append(
                    _gate_row(
                        image=image,
                        annotations=image_annotations,
                        relative_path=relative_path,
                        auto_route_eligible="false",
                        review_reason="ambiguous_scene",
                        annotation_type="single_object",
                        object_area_ratio=ratio,
                        license_fields=license_fields,
                        source_labels=source_labels,
                        mapped_label=mapping["mapped_label"],
                    )
                )
            continue
        if mapping["mapping_status"] in {"excluded", "needs_review"}:
            gate_rows.append(
                _gate_row(
                    image=image,
                    annotations=image_annotations,
                    relative_path=relative_path,
                    auto_route_eligible="false",
                    review_reason="unsupported_material",
                    annotation_type="single_object",
                    object_area_ratio=ratio,
                    license_fields=license_fields,
                    source_labels=source_labels,
                )
            )
            continue
        excluded_rows.append(_excluded_row(reason="unusable_mapping", image=image, relative_path=relative_path, source_labels=source_labels, license_fields=license_fields))

    inventory_rows = _category_inventory(categories, annotations_by_category, mappings)
    gallery_rows = []
    for row in classifier_rows:
        gallery_rows.append({**row, "auto_route_eligible": "", "review_reason": ""})
    for row in gate_rows:
        gallery_rows.append(row)

    _write_csv(output_dir / "taco_category_inventory.csv", CATEGORY_INVENTORY_COLUMNS, inventory_rows)
    _write_csv(output_dir / "taco_classifier_candidates.draft.csv", PUBLIC_CLASSIFIER_COLUMNS, classifier_rows)
    _write_csv(output_dir / "taco_gate_candidates.draft.csv", PUBLIC_GATE_COLUMNS, gate_rows)
    _write_csv(output_dir / "taco_excluded_rows.csv", EXCLUDED_ROWS_COLUMNS, excluded_rows)
    write_text(output_dir / "taco_review_gallery.html", _gallery_html(output_dir, image_root, gallery_rows))

    report = {
        "total_images": len(images),
        "total_annotations": len(annotations_rows),
        "categories": len(categories),
        "images_with_single_annotation": images_with_single_annotation,
        "images_with_multiple_annotations": images_with_multiple_annotations,
        "images_with_eligible_licence": images_with_eligible_licence,
        "images_blocked_by_licence": images_blocked_by_licence,
        "classifier_draft_rows": len(classifier_rows),
        "gate_draft_rows": len(gate_rows),
        "excluded_rows": len(excluded_rows),
        "min_object_area_ratio": min_object_area_ratio,
    }
    write_json(output_dir / "taco_intake_report.json", report)
    return report
