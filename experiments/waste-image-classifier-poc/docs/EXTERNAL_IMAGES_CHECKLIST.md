# External Images Checklist

Add exactly 20 manually sourced images to `data/external_images/images/` and fill `data/external_images/external_manifest.csv`.

This set is a qualitative challenge set. It is not statistically representative and must not be used as the only evidence for production readiness.

## Required image mix

1. **12 clean in-scope images**: two single-primary-item images per TrashNet class:
   - cardboard
   - glass
   - metal
   - paper
   - plastic
   - trash
2. **4 difficult but in-scope images**:
   - unusual angle
   - crumpled item
   - cluttered background
   - imperfect lighting
3. **4 out-of-scope or ambiguous images**:
   - multiple waste items
   - non-waste object
   - image with no obvious primary item
   - dark, blurred, or otherwise unusable image

## Manifest schema

```csv
image_id,relative_path,expected_label,scenario,expected_routing,notes
```

- `expected_label`: one of the six TrashNet labels for in-scope images; blank for out-of-scope or intentionally ambiguous images.
- `scenario`: `in_scope_clean`, `in_scope_hard`, `multi_item`, `non_waste`, `ambiguous`, `low_light`, or `blurred`.
- `expected_routing`: `auto_route_if_confident` or `needs_review`.

Do not download images automatically from Google Images, random websites, or copyright-unclear sources.
