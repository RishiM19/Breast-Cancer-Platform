"""Validation-only input sensitivity research; never imports the production backend.

Use --audit-only, --smoke-test, or --run; see generated README for interpretation.
All perturbations are in native image coordinates, before model preprocessing.
Positive score/logit delta means INCREASED support for the ORIGINAL predicted class.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageOps
from scipy.ndimage import distance_transform_edt
from torchvision.transforms import functional as TF

from lumina_gradcam_all_classifiers import (
    ROOT, MEAN, STD, PROFILES, load_classifier, preprocess, gradcam,
)

OUTPUT = ROOT / "data/processed/lumina_reliability_investigation"
MANIFEST = ROOT / "data/processed/bra_val.csv"
FILLS = ("local_median", "blur")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve(raw):
    p = Path(str(raw))
    return (p if p.is_absolute() else ROOT / p).resolve()


class ValidationAccessGuard:
    """Deny Python file opens under data except this output and val allowlist.

    This records Python-level accesses, not an OS-wide/native-library tracer.
    The allowlist comes ONLY from bra_val.csv; no other split is inspected.
    """
    def __init__(self, output):
        self.allowed = {MANIFEST.resolve()}
        self.output = output.resolve()
        self.accessed = set()
        self.denied = []

    def __call__(self, event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(ROOT / "data"):
            if path.is_relative_to(self.output):
                return
            if path not in self.allowed:
                self.denied.append(str(path))
                raise PermissionError(f"Not a validation-allowlisted file: {path}")
            mode, flags = args[1], args[2]
            if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            ):
                raise PermissionError(f"Dataset write refused: {path}")
            self.accessed.add(str(path))


def read_pair(row):
    with Image.open(resolve(row.image_path)) as f:
        rgb = np.asarray(f.convert("RGB")).copy()
    with Image.open(resolve(row.mask_path)) as f:
        if f.mode not in ("1", "L", "P"):
            raise ValueError(f"Unsupported mask mode {f.mode}")
        mask = np.asarray(f.convert("L"))
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"Image/mask geometry mismatch: {rgb.shape[:2]} vs {mask.shape}")
    values = set(np.unique(mask).tolist())
    if not values <= {0, 255}:
        raise ValueError(f"Expected binary 0/255 annotation, got {sorted(values)[:20]}")
    lesion = mask > 127
    if not lesion.any() or lesion.all():
        raise ValueError("Empty or all-image lesion annotation")
    return rgb, lesion


def audit_dataset(out, guard):
    df = pd.read_csv(MANIFEST, dtype={"case_id": str, "image_id": str})
    required = {"case_id", "image_id", "image_path", "mask_path", "birads", "pathology"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    for col in ("image_path", "mask_path"):
        for value in df[col].dropna():
            path = resolve(value)
            expected = ROOT / "data/BUS_BRA/BUSBRA" / ("Images" if col == "image_path" else "Masks")
            if not path.is_relative_to(expected):
                raise ValueError(f"Unexpected validation path: {path}")
            guard.allowed.add(path)
    records, valid, seen_paths, seen_hashes = [], [], set(), set()
    for idx, row in df.iterrows():
        record = {"val_row": int(idx), "case_id": str(row.case_id), "image_id": str(row.image_id),
                  "image_exists": resolve(row.image_path).is_file(),
                  "mask_exists": resolve(row.mask_path).is_file()}
        try:
            if row[list(required)].isna().any() or not str(row.case_id).strip():
                raise ValueError("Null/blank required metadata")
            if float(row.birads) not in (2, 3, 4, 5):
                raise ValueError("Invalid BI-RADS label")
            if str(row.pathology).lower() not in ("benign", "malignant"):
                raise ValueError("Invalid pathology label")
            rgb, lesion = read_pair(row)
            digest = sha256(resolve(row.image_path))
            duplicate = resolve(row.image_path) in seen_paths or digest in seen_hashes
            record.update(width=rgb.shape[1], height=rgb.shape[0], lesion_pixels=int(lesion.sum()),
                          lesion_fraction=float(lesion.mean()), image_sha256=digest,
                          mask_sha256=sha256(resolve(row.mask_path)), duplicate=duplicate,
                          birads=int(row.birads), valid=not duplicate)
            if "image_sha256" in df and pd.notna(row.image_sha256) and row.image_sha256 != digest:
                raise ValueError("Image checksum differs from manifest")
            seen_paths.add(resolve(row.image_path)); seen_hashes.add(digest)
            if not duplicate:
                valid.append(idx)
        except Exception as exc:
            record.update(valid=False, error=f"{type(exc).__name__}: {exc}")
        records.append(record)
    pd.DataFrame(records).to_csv(out / "audit_rows.csv", index=False)
    accepted = df.loc[valid].copy()
    accepted["val_row"] = accepted.index
    summary = {
        "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
        "rows": len(df), "unique_cases": int(df.case_id.nunique()),
        "valid_unique_images": len(accepted), "eligible_cases": int(accepted.case_id.nunique()),
        "missing_images": sum(not r["image_exists"] for r in records),
        "missing_masks": sum(not r["mask_exists"] for r in records),
        "duplicate_paths": int(df.image_path.duplicated().sum()),
        "duplicate_image_ids": int(df.image_id.duplicated().sum()),
        "duplicate_content_or_paths": sum(r.get("duplicate", False) for r in records),
        "errors": [r for r in records if "error" in r],
        "cases_with_conflicting_birads": df.groupby("case_id").birads.nunique().loc[lambda x: x > 1].index.tolist(),
        "cases_by_birads": {str(k): int(g.case_id.nunique()) for k, g in accepted.groupby("birads")},
        "image_counts_by_birads": {str(k): len(g) for k, g in accepted.groupby("birads")},
        "test_manifest_read": False,
        "scope": "Validation manifest only; train/test membership independently unverified (other splits not read).",
    }
    write_json(out / "audit_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return accepted, summary


def choose_cases(df, count, seed):
    """Seeded, label-stratified round robin, independent of predictions/effects."""
    rng = np.random.default_rng(seed)
    candidates = {}
    for label in (2, 3, 4, 5):
        group = df[df.birads == label].sort_values(["case_id", "image_id"])
        candidates[label] = list(rng.permutation(group.index))
    chosen, used = [], set()
    target = df.case_id.nunique() if count == 0 else min(count, df.case_id.nunique())
    while len(chosen) < target:
        progress = False
        for label in (2, 3, 4, 5):
            while candidates[label]:
                idx = candidates[label].pop()
                case = str(df.loc[idx, "case_id"])
                if case not in used:
                    chosen.append(idx); used.add(case); progress = True
                    break
            if len(chosen) == target:
                break
        if not progress:
            break
    return df.loc[chosen].copy()


def transform_image(rgb, profile):
    image = Image.fromarray(rgb)
    size = (profile.input_size,) * 2
    if profile.preprocessing == "pad":
        image = ImageOps.pad(image, size, method=Image.Resampling.BILINEAR,
                             color=(0, 0, 0), centering=(0.5, 0.5))
    else:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return TF.normalize(TF.to_tensor(image), MEAN, STD)


def transform_mask(mask, profile):
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    size = (profile.input_size,) * 2
    if profile.preprocessing == "pad":
        image = ImageOps.pad(image, size, method=Image.Resampling.NEAREST, color=0, centering=(0.5, 0.5))
    else:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def compact_region(allowed, count, center):
    """Exact-area nearest pixels to a center; ties broken by stable flat index."""
    indices = np.flatnonzero(allowed)
    if count <= 0 or count > len(indices):
        raise ValueError(f"Cannot place {count} pixels in {len(indices)} available pixels")
    y, x = np.unravel_index(indices, allowed.shape)
    d = (y - center[0]) ** 2 + (x - center[1]) ** 2
    chosen = indices[np.argsort(d, kind="stable")[:count]]
    result = np.zeros(allowed.shape, dtype=bool)
    result.flat[chosen] = True
    return result


def case_rng(seed, image_id, region, repeat):
    key = f"{seed}|{image_id}|{region}|{repeat}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))


def define_regions(rgb, lesion, row, config, seed, repeats, border_fraction, top_fraction):
    h, w = lesion.shape
    width, height = max(1, round(w * border_fraction)), max(1, round(h * top_fraction))
    left = np.zeros_like(lesion); left[:, :width] = True
    right = np.zeros_like(lesion); right[:, w-width:] = True
    top = np.zeros_like(lesion); top[:height] = True
    interior = ~(left | right | top | lesion)
    context_count = min(round(h*w*0.10), int(interior.sum()))
    regions = {
        "left_border": (left & ~lesion, f"left {width} native columns, minus lesion"),
        "right_border": (right & ~lesion, f"right {width} native columns, minus lesion"),
        "top_strip": (top & ~lesion, f"top {height} native rows, minus lesion"),
        "lesion": (lesion.copy(), "all original annotation foreground pixels"),
    }
    if context_count:
        regions["context"] = (compact_region(interior, context_count, (h/2, w/2)),
                              "nearest-to-image-center 10% native pixels excluding lesion, top/side strips; tissue not segmented")
    skips = []
    spec = config.get("images", {}).get(str(row.image_id))
    if spec is None or not spec.get("verified", False):
        skips.append({"image_id": str(row.image_id), "region": "text", "reason": "No manually verified text ROI"})
    else:
        if spec.get("size") != [w, h] or spec.get("image_sha256") != sha256(resolve(row.image_path)):
            raise ValueError(f"Text ROI geometry/checksum mismatch for {row.image_id}")
        if not spec.get("review_note"):
            raise ValueError("Verified text ROI requires a review note")
        text_mask = np.zeros_like(lesion)
        for box in spec.get("boxes", []):
            if len(box) != 4 or not all(isinstance(v, int) for v in box):
                raise ValueError("Text boxes require four integer pixel coordinates")
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError("Text box is outside the image")
            text_mask[y0:y1, x0:x1] = True
        if text_mask.any():
            regions["text"] = (text_mask & ~lesion, "reviewed text rectangles minus lesion; " + json.dumps(spec))
        else:
            skips.append({"image_id": str(row.image_id), "region": "text", "reason": spec["review_note"]})
    variants = []
    for name, (mask, rule) in regions.items():
        n = int(mask.sum())
        if not n:
            skips.append({"image_id": str(row.image_id), "region": name, "reason": "Empty region after lesion exclusion"})
            continue
        if name != "lesion" and (mask & lesion).any():
            raise AssertionError("Artifact intersects lesion")
        variants.append(dict(region=name, matched_region=name, repeat=-1, mask=mask, rule=rule,
                             control_overlap_fraction=0.0))
        # Exact pixel-area controls; exclude BOTH lesion and tested region when possible.
        allowed = ~lesion & ~mask
        disjoint = int(allowed.sum()) >= n
        if not disjoint:
            allowed = ~lesion
        if int(allowed.sum()) < n:
            skips.append({"image_id": str(row.image_id), "region": "random_control", "matched_region": name,
                          "reason": "Insufficient lesion-free pixels for exact area matching"})
            continue
        points = np.argwhere(allowed)
        for rep in range(repeats):
            rng = case_rng(seed, row.image_id, name, rep)
            center = points[int(rng.integers(len(points)))]
            control = compact_region(allowed, n, center)
            assert int(control.sum()) == n and not (control & lesion).any()
            variants.append(dict(region="random_control", matched_region=name, repeat=rep, mask=control,
                                 rule=f"exact-area compact nearest-pixel control; center_yx={center.tolist()}; disjoint={disjoint}",
                                 control_overlap_fraction=float((control & mask).sum() / n)))
    return variants, skips


def perturb(rgb, mask, lesion, fill):
    """Modify exactly the selected native pixels, with no resampling here."""
    scale = min(mask.shape)
    if fill == "blur":
        radius = max(2.0, 0.025 * scale)
        replacement = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius)))
        details = {"gaussian_radius_native_px": radius}
    elif fill == "local_median":
        radius = max(3, round(0.03 * scale))
        ring = (distance_transform_edt(~mask) <= radius) & ~mask & ~lesion
        fallback = False
        if not ring.any():
            ring = ~mask & ~lesion
            fallback = True
        if not ring.any():
            raise ValueError("No lesion-free donor pixels for median fill")
        replacement = np.median(rgb[ring], axis=0).astype(np.uint8)
        details = {"ring_radius_native_px": radius, "median_rgb": replacement.tolist(),
                   "donor_pixels": int(ring.sum()), "global_donor_fallback": fallback}
    else:
        raise ValueError(f"Unknown fill {fill}")
    result = rgb.copy()
    result[mask] = replacement[mask] if replacement.ndim == 3 else replacement
    if not np.array_equal(result[~mask], rgb[~mask]):
        raise AssertionError("Pixels outside the requested ROI changed")
    return result, details


def infer(model, tensors, device, batch_size):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            logits = model(torch.stack(tensors[start:start+batch_size]).to(device))
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite model output")
            outputs.append(logits.cpu())
    return torch.cat(outputs).numpy()


def softmax(logits):
    values = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def verify_original(model, profile, row, rgb, device, smoke):
    _, reference = preprocess(resolve(row.image_path), profile)
    current = transform_image(rgb, profile).unsqueeze(0)
    torch.testing.assert_close(current, reference, rtol=0, atol=0)
    # Compare directly with original validation dataset transformations; do not call dataloader factories.
    training_match = None
    if profile.model_id == "bra":
        from bra_dataset import get_bra_transforms
        actual = get_bra_transforms()[1](Image.fromarray(rgb))
        torch.testing.assert_close(current[0], actual, rtol=0, atol=0)
        training_match = True
    elif profile.model_id == "uclm":
        from uclm_dataset import get_uclm_transforms
        actual = get_uclm_transforms()[1](Image.fromarray(rgb))
        torch.testing.assert_close(current[0], actual, rtol=0, atol=0)
        training_match = True
    elif profile.model_id == "birads":
        from train_bra_birads_v1 import BRADataset
        one = pd.DataFrame([{"__image_abs__": resolve(row.image_path), "birads": int(row.birads)}])
        actual, _ = BRADataset(one, training=False)[0]
        torch.testing.assert_close(current[0], actual, rtol=0, atol=0)
        training_match = True
    with torch.no_grad():
        a = model(current.to(device)).cpu().numpy()[0]
        b = model(reference.to(device)).cpu().numpy()[0]
    np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
    cam_error = None
    if smoke:
        _, cam_probs = gradcam(model, reference.to(device), int(a.argmax()))
        cam_error = float(np.abs(cam_probs - softmax(a)).max())
        np.testing.assert_allclose(cam_probs, softmax(a), atol=1e-5, rtol=1e-5)
    return {"tensor_exact_match": True, "logit_max_error": float(np.abs(a-b).max()),
            "training_eval_transform_match": training_match,
            "gradcam_probability_max_error": cam_error}


def summarize(frame, group_cols, seed):
    results = []
    for key, group in frame.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        record = dict(zip(group_cols, key))
        # Repeat/fill/region observations are not independent patients. Average within case first.
        cases = group.groupby("case_id").agg(
            score_delta=("score_delta", "mean"), logit_delta=("logit_delta", "mean"),
            abs_score_delta=("abs_score_delta", "mean"), flip=("class_changed", "mean"),
            drop=("score_drop", "mean"), increase=("score_increase", "mean"))
        rng = np.random.default_rng(seed)
        record.update(n_cases=len(cases), n_observations=len(group),
                      class_flip_rate=float(cases.flip.mean()), mean_score_delta=float(cases.score_delta.mean()),
                      median_score_delta=float(cases.score_delta.median()),
                      mean_logit_delta=float(cases.logit_delta.mean()), median_logit_delta=float(cases.logit_delta.median()),
                      mean_abs_score_delta=float(cases.abs_score_delta.mean()),
                      median_abs_score_delta=float(cases.abs_score_delta.median()),
                      score_delta_sd=float(cases.score_delta.std()) if len(cases) > 1 else 0.0,
                      mean_score_drop=float(cases["drop"].mean()), mean_score_increase=float(cases.increase.mean()),
                      mean_area_fraction=float(group.occluded_fraction.mean()))
        for metric, output in (("score_delta", "score_delta"), ("abs_score_delta", "abs_score_delta"), ("flip", "flip_rate")):
            values = cases[metric].to_numpy()
            if len(cases) >= 5:
                boot = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                record[output+"_ci_low"], record[output+"_ci_high"] = float(lo), float(hi)
            else:
                record[output+"_ci_low"], record[output+"_ci_high"] = None, None
        results.append(record)
    return pd.DataFrame(results)


def paired_controls(frame, seed):
    perturbed = frame[(frame.region != "original") & (frame.region != "random_control")].copy()
    controls = frame[frame.region == "random_control"].groupby(
        ["model", "case_id", "fill", "matched_region"], as_index=False
    ).agg(control_abs=("abs_score_delta", "mean"), control_delta=("score_delta", "mean"),
          control_drop=("score_drop", "mean"), control_area=("occluded_fraction", "mean"))
    paired = perturbed.merge(controls, on=["model", "case_id", "fill", "matched_region"], how="inner")
    np.testing.assert_allclose(paired.occluded_fraction, paired.control_area, rtol=0, atol=1e-12)
    paired["excess_abs_effect"] = paired.abs_score_delta - paired.control_abs
    paired["excess_support_drop"] = paired.score_drop - paired.control_drop
    rows = []
    for (model, region, fill), g in paired.groupby(["model", "region", "fill"]):
        rng = np.random.default_rng(seed)
        values = g.excess_abs_effect.to_numpy()
        lo = hi = None
        if len(values) >= 5:
            means = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
            lo, hi = [float(x) for x in np.quantile(means, [.025, .975])]
        rows.append(dict(model=model, region=region, fill=fill, n_cases=len(g),
                         mean_excess_abs_effect=float(values.mean()), median_excess_abs_effect=float(np.median(values)),
                         fraction_cases_exceed_control=float((values > 0).mean()), ci_low=lo, ci_high=hi,
                         mean_excess_support_drop=float(g.excess_support_drop.mean())))
    return paired, pd.DataFrame(rows)


def plot_contact(selected, out):
    for start in range(0, len(selected), 12):
        part = selected.iloc[start:start+12]
        fig, axes = plt.subplots(3, 4, figsize=(14, 14))
        for ax in axes.flat:
            ax.axis("off")
        for ax, (_, row) in zip(axes.flat, part.iterrows()):
            rgb, lesion = read_pair(row)
            ax.imshow(rgb); ax.contour(lesion, levels=[.5], colors=["lime"], linewidths=.7)
            ax.set_title(f"{row.image_id} | case {row.case_id} | BI-RADS {row.birads}")
        fig.tight_layout(); fig.savefig(out / f"selection_contact_{start//12+1:02d}.png", dpi=130); plt.close(fig)


def plots(frame, summary, out, selected, config, args):
    region_order = ["text", "left_border", "right_border", "top_strip", "lesion", "context"]
    models = list(frame.model.unique())
    fig, axes = plt.subplots(len(models), 2, figsize=(14, 3.5*len(models)), squeeze=False)
    for i, model in enumerate(models):
        for j, fill in enumerate(FILLS):
            s = summary[(summary.model == model) & (summary.fill == fill)]
            names = [r for r in region_order if r in s.region.values]
            primary = s[s.region.isin(names)].set_index("region").loc[names]
            control = s[s.region == "random_control"].set_index("matched_region").reindex(names)
            x = np.arange(len(names))
            axes[i,j].bar(x-.18, primary.mean_abs_score_delta*100, .36, label="Region")
            axes[i,j].bar(x+.18, control.mean_abs_score_delta*100, .36, label="Area-matched controls")
            lo = primary.abs_score_delta_ci_low.to_numpy(dtype=float)
            hi = primary.abs_score_delta_ci_high.to_numpy(dtype=float)
            mean = primary.mean_abs_score_delta.to_numpy()
            if np.isfinite(lo).all():
                axes[i,j].errorbar(x-.18, mean*100, yerr=np.maximum(0, np.array([mean-lo, hi-mean])*100), fmt="none", color="black", capsize=2)
            axes[i,j].set_xticks(x, names, rotation=25)
            axes[i,j].set_ylabel("Mean absolute score change (pp)")
            text_n = int(primary.loc["text", "n_cases"]) if "text" in names else 0
            axes[i,j].set_title(f"{model.upper()} | {fill} | n={len(selected)}; text n={text_n}")
            axes[i,j].legend(fontsize=8)
    fig.suptitle("Validation-only sensitivity | original predicted class | case bootstrap 95% intervals", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .975)); fig.savefig(out / "sensitivity_comparison.png", dpi=150); plt.close(fig)
    # Representative examples fixed by manifest selection, never chosen for largest effect.
    row = selected.iloc[0]
    rgb, lesion = read_pair(row)
    variants, _ = define_regions(rgb, lesion, row, config, args.seed, args.control_repeats, args.border_fraction, args.top_fraction)
    chosen = [v for v in variants if v["region"] != "random_control"]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(12, 3.7*len(chosen)), squeeze=False)
    model = "birads" if "birads" in models else models[0]
    for i, v in enumerate(chosen):
        pert, _ = perturb(rgb, v["mask"], lesion, "local_median")
        rec = frame[(frame.image_id == row.image_id) & (frame.model == model) &
                    (frame.region == v["region"]) & (frame.fill == "local_median")].iloc[0]
        axes[i,0].imshow(rgb); axes[i,0].contour(lesion, levels=[.5], colors=["lime"], linewidths=.8)
        axes[i,0].set_title(f"Original: {rec.original_predicted_class}\n{model} target score {rec.original_score:.3f}")
        axes[i,1].imshow(rgb); axes[i,1].imshow(np.ma.masked_where(~v["mask"], v["mask"]), cmap="autumn", alpha=.55, vmin=0, vmax=1)
        axes[i,1].set_title(f"Modified region: {v['region']}\n{rec.occluded_fraction:.1%} of native pixels")
        axes[i,2].imshow(pert)
        axes[i,2].set_title(f"Median fill: {rec.perturbed_predicted_class}\nOriginal-class score change {100*rec.score_delta:+.2f} pp")
        for ax in axes[i]: ax.axis("off")
    fig.suptitle(f"{row.image_id} | green = original lesion boundary | no attribution heatmaps")
    fig.tight_layout(rect=(0, 0, 1, .975)); fig.savefig(out / "representative_examples.png", dpi=140); plt.close(fig)
    text_rows = selected[selected.image_id.isin(config.get("images", {}))]
    if len(text_rows):
        fig, axes = plt.subplots((len(text_rows)+3)//4, 4, figsize=(14, 4*((len(text_rows)+3)//4)), squeeze=False)
        for ax in axes.flat: ax.axis("off")
        for ax, (_, row) in zip(axes.flat, text_rows.iterrows()):
            rgb, lesion = read_pair(row)
            variants, _ = define_regions(rgb, lesion, row, config, args.seed, args.control_repeats, args.border_fraction, args.top_fraction)
            text_mask = next((v["mask"] for v in variants if v["region"] == "text"), None)
            ax.imshow(rgb)
            ax.contour(lesion, levels=[.5], colors=["lime"], linewidths=.7)
            if text_mask is not None:
                ax.contour(text_mask, levels=[.5], colors=["orange"], linewidths=1)
            ax.set_title(f"{row.image_id} | BI-RADS {row.birads}")
        fig.suptitle("Reviewed native text ROIs (orange) and original lesion boundaries (green)")
        fig.tight_layout(rect=(0, 0, 1, .975)); fig.savefig(out / "text_roi_review.png", dpi=140); plt.close(fig)


def run_experiment(args, out, selected, audit, config):
    device = torch.device("cpu" if args.device == "cpu" else "cuda" if torch.cuda.is_available() else "cpu")
    names = list(PROFILES) if args.models == "all" else args.models.split(",")
    if not names or any(n not in PROFILES for n in names):
        raise ValueError("--models must be all or comma-separated busi,uclm,bra,birads")
    model_data, verification = {}, {}
    for name in names:
        model, extra = load_classifier(PROFILES[name], device)
        model_data[name] = model
        verification[name] = {"checkpoint_loaded_strictly": True, "profile": asdict(PROFILES[name]),
                              "checkpoint_metadata": extra,
                              "checkpoint_sha256": sha256(ROOT / "models" / PROFILES[name].checkpoint)}
    rows, skips, region_log = [], [], []
    region_dir = out / "regions"; region_dir.mkdir()
    example_dir = out / "perturbed_examples"; example_dir.mkdir()
    for index, (_, row) in enumerate(selected.iterrows()):
        rgb, lesion = read_pair(row)
        variants, missing = define_regions(rgb, lesion, row, config, args.seed, args.control_repeats, args.border_fraction, args.top_fraction)
        repeated, _ = define_regions(rgb, lesion, row, config, args.seed, args.control_repeats, args.border_fraction, args.top_fraction)
        assert len(variants) == len(repeated)
        assert all(np.array_equal(a["mask"], b["mask"]) for a,b in zip(variants, repeated))
        skips.extend(missing)
        arrays, metadata = [rgb], [dict(region="original", matched_region="original", repeat=-1,
                                       mask=np.zeros_like(lesion), rule="unchanged", fill="none", fill_details={}, control_overlap_fraction=0.0)]
        for v in variants:
            key = f"{row.image_id}_{v['region']}_{v['matched_region']}_{v['repeat']}"
            Image.fromarray(v["mask"].astype(np.uint8)*255).save(region_dir / f"{key}.png")
            region_log.append({k: value for k,value in v.items() if k != "mask"} | {
                "image_id": str(row.image_id), "mask_file": str(region_dir / f"{key}.png"),
                "native_pixels": int(v["mask"].sum()), "mask_sha256": hashlib.sha256(v["mask"].tobytes()).hexdigest()})
            for fill in FILLS:
                pert, details = perturb(rgb, v["mask"], lesion, fill)
                arrays.append(pert); metadata.append(v | dict(fill=fill, fill_details=details))
                if index == 0 and v["region"] != "random_control":
                    Image.fromarray(pert).save(example_dir / f"{key}_{fill}.png")
        for name, model in model_data.items():
            profile = PROFILES[name]
            check = verify_original(model, profile, row, rgb, device, args.smoke_test)
            verification[name].setdefault("original_parity", {})[str(row.image_id)] = check
            tensors = [transform_image(array, profile) for array in arrays]
            logits = infer(model, tensors, device, args.batch_size)
            probs = softmax(logits)
            target = int(probs[0].argmax())
            with torch.no_grad():
                single = model(tensors[0].unsqueeze(0).to(device)).cpu().numpy()[0]
            np.testing.assert_allclose(logits[0], single, atol=2e-5, rtol=2e-5)
            valid = transform_mask(np.ones_like(lesion), profile)
            mapped_lesion = transform_mask(lesion, profile)
            assert not (mapped_lesion & ~valid).any() and mapped_lesion.any()
            for k, meta in enumerate(metadata):
                mask = meta["mask"]
                projected = transform_mask(mask, profile)
                assert not (projected & ~valid).any()
                if meta["region"] != "lesion":
                    assert not (projected & mapped_lesion).any()
                pred = int(probs[k].argmax())
                delta = float(probs[k, target] - probs[0, target])
                gt = str(int(row.birads)) if name == "birads" else str(row.pathology).capitalize()
                original_margin = float(probs[0,target] - np.max(np.delete(probs[0], target)))
                perturbed_margin = float(probs[k,target] - np.max(np.delete(probs[k], target)))
                rows.append(dict(model=name, case_id=str(row.case_id), image_id=str(row.image_id),
                    val_row=int(row.val_row), birads=int(row.birads), ground_truth_label=gt,
                    task="birads" if name == "birads" else "pathology", cross_dataset=name in ("busi", "uclm"),
                    region=meta["region"], matched_region=meta["matched_region"], control_repeat=meta["repeat"], fill=meta["fill"],
                    original_predicted_class=profile.classes[target], perturbed_predicted_class=profile.classes[pred],
                    original_correct=profile.classes[target] == gt, class_changed=pred != target,
                    original_score=float(probs[0,target]), perturbed_score=float(probs[k,target]),
                    score_delta=delta, abs_score_delta=abs(delta), score_drop=max(0,-delta), score_increase=max(0,delta),
                    original_logits=json.dumps(logits[0].tolist()), perturbed_logits=json.dumps(logits[k].tolist()),
                    class_order=json.dumps(profile.classes), logit_delta=float(logits[k,target]-logits[0,target]),
                    original_margin=original_margin, perturbed_margin=perturbed_margin,
                    original_top2_margin=float(np.diff(np.sort(probs[0])[-2:])[0]),
                    perturbed_top2_margin=float(np.diff(np.sort(probs[k])[-2:])[0]),
                    occluded_fraction=float(mask.mean()), native_pixels=int(mask.sum()),
                    model_valid_pixel_fraction=float(projected.sum()/valid.sum()),
                    actual_changed_fraction=float(np.any(arrays[k] != rgb, axis=2).mean()),
                    control_overlap_fraction=meta["control_overlap_fraction"], region_selection=meta["rule"],
                    fill_details=json.dumps(meta["fill_details"])))
        print(f"Completed case {index+1}/{len(selected)}: {row.image_id}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "image_level_results.csv", index=False)
    changed = frame[frame.region != "original"]
    summary = summarize(changed, ["model", "region", "matched_region", "fill"], args.seed)
    summary.to_csv(out / "perturbation_summary.csv", index=False)
    summarize(changed[changed.region != "random_control"], ["model", "fill"], args.seed).to_csv(out / "model_summary.csv", index=False)
    summarize(changed, ["model", "region", "matched_region", "fill", "birads"], args.seed).to_csv(out / "birads_summary.csv", index=False)
    summarize(changed, ["model", "case_id", "region", "matched_region", "fill"], args.seed).to_csv(out / "case_level_summary.csv", index=False)
    paired, comparison = paired_controls(frame, args.seed)
    paired.to_csv(out / "paired_control_results.csv", index=False)
    comparison.to_csv(out / "paired_control_summary.csv", index=False)
    write_json(out / "region_definitions.json", region_log)
    write_json(out / "skipped_regions.json", skips)
    write_json(out / "verification.json", verification)
    plots(frame, summary, out, selected, config, args)
    # Read saved deliverables back; checks reported only after successful completion.
    reloaded = pd.read_csv(out / "image_level_results.csv")
    assert len(reloaded) == len(frame) and reloaded.case_id.nunique() == len(selected)
    assert np.isfinite(reloaded[["score_delta", "logit_delta", "original_score", "perturbed_score"]].to_numpy()).all()
    for path in out.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    for name in ("representative_examples.png", "sensitivity_comparison.png"):
        with Image.open(out / name) as im: im.verify()
    report = dict(status="completed", cases=len(selected), models=names, observations=len(frame), seed=args.seed,
                  device=str(device), torch_version=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                  command=[sys.executable, *sys.argv], created_utc=datetime.now(timezone.utc).isoformat(),
                  selected_cases_by_birads={str(k): len(g) for k,g in selected.groupby("birads")},
                  baseline_birads_correct=int(frame[(frame.model == "birads") & (frame.region == "original")].original_correct.sum()),
                  skipped_regions=len(skips), text_cases=int(frame[frame.region == "text"].case_id.nunique()),
                  checks_passed=["strict checkpoint load", "original tensor/logit parity", "batch/single logit parity",
                                 "native mask geometry", "lesion exclusion", "padding exclusion", "exact control area",
                                 "repeatable random masks", "unchanged outside modified region", "CSV/JSON readback", "plot integrity"],
                  limitations=["BUSI original training source absent; current backend profile reproduced",
                               "Validation-only, exploratory; no clinical validity or causal shortcut proof",
                               "BUSI/UCLM are cross-dataset observations; pathology and BI-RADS are separate tasks",
                               "Softmax scores are uncalibrated", "Area-matched controls do not match shape, texture or anatomy",
                               "Median fill introduces edges; blur retains low frequencies and can import neighboring content",
                               "Context is a fixed 10% patch, not all background; no tissue segmentation",
                               "Native area matching can differ slightly after resizing; model-space area recorded",
                               "Case bootstrap intervals descriptive, no multiplicity correction; small text strata",
                               "No independent test/train membership audit, because those manifests were not opened"])
    write_json(out / "investigation_summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--max-cases", type=int, default=24, help="0 selects one image from every eligible case")
    parser.add_argument("--models", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--control-repeats", type=int, default=3)
    parser.add_argument("--border-fraction", type=float, default=.05)
    parser.add_argument("--top-fraction", type=float, default=.10)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--text-rois", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.max_cases < 0 or args.batch_size < 1 or args.control_repeats < 1:
        parser.error("Counts must be positive; max-cases may be zero for all cases")
    if not 0 < args.border_fraction < .5 or not 0 < args.top_fraction < .5:
        parser.error("Strip fractions must be in (0, 0.5)")
    out = (args.output_dir or (OUTPUT / "audit" if args.audit_only else OUTPUT / "smoke" if args.smoke_test else OUTPUT)).resolve()
    if not out.is_relative_to(OUTPUT):
        parser.error(f"Outputs must stay in {OUTPUT}")
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("audit_summary.json", "image_level_results.csv", "failure.json")):
        raise FileExistsError(f"Refusing to overwrite previous run in {out}; use --output-dir with a fresh subdirectory")
    guard = ValidationAccessGuard(OUTPUT)
    sys.addaudithook(guard)
    try:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.set_num_threads(4)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Avoid TF32 convolution rounding that can vary with batch shape on Ampere.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        df, audit = audit_dataset(out, guard)
        if df.empty: raise ValueError("No valid images")
        selected = df[df.image_id == "bus_0001-l"].copy() if args.smoke_test else choose_cases(df, args.max_cases, args.seed)
        if selected.empty: raise ValueError("Requested smoke-test image absent from valid validation rows")
        assert selected.case_id.nunique() == len(selected)
        selected.to_csv(out / "selected_cases.csv", index=False)
        plot_contact(selected, out)
        if not args.audit_only:
            config = json.loads(args.text_rois.read_text(encoding="utf-8")) if args.text_rois else {}
            write_json(out / "text_roi_config_used.json", config)
            result = run_experiment(args, out, selected, audit, config)
            print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        write_json(out / "failure.json", {"error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        write_json(out / "data_access_log.json", {"allowed_source": "bra_val.csv only", "accessed": sorted(guard.accessed),
                   "denied": guard.denied, "scope": "Python open audit hook; not an OS/native-library trace"})
    if not args.audit_only:
        from report_lumina_reliability_investigation import build_reports
        build_reports(out)


if __name__ == "__main__":
    main()
