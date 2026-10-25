"""Research-only, standalone genuine Grad-CAM for four Lumina EfficientNet-B0 classifiers.

PowerShell (from C:\\BC\\Breast-Cancer-Platform):
    python src/lumina_gradcam_all_classifiers.py --audit-only --bra-val-index 0
    python src/lumina_gradcam_all_classifiers.py --run --bra-val-index 0 --model all
    python src/lumina_gradcam_all_classifiers.py --run --bra-val-index 0 --model birads --target 4
    python src/lumina_gradcam_all_classifiers.py --run --image path/to/ultrasound.png --model busi

Notes:
    - No training, external Grad-CAM dependencies, network downloads, or synthetic fallbacks.
    - Pathology model preprocessing follows the previously shared main.py: RGB,
      Resize((256,256)), ImageNet normalization.
    - BI-RADS V1 preprocessing matches gradcam_bra_birads_v1.py exactly:
      RGB, ImageOps.pad to 224x224, ImageNet normalization.
    - The script does NOT use/inspect any test manifest. --bra-val-index uses bra_val.csv only.
    - Grad-CAM is an attribution visualization, NOT lesion segmentation or clinical advice.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.models import efficientnet_b0
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "lumina_gradcam_research"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Profile:
    model_id: str
    checkpoint: str
    classes: tuple[str, ...]
    input_size: int
    preprocessing: str  # "resize" or "pad"


PROFILES = {
    "busi": Profile("busi", "best_model.pth", ("Normal", "Benign", "Malignant"), 256, "resize"),
    "uclm": Profile("uclm", "uclm_v2_best_model.pth", ("Normal", "Benign", "Malignant"), 256, "resize"),
    "bra": Profile("bra", "bra_best_model.pth", ("Benign", "Malignant"), 256, "resize"),
    "birads": Profile("birads", "bra_birads_v1_best.pth", ("2", "3", "4", "5"), 224, "pad"),
}


def source_image(args: argparse.Namespace) -> tuple[Path | None, dict]:
    if args.bra_val_index is not None:
        manifest = ROOT / "data" / "processed" / "bra_val.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Validation manifest missing: {manifest}")
        df = pd.read_csv(manifest)
        for required in ("image_path", "case_id", "birads"):
            if required not in df.columns:
                raise ValueError(f"Required field {required!r} missing from {manifest}")
        idx = args.bra_val_index
        if idx < 0 or idx >= len(df):
            raise ValueError(f"Validation index {idx} outside 0..{len(df)-1}")
        row = df.iloc[idx]
        p = Path(str(row["image_path"]))
        image_path = p if p.is_absolute() else ROOT / p
        metadata = {"source": "BUS-BRA validation", "val_row": int(idx),
                    "case_id": str(row["case_id"]), "annotated_birads": str(row["birads"])}
    elif args.image:
        p = Path(args.image)
        image_path = p if p.is_absolute() else ROOT / p
        metadata = {"source": "user-selected image"}
    else:
        return None, {"source": "not selected"}
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image_path.resolve(), metadata


def checkpoint_state(ckpt: object, profile: Profile) -> tuple[dict, dict]:
    if not isinstance(ckpt, dict):
        raise ValueError(f"{profile.model_id}: Expected a state dict or checkpoint dictionary")
    extra = {}
    if any(k in ckpt for k in ("model_state_dict", "state_dict", "model")):
        state = next((ckpt[k] for k in ("model_state_dict", "state_dict", "model")
                      if isinstance(ckpt.get(k), dict)), None)
        if state is None:
            raise ValueError(f"{profile.model_id}: Checkpoint contains no supported state dict")
        extra = {k: v for k, v in ckpt.items() if isinstance(v, (int, float, str, bool, type(None)))}
    else:
        state = ckpt
    if not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                            for k, v in state.items()):
        raise ValueError(f"{profile.model_id}: Model parameters are not a valid tensor state dict")
    if profile.model_id == "birads":
        if tuple(str(x) for x in ckpt.get("class_values", ())) != profile.classes:
            raise ValueError("BI-RADS checkpoint class mapping differs from (2,3,4,5)")
        if ckpt.get("architecture") != "torchvision.efficientnet_b0":
            raise ValueError("BI-RADS checkpoint architecture differs from EfficientNet-B0")
        if ckpt.get("image_size", 224) != 224:
            raise ValueError("BI-RADS checkpoint expects a different input image size")
    else:
        for key in ("class_names", "classes"):
            if key in ckpt and isinstance(ckpt[key], (list, tuple)):
                stored = tuple(str(x) for x in ckpt[key])
                if stored != profile.classes:
                    raise ValueError(f"{profile.model_id} {key} mismatch: {stored} != {profile.classes}")
    return state, extra


def load_classifier(profile: Profile, device: torch.device) -> tuple[torch.nn.Module, dict]:
    path = ROOT / "models" / profile.checkpoint
    if not path.is_file():
        raise FileNotFoundError(f"{profile.model_id}: Missing checkpoint {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state, checkpoint_metadata = checkpoint_state(ckpt, profile)
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(profile.classes))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, checkpoint_metadata


def preprocess(path: Path, profile: Profile) -> tuple[np.ndarray, torch.Tensor]:
    with Image.open(path) as file:
        rgb = file.convert("RGB")
        if profile.preprocessing == "pad":
            transformed = ImageOps.pad(rgb, (profile.input_size, profile.input_size),
                                       method=Image.Resampling.BILINEAR,
                                       color=(0, 0, 0), centering=(0.5, 0.5))
        else:
            transformed = rgb.resize((profile.input_size, profile.input_size),
                                     Image.Resampling.BILINEAR)
    display_image = np.asarray(transformed, dtype=np.uint8)
    tensor = TF.normalize(TF.to_tensor(transformed), MEAN, STD).unsqueeze(0)
    return display_image, tensor


def gradcam(model: torch.nn.Module, input_batch: torch.Tensor,
            target_index: int, feature_index: int = 5) -> tuple[np.ndarray, np.ndarray]:
    captured: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, activation: torch.Tensor):
        captured["activation"] = activation.detach()
        activation.register_hook(lambda g: captured.__setitem__("gradient", g.detach()))

    hook = model.features[feature_index].register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(input_batch)
            probs = logits.softmax(dim=1).detach().cpu().numpy()[0]
            logits[0, target_index].backward()
        if "activation" not in captured or "gradient" not in captured:
            raise RuntimeError("No activation/gradient captured; refusing to fabricate an explanation")
        a, g = captured["activation"], captured["gradient"]
        weights = g.mean(dim=(2, 3), keepdim=True)
        heat = F.relu((a * weights).sum(dim=1, keepdim=True))
        heat = F.interpolate(heat, size=tuple(input_batch.shape[-2:]),
                             mode="bilinear", align_corners=False)
        result = heat[0, 0].detach().cpu().float().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite Grad-CAM values; no image saved")
        lo, hi = float(result.min()), float(result.max())
        if hi - lo < 1e-12:
            raise RuntimeError("Constant/zero Grad-CAM; no synthetic fallback")
        return np.clip((result - lo) / (hi - lo), 0, 1), probs
    finally:
        hook.remove()


def target_index_from_arg(text: str | None, profile: Profile, default_idx: int) -> int:
    if text is None:
        return default_idx
    for i, label in enumerate(profile.classes):
        if text.strip().lower() == label.lower():
            return i
    raise ValueError(f"Invalid --target {text!r} for {profile.model_id}; choose {list(profile.classes)}")


def save_outputs(folder: Path, image_path: Path, src: dict, profile: Profile,
                 input_image: np.ndarray, heat: np.ndarray, probs: np.ndarray,
                 target_idx: int, ckpt_extra: dict):
    folder.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", image_path.stem)
    target = profile.classes[target_idx]
    prefix = f"{safe_stem}_{profile.model_id}_target_{target}"
    base = folder / prefix
    version = 1
    while any((folder / (base.name + suffix)).exists()
              for suffix in (".png", "_cam.npy", "_metadata.json")):
        version += 1
        base = folder / f"{prefix}_run{version}"

    image_float = input_image.astype(np.float32) / 255
    colormap = plt.get_cmap("jet")(heat)[..., :3]
    blended = np.clip(image_float * 0.65 + colormap * 0.35, 0, 1)
    predicted_idx = int(np.argmax(probs))
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4))
    axes[0].imshow(input_image); axes[0].set_title("Model input")
    axes[1].imshow(heat, cmap="jet", vmin=0, vmax=1); axes[1].set_title(f"Grad-CAM: {target}")
    axes[2].imshow(blended); axes[2].set_title("Attribution overlay")
    for ax in axes: ax.axis("off")
    fig.suptitle(f"{profile.model_id.upper()} | Prediction: {profile.classes[predicted_idx]} "
                 f"({float(probs[predicted_idx]):.1%}) | Target: {target}")
    fig.tight_layout()
    png = base.with_suffix(".png")
    npy = base.parent / (base.name + "_cam.npy")
    metadata_path = base.parent / (base.name + "_metadata.json")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    np.save(npy, heat.astype(np.float32))
    metadata = {
        "research_only": True,
        "gradcam_is_not_lesion_segmentation": True,
        "synthetic_fallback_used": False,
        "algorithm": "positive gradient-weighted class activation mapping",
        "feature_layer": "efficientnet_b0.features[5]",
        "image_path": str(image_path),
        "dataset_context": src,
        "model": profile.model_id,
        "checkpoint": str(ROOT / "models" / profile.checkpoint),
        "checkpoint_metadata": ckpt_extra,
        "class_order": list(profile.classes),
        "input_size": profile.input_size,
        "preprocessing": profile.preprocessing,
        "target_class": target,
        "predicted_class": profile.classes[predicted_idx],
        "softmax_scores_not_calibrated": {label: float(probs[i]) for i, label in enumerate(profile.classes)},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved:", png)
    print("Saved:", npy)
    print("Saved:", metadata_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", help="Image path, absolute or relative to repo root")
    source.add_argument("--bra-val-index", type=int, help="Zero-based index in bra_val.csv")
    parser.add_argument("--model", choices=["all", *PROFILES], default="all")
    parser.add_argument("--target", help="Explicit target CLASS LABEL, e.g. Benign or 4 (single model only)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.run and not (args.image or args.bra_val_index is not None):
        parser.error("--run requires --image or --bra-val-index")
    if args.model == "all" and args.target is not None:
        parser.error("--target requires selecting one model (not --model all)")
    image_path, src = source_image(args)
    profiles = list(PROFILES.values()) if args.model == "all" else [PROFILES[args.model]]
    print("Repository:", ROOT)
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
    if image_path:
        print("Input:", image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for profile in profiles:
        print(f"\n[{profile.model_id}] checkpoint={profile.checkpoint}; classes={profile.classes}; "
              f"input={profile.input_size}/{profile.preprocessing}")
        model, ckpt_extra = load_classifier(profile, device)
        if image_path:
            display, batch = preprocess(image_path, profile)
            batch = batch.to(device)
            with torch.no_grad():
                probs = model(batch).softmax(dim=1)[0].detach().cpu().numpy()
            pred_idx = int(np.argmax(probs))
            target_idx = target_index_from_arg(args.target, profile, pred_idx)
            print(f"  Prediction: {profile.classes[pred_idx]}, target: {profile.classes[target_idx]}")
            print("  Softmax scores:", {str(l): round(float(v), 4) for l, v in zip(profile.classes, probs)})
            if args.run:
                heat, attr_probs = gradcam(model, batch, target_idx)
                if not np.allclose(attr_probs, probs, atol=1e-5, rtol=1e-4):
                    raise RuntimeError(f"{profile.model_id}: Prediction changed between no-grad and Grad-CAM passes")
                out = args.output_dir
                out = out if out.is_absolute() else ROOT / out
                save_outputs(out, image_path, src, profile, display, heat, attr_probs, target_idx, ckpt_extra)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.audit_only:
        print("\nPASS: checkpoints loaded strictly and class mappings matched; no output files generated.")
    else:
        print("\nGrad-CAM outputs are model attributions only, not lesion masks or clinical explanations.")


if __name__ == "__main__":
    main()
