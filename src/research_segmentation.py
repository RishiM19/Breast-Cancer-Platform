"""Pinned UCLM V3 foreground inference. No dataset or annotation access at runtime."""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import threading
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .train_uclm_segmentation_v2 import UNet
from .uclm_segmentation_dataset_v2 import MEAN, STD, normalized_rgb_tensor

FILENAME = "uclm_unet_v3_binary_best.pth"
SHA256 = "7acf7f5331e92389537ac85aac0c91ed848ece7455a4b688d49ce4da4d6ea957"
VERSION = "UCLM_V3_BINARY_EXPLORATORY"
PREPROCESSING = dict(color="RGB", input_size=[256, 256], dtype="float32",
    pixel_scaling="uint8 / 255 exactly once", mean=MEAN.tolist(), std=STD.tolist(),
    geometry="stretch; no padding, crop, rotation, mirroring or EXIF transpose",
    image_interpolation="PIL bilinear", mask_restoration="threshold at 256x256 then PIL nearest to original width/height",
    probability_coordinates="model input 256x256; not original image coordinates",
    output_activation="two-class softmax (float32); foreground channel 1",
    decision="two-logit argmax; foreground iff logit[1] > logit[0]; exact ties background",
    threshold_note="equivalent to foreground probability > 0.5 in exact arithmetic; argmax avoids softmax rounding ties",
    runtime="CUDA float16 autocast as original evaluator; CPU float32; eval/inference_mode")
LIMITATIONS = [
    "Exploratory UCLM native-domain research only. Source domain is caller-declared and unverified.",
    "Foreground combines lesion annotations; it does not classify benign/malignant or assign BI-RADS.",
    "Stretch and reverse resize may lose detail; original pixel alignment is a coordinate convention, not perfect spatial fidelity.",
    "Pixel softmax scores are uncalibrated. No clinically validated localization, measurements, diagnosis or treatment advice.",
    "An empty predicted mask is a model output, not evidence that an image is normal.",
    "Non-default EXIF orientation is unsupported for segmentation; classifier outputs remain separate.",
]


class SegmentationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model_id: Literal["uclm_v3_binary"] = "uclm_v3_binary"
    task: Literal["segmentation"] = "segmentation"
    model_version: Literal["UCLM_V3_BINARY_EXPLORATORY"] = VERSION
    checkpoint_filename: Literal["uclm_unet_v3_binary_best.pth"] = FILENAME
    expected_checkpoint_sha256: Literal[SHA256] = SHA256
    checkpoint_sha256: str | None = None
    checkpoint_verification: Literal["verified", "missing", "mismatch", "not_checked"] = "not_checked"
    source_training_domain: Literal["UCLM"] = "UCLM"
    requested_domain: Literal["unknown", "BUSI", "UCLM", "BUS-BRA"]
    source_domain_verification: Literal["caller_declared_unverified"] = "caller_declared_unverified"
    domain_relation: Literal["native_research", "cross_dataset", "unknown"]
    research_only: Literal[True] = True
    status: Literal["ok", "missing", "not_run", "error"]
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    original_coordinate_system: Literal["decoded input pixels; x right, y down; origin top left; no EXIF transpose"] = "decoded input pixels; x right, y down; origin top left; no EXIF transpose"
    ordered_class_labels: list[str] = Field(default_factory=lambda: ["Background", "Lesion"])
    threshold: Literal[0.5] = 0.5
    preprocessing: dict = Field(default_factory=lambda: dict(PREPROCESSING))
    predicted_mask_encoding: Literal["PNG L uint8 0=background 255=lesion; original dimensions"] = "PNG L uint8 0=background 255=lesion; original dimensions"
    overlay_encoding: Literal["PNG RGBA; background alpha=0; lesion RGBA=(0,255,128,112); original dimensions"] = "PNG RGBA; background alpha=0; lesion RGBA=(0,255,128,112); original dimensions"
    probability_encoding: Literal["base64 NPY float32 (256,256); foreground softmax; allow_pickle=False"] = "base64 NPY float32 (256,256); foreground softmax; allow_pickle=False"
    predicted_mask: str | None = None
    overlay: str | None = None
    foreground_probability_npy_base64: str | None = None
    empty_mask: bool | None = None
    reason: str
    error_code: str | None = None
    limitations: list[str] = Field(default_factory=lambda: list(LIMITATIONS))

    @model_validator(mode="after")
    def coherent(self):
        relation = "unknown" if self.requested_domain == "unknown" else "native_research" if self.requested_domain == "UCLM" else "cross_dataset"
        if self.domain_relation != relation or self.ordered_class_labels != ["Background", "Lesion"] or self.preprocessing != PREPROCESSING:
            raise ValueError("Segmentation mapping/preprocessing/domain mismatch")
        artifacts = (self.predicted_mask, self.overlay, self.foreground_probability_npy_base64, self.empty_mask)
        if self.status == "ok":
            if self.requested_domain != "UCLM" or self.checkpoint_verification != "verified" or self.checkpoint_sha256 != SHA256 or self.error_code is not None:
                raise ValueError("Segmentation success requires eligible domain and pinned checkpoint")
            if any(v is None for v in artifacts):
                raise ValueError("Successful segmentation requires actual outputs")
            # Check saved artifacts too: malformed output cannot masquerade as a valid mask.
            mask = decode_png(self.predicted_mask, "L", (self.image_width, self.image_height))
            overlay = decode_png(self.overlay, "RGBA", (self.image_width, self.image_height))
            if not np.isin(mask, [0, 255]).all() or bool(not mask.any()) != self.empty_mask:
                raise ValueError("Invalid binary mask or empty status")
            expected = np.zeros((*mask.shape, 4), dtype=np.uint8);expected[mask != 0] = (0, 255, 128, 112)
            if not np.array_equal(overlay, expected):
                raise ValueError("Overlay does not match predicted foreground")
            buf = base64.b64decode(self.foreground_probability_npy_base64, validate=True)
            if len(buf) > 300000:
                raise ValueError("Probability map exceeds fixed shape bounds")
            p = np.load(io.BytesIO(buf), allow_pickle=False)
            if p.shape != (256, 256) or p.dtype != np.float32 or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
                raise ValueError("Invalid probability map")
        elif any(v is not None for v in artifacts):
            raise ValueError("Unavailable segmentation cannot carry fabricated outputs")
        if self.requested_domain != "UCLM" and self.status != "not_run":
            raise ValueError("Non-UCLM segmentation must not run")
        return self


def decode_png(value, mode, size):
    prefix = "data:image/png;base64,"
    if not value.startswith(prefix):
        raise ValueError("Segmentation requires embedded PNG")
    data = base64.b64decode(value[len(prefix):], validate=True)
    with Image.open(io.BytesIO(data)) as im:
        if im.format != "PNG" or im.size != size or im.mode != mode:
            raise ValueError("Invalid segmentation PNG geometry/encoding")
        return np.asarray(im).copy()


def png_url(image):
    buf = io.BytesIO();image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def preprocess_image(image):
    return normalized_rgb_tensor(image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)).unsqueeze(0)


def restore_mask(model_mask, size):
    mask = np.asarray(model_mask)
    if mask.shape != (256, 256) or not np.isin(mask, [0, 1]).all():
        raise ValueError("Expected binary 256x256 model mask")
    return Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)


def encoded_outputs(logits, size):
    if logits.shape != (1, 2, 256, 256) or not torch.isfinite(logits).all():
        raise ValueError("Invalid segmentation logits")
    probability = logits.float().softmax(1)[0, 1].cpu().numpy().astype(np.float32)
    # Match the original evaluator exactly, including class-0 tie handling.
    mask = restore_mask(logits.argmax(1)[0].cpu().numpy(), size)
    a = np.asarray(mask);rgba = np.zeros((*a.shape, 4), dtype=np.uint8);rgba[a != 0] = (0, 255, 128, 112)
    buf = io.BytesIO();np.save(buf, probability, allow_pickle=False)
    return dict(predicted_mask=png_url(mask), overlay=png_url(Image.fromarray(rgba)),
        foreground_probability_npy_base64=base64.b64encode(buf.getvalue()).decode("ascii"), empty_mask=not bool(a.any()))


class SegmentationEngine:
    def __init__(self, device=None, checkpoint_path=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = checkpoint_path or Path(__file__).resolve().parents[1] / "models" / FILENAME
        self._model = None;self._identity = None;self.load_count = 0
        self._lock = threading.RLock()

    def _load(self):
        ck = torch.load(self.path, map_location="cpu", weights_only=True)
        if (ck.get("model_version") != VERSION or ck.get("num_classes") != 2 or
            ck.get("class_names") != ["Background", "Lesion"] or ck.get("base_channels") != 32 or ck.get("image_size") != 256):
            raise ValueError("Checkpoint metadata incompatible with frozen evaluator")
        model = UNet(base=32, num_classes=2)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        self._model = model.to(self.device).eval();self.load_count += 1

    def analyze(self, image, source_domain):
        relation = "unknown" if source_domain == "unknown" else "native_research" if source_domain == "UCLM" else "cross_dataset"
        data = dict(requested_domain=source_domain, domain_relation=relation, image_width=image.width,
            image_height=image.height, status="not_run", reason="Segmentation runs only for caller-declared UCLM; provenance remains unverified.")
        if source_domain != "UCLM":
            return SegmentationResult(**data)
        with self._lock:
            try:
                # Browsers may honor EXIF orientation while the original PIL
                # evaluator does not transpose it. Refuse masks rather than
                # silently align them to a differently displayed original.
                if image.getexif().get(274, 1) != 1:
                    return SegmentationResult(**(data | dict(error_code="unsupported_image_orientation",
                        reason="Non-default EXIF orientation cannot be overlaid safely with the original evaluator coordinates; segmentation was not run.")))
                if not self.path.is_file():
                    self._model = None;self._identity = None
                    return SegmentationResult(**(data | dict(status="missing", checkpoint_verification="missing",
                        error_code="checkpoint_missing", reason="UCLM segmentation checkpoint unavailable.")))
                stat = self.path.stat();identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                if self._identity != identity:
                    self._model = None;self._identity = None
                    with self.path.open("rb") as f:
                        actual = hashlib.file_digest(f, "sha256").hexdigest()
                    data.update(checkpoint_sha256=actual)
                    if actual != SHA256:
                        return SegmentationResult(**(data | dict(status="error", checkpoint_verification="mismatch",
                            error_code="checkpoint_hash_mismatch", reason="UCLM segmentation checkpoint hash differs from pinned baseline.")))
                    self._identity = identity
                data.update(checkpoint_sha256=SHA256, checkpoint_verification="verified")
                if self._model is None:
                    self._load()  # Construct outside inference_mode, matching classifier safety.
                with torch.inference_mode():
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                        logits = self._model(preprocess_image(image).to(self.device))
                    outputs = encoded_outputs(logits, image.size)
                return SegmentationResult(**(data | outputs | dict(status="ok", reason="Genuine binary foreground output; exploratory UCLM research only.")))
            except Exception:
                logging.getLogger(__name__).exception("UCLM segmentation unavailable")
                return SegmentationResult(**(data | dict(status="error", error_code="segmentation_inference_error",
                    reason="Segmentation inference failed; no mask or overlay was generated.")))
