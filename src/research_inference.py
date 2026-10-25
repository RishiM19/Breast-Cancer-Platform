"""Versioned, database-free research inference; frozen loaders/transforms are reused.

No ensemble has been validated. Individual predictions are retained, while the
consensus policy fails closed regardless of agreement, score, or claimed domain.
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .lumina_gradcam_all_classifiers import (
    ROOT, PROFILES, MEAN, STD, load_classifier, preprocess, gradcam,
)
from .research_segmentation import SegmentationEngine, SegmentationResult
from .research_decision_engine import ResearchVerdict, compute_research_verdict, decision_policy

CONTRACT_VERSION = "1.0.0"
POLICY_VERSION = "1.0.0"
ABSTENTION = "No validated consensus available"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
Domain = Literal["unknown", "BUSI", "UCLM", "BUS-BRA"]
ModelID = Literal["busi", "uclm", "bra", "birads"]
Task = Literal["pathology", "birads"]
Status = Literal["ok", "missing", "not_run", "error"]
LOGGER = logging.getLogger(__name__)
LIMITATIONS = (
    "Research use only; no model or ensemble has established clinical deployment readiness.",
    "Softmax scores are uncalibrated model scores, not clinical probabilities.",
    "Grad-CAM is attribution, not lesion segmentation, medical reasoning or a validated explanation.",
    "Upload source domain is caller-declared, not independently verified; unknown uploads remain unknown.",
    "Cross-dataset predictions do not establish generalization. Pathology and BI-RADS are separate tasks.",
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    task: str
    source_domain: str
    expected_sha256: str
    consensus_exclusion: str


# Ordered IDs, class indices, loader and preprocessing remain those verified in
# Phase 2. Hashes pin these exact frozen checkpoint versions, not just filenames.
SPECS = MappingProxyType({
    "busi": ModelSpec("busi", "BUSI", "pathology", "BUSI",
        "8e59e91c25cc6e591ae042d98951bda05b5c2fd31a49c78f4ce47b0d34d060e5",
        "Historical baseline; permanently excluded by the current reliability policy."),
    "uclm": ModelSpec("uclm", "UCLM", "pathology", "UCLM",
        "bf4b912c65fad8c78d0c99ba541502cabdca9b050f4d82e9f383803785e6fae4",
        "No validated ensemble policy; cross-dataset uploads are not supported for voting."),
    "bra": ModelSpec("bra", "BUS-BRA", "pathology", "BUS-BRA",
        "16836657cccbc40132932d6d7c380539685e0fc2e5d5ec8e15ec8b11af6cdffa",
        "Individual binary research classifier; no validated ensemble policy."),
    "birads": ModelSpec("birads", "BUS-BRA BI-RADS", "birads", "BUS-BRA",
        "c6a1e738a497ce0ca155e2b1c58c9581358d36d1369d6f4d687e9ff33ae0d491",
        "Separate BI-RADS task; never a pathology vote. No validated ensemble policy."),
})


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Checkpoint(ContractModel):
    filename: str
    version: str
    expected_sha256: str
    sha256: str | None = None
    verification: Literal["verified", "missing", "mismatch", "not_checked"] = "not_checked"


class Eligibility(ContractModel):
    requested_domain: Domain
    domain_relation: Literal["native_research", "cross_dataset", "unknown"]
    research_task_supported: bool
    consensus_eligible: Literal[False] = False
    reason: str


class ModelResult(ContractModel):
    contract_version: Literal["1.0.0"] = CONTRACT_VERSION
    model_id: ModelID
    checkpoint: Checkpoint
    task: Task
    ordered_class_labels: list[str]
    preprocessing: dict
    source_training_domain: str
    research_only: Literal[True] = True
    status: Status
    prediction: str | None = None
    predicted_class_index: int | None = None
    score: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)  # legacy alias, same label/score
    score_semantics: Literal["uncalibrated_softmax"] = "uncalibrated_softmax"
    probabilities: dict[str, float | None] = Field(default_factory=dict)
    logits: list[float] | None = None
    eligibility: Eligibility
    note: str
    error_code: str | None = None

    @model_validator(mode="after")
    def validate_prediction(self):
        profile = PROFILES[self.model_id]
        if self.ordered_class_labels != list(profile.classes) or self.task != SPECS[self.model_id].task:
            raise ValueError("Model task/class mapping differs from frozen configuration")
        if self.status == "ok":
            if self.checkpoint.verification != "verified" or self.checkpoint.sha256 != SPECS[self.model_id].expected_sha256:
                raise ValueError("Successful prediction requires the pinned checkpoint")
            if self.predicted_class_index is None or not 0 <= self.predicted_class_index < len(profile.classes):
                raise ValueError("Missing/invalid predicted index")
            if self.prediction != profile.classes[self.predicted_class_index]:
                raise ValueError("Prediction/index mismatch")
            values = [self.probabilities.get(c) for c in profile.classes]
            if any(v is None or not 0 <= v <= 1 for v in values) or abs(sum(values) - 1) > 1e-5:
                raise ValueError("Invalid per-class scores")
            if self.score != values[self.predicted_class_index] or self.confidence != self.score:
                raise ValueError("Score must belong to this model's predicted label")
            if self.predicted_class_index != int(np.argmax(values)):
                raise ValueError("Prediction must match the score maximum")
            if self.logits is None or len(self.logits) != len(values) or not np.isfinite(self.logits).all():
                raise ValueError("Missing/nonfinite logits")
        elif any(v is not None for v in (self.prediction, self.predicted_class_index, self.score, self.confidence, self.logits)) or self.probabilities:
            raise ValueError("Unsuccessful models cannot carry fabricated predictions/scores")
        return self


class ConsensusResult(ContractModel):
    policy_version: Literal["1.0.0"] = POLICY_VERSION
    task: Literal["pathology"] = "pathology"
    status: Literal["abstained"] = "abstained"
    prediction: Literal["No validated consensus available"] = ABSTENTION
    confidence: None = None
    score: None = None
    score_semantics: Literal["unavailable_abstention"] = "unavailable_abstention"
    agreement_count: Literal[0] = 0
    model_count: Literal[0] = 0
    participating_models: Literal[0] = 0
    agreement_ratio: None = None
    eligible_model_ids: list[str] = Field(default_factory=list, max_length=0)
    excluded_models: dict[str, str]
    reason_code: Literal["no_validated_consensus_policy"] = "no_validated_consensus_policy"
    note: str = "Agreement, disagreement and ties all abstain. No cross-task/domain majority or borrowed model score."


class Attribution(ContractModel):
    status: Literal["ok", "not_run", "error"]
    method: Literal["Grad-CAM"] = "Grad-CAM"
    model_id: Literal["bra"] = "bra"
    target_index: int | None = None
    target_label: str | None = None
    note: str = "Genuine BUS-BRA pathology attribution only; no clinical explanation or lesion measurement."
    error_code: str | None = None


class ImageInfo(ContractModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ResearchResponse(ContractModel):
    contract_version: Literal["1.0.0"] = CONTRACT_VERSION
    success: bool
    status: Literal["ok", "partial", "unavailable"]
    research_only: Literal[True] = True
    source_domain: Domain
    source_domain_verification: Literal["caller_declared_unverified"] = "caller_declared_unverified"
    consensus: ConsensusResult
    models: dict[str, ModelResult]
    heatmap_base64: str = ""
    attribution: Attribution
    image: ImageInfo
    research_limitations: list[str] = Field(default_factory=lambda: list(LIMITATIONS))
    report_details: dict[str, str]

    @model_validator(mode="after")
    def verify_models(self):
        if set(self.models) != {s.display_name for s in SPECS.values()}:
            raise ValueError("Exactly four model records are required, including unavailable statuses")
        for s in SPECS.values():
            if self.models[s.display_name].model_id != s.model_id:
                raise ValueError("Display/model identity mismatch")
        return self


class ResearchResponseV11(ResearchResponse):
    contract_version: Literal["1.1.0"] = "1.1.0"
    segmentation: SegmentationResult

    @model_validator(mode="after")
    def segmentation_matches_image(self):
        s = self.segmentation
        if (s.image_width, s.image_height) != (self.image.width, self.image.height) or s.requested_domain != self.source_domain:
            raise ValueError("Segmentation must match response image and declared domain")
        return self


class ResearchResponseV12(ResearchResponseV11):
    contract_version: Literal["1.2.0"] = "1.2.0"
    research_verdict: ResearchVerdict

    @model_validator(mode="after")
    def verdict_matches_frozen_models(self):
        if self.research_verdict != compute_research_verdict(self.models,self.source_domain):
            raise ValueError("Research verdict must match deterministic primary selection and original model scores")
        return self


def validate_research_response(raw: dict):
    version = raw.get("contract_version")
    if version == "1.0.0":
        return ResearchResponse.model_validate(raw)
    if version == "1.1.0":
        return ResearchResponseV11.model_validate(raw)
    if version == "1.2.0":
        return ResearchResponseV12.model_validate(raw)
    raise ValueError("Unsupported research inference contract")


def preprocessing_config(model_id: str) -> dict:
    p = PROFILES[model_id]
    return dict(color="RGB", dtype="float32", pixel_scaling="uint8 / 255 exactly once",
                input_size=[p.input_size, p.input_size], geometry=p.preprocessing,
                interpolation="PIL bilinear", pad_color=[0, 0, 0] if p.preprocessing == "pad" else None,
                pad_centering=[.5, .5] if p.preprocessing == "pad" else None,
                mean=list(MEAN), std=list(STD))


def eligibility(model_id: str, domain: Domain) -> Eligibility:
    s = SPECS[model_id]
    relation = "unknown" if domain == "unknown" else "native_research" if domain == s.source_domain else "cross_dataset"
    return Eligibility(requested_domain=domain, domain_relation=relation,
                       research_task_supported=relation == "native_research" and model_id != "busi",
                       reason=s.consensus_exclusion)


def resolve_consensus(results: dict[str, ModelResult], domain: Domain) -> ConsensusResult:
    # Policy is recomputed from immutable specifications, never trusted from an
    # uploaded/client-supplied consensus_eligible flag or high softmax score.
    excluded = {}
    for model_id, s in SPECS.items():
        r = results.get(s.display_name)
        state = r.status if r is not None else "missing"
        excluded[model_id] = f"{eligibility(model_id, domain).reason} Status: {state}."
    return ConsensusResult(excluded_models=excluded)


def contract_descriptor(version="1.0.0") -> dict:
    if version not in {"1.0.0", "1.1.0", "1.2.0"}:
        raise ValueError("Unsupported research inference contract")
    result = dict(contract_version=version, policy_version=POLICY_VERSION,
        runtime=dict(dtype="float32",autocast=False,tf32=False,deterministic_algorithms=True,
                     cudnn_benchmark=False,requests="serialized",gradcam_feature_index=5),
        default_source_domain="unknown", domains=["unknown", "BUSI", "UCLM", "BUS-BRA"],
        source_domain_verification="caller_declared_unverified", research_only=True,
        models={s.display_name: dict(model_id=k, checkpoint=PROFILES[k].checkpoint,
            checkpoint_version=f"sha256:{s.expected_sha256}", expected_sha256=s.expected_sha256,
            task=s.task, ordered_class_labels=list(PROFILES[k].classes), preprocessing=preprocessing_config(k),
            source_training_domain=s.source_domain, consensus_eligible=False, exclusion_reason=s.consensus_exclusion,
            eligibility_by_domain={d:eligibility(k,d).model_dump() for d in ("unknown","BUSI","UCLM","BUS-BRA")})
            for k,s in SPECS.items()},
        statuses=["ok","missing","not_run","error"], consensus=resolve_consensus({}, "unknown").model_dump(),
        limitations=list(LIMITATIONS), response_schema=ResearchResponse.model_json_schema())
    if version in {"1.1.0", "1.2.0"}:
        result["response_schema"] = ResearchResponseV11.model_json_schema()
        result["segmentation"] = dict(schema=SegmentationResult.model_json_schema(), eligible_domains=["UCLM"],
            provenance="caller_declared_unverified", runtime="CUDA float16 autocast; CPU float32; serialized with classifiers and Grad-CAM",
            classifier_contract_version="1.0.0", consensus_participant=False)
    if version == "1.2.0":
        result["response_schema"] = ResearchResponseV12.model_json_schema()
        result["research_decision_policy"] = decision_policy()
    return result


class InvalidImage(ValueError):
    pass


def decode_image(raw: bytes) -> Image.Image:
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise InvalidImage("Image is empty or exceeds the 10 MiB upload limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as im:
                if im.format not in {"PNG", "JPEG", "BMP", "TIFF"}:
                    raise InvalidImage("Supported single-frame image formats: PNG, JPEG, BMP, TIFF")
                if im.width * im.height > MAX_IMAGE_PIXELS or getattr(im, "n_frames", 1) != 1:
                    raise InvalidImage("Image exceeds the pixel limit or contains multiple frames")
                im.load()
                return im.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidImage("Invalid or unsupported image") from exc


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ResearchEngine:
    def __init__(self, device: torch.device | None = None, model_root: Path | None = None):
        # Production uses the same precision/determinism settings as verification.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_root = model_root or ROOT / "models"
        self._models = {}
        self._identities = {}
        self._lock = threading.RLock()  # serializes inference and temporary Grad-CAM hooks
        self.segmentation = SegmentationEngine(self.device, self.model_root / "uclm_unet_v3_binary_best.pth")

    def _checkpoint(self, model_id: str) -> Checkpoint:
        p = PROFILES[model_id];s = SPECS[model_id];path = self.model_root / p.checkpoint
        record = Checkpoint(filename=p.checkpoint, version=f"sha256:{s.expected_sha256}", expected_sha256=s.expected_sha256)
        if not path.is_file():
            record.verification = "missing"
            self._models.pop(model_id, None)
            self._identities.pop(model_id, None)
            return record
        stat = path.stat();identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self._identities.get(model_id) != identity:
            self._models.pop(model_id, None)
            actual = sha256(path)
            if actual != s.expected_sha256:
                record.sha256 = actual;record.verification = "mismatch"
                return record
            self._identities[model_id] = identity
        record.sha256 = s.expected_sha256;record.verification = "verified"
        return record

    def _load(self, model_id: str):
        if model_id not in self._models:
            # The canonical loader reads ROOT/models; alternate roots are only
            # for missing/corrupt-checkpoint tests and cannot load other weights.
            if self.model_root.resolve() != (ROOT / "models").resolve():
                raise RuntimeError("Alternate checkpoint roots cannot perform inference")
            self._models[model_id], _ = load_classifier(PROFILES[model_id], self.device)
        return self._models[model_id]

    def analyze(self, raw: bytes, source_domain: Domain = "unknown", run_models=None, attribution=True, contract_version="1.0.0") -> ResearchResponse:
        if contract_version not in {"1.0.0", "1.1.0", "1.2.0"}:
            raise ValueError("Unsupported research inference contract")
        if source_domain not in ("unknown", "BUSI", "UCLM", "BUS-BRA"):
            raise ValueError("Unsupported source domain")
        image = decode_image(raw)
        requested = set(SPECS if run_models is None else run_models)
        if not requested <= set(SPECS):
            raise ValueError("Unsupported model ID")
        results = {};tensors = {}
        with self._lock:
            for k,s in SPECS.items():
                p = PROFILES[k]
                record = ModelResult(model_id=k, checkpoint=Checkpoint(filename=p.checkpoint,
                    version=f"sha256:{s.expected_sha256}", expected_sha256=s.expected_sha256), task=s.task,
                    ordered_class_labels=list(p.classes), preprocessing=preprocessing_config(k), source_training_domain=s.source_domain,
                    status="not_run", eligibility=eligibility(k,source_domain), note=s.consensus_exclusion)
                if k in requested:
                    try:
                        ck = self._checkpoint(k);record.checkpoint = ck
                        if ck.verification == "missing":
                            record.status="missing";record.error_code="checkpoint_missing"
                        elif ck.verification != "verified":
                            record.status="error";record.error_code="checkpoint_hash_mismatch"
                        else:
                            _,tensor = preprocess(io.BytesIO(raw),p)  # exact frozen Phase 2 transform
                            tensors[k] = tensor.to(self.device)
                            # Construct/load parameters outside inference_mode:
                            # Grad-CAM later needs ordinary autograd-capable tensors.
                            model = self._load(k)
                            with torch.inference_mode():
                                logits = model(tensors[k])[0].float().cpu()
                                if logits.shape != (len(p.classes),) or not torch.isfinite(logits).all():
                                    raise ValueError("Invalid model output")
                                scores = torch.softmax(logits,dim=0)
                            index = int(scores.argmax());values={c:float(scores[i]) for i,c in enumerate(p.classes)}
                            if k == "bra":values["Normal"] = None  # old client compatibility; never a third BRA class
                            record = ModelResult(**(record.model_dump() | dict(status="ok", prediction=p.classes[index],
                                predicted_class_index=index, score=float(scores[index]), confidence=float(scores[index]),
                                probabilities=values, logits=logits.tolist())))
                    except Exception as exc:
                        LOGGER.warning("Research inference unavailable for %s: %s", k, exc)
                        record = ModelResult(**(record.model_dump() | dict(status="error", error_code="inference_error")))
                results[s.display_name] = record
            heatmap = "";cam_record = Attribution(status="not_run")
            bra = results["BUS-BRA"]
            if attribution and bra.status == "ok":
                try:
                    heat, _ = gradcam(self._load("bra"), tensors["bra"], bra.predicted_class_index, feature_index=5)
                    heat_image=Image.fromarray(heat.astype(np.float32),mode="F").resize(image.size,Image.Resampling.BILINEAR)
                    h=np.asarray(heat_image).clip(0,1)
                    colors=np.stack([h,1-np.abs(2*h-1),1-h],axis=-1)*255
                    blended=(.65*np.asarray(image,dtype=np.float32)+.35*colors).clip(0,255).astype(np.uint8)
                    buf=io.BytesIO();Image.fromarray(blended).save(buf,format="JPEG",quality=90)
                    heatmap="data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode("ascii")
                    cam_record=Attribution(status="ok",target_index=bra.predicted_class_index,target_label=bra.prediction)
                except Exception as exc:
                    LOGGER.warning("Grad-CAM unavailable for BRA: %s", exc)
                    cam_record=Attribution(status="error",target_index=bra.predicted_class_index,
                                           target_label=bra.prediction,error_code="attribution_unavailable")
            segmentation = self.segmentation.analyze(image, source_domain) if contract_version in {"1.1.0", "1.2.0"} else None
        ok=sum(r.status=="ok" for r in results.values())
        total=4
        if segmentation is not None and source_domain == "UCLM":
            total+=1;ok+=int(segmentation.status=="ok")
        birads=results["BUS-BRA BI-RADS"]
        response_type = ResearchResponseV11 if segmentation is not None else ResearchResponse
        extension = dict(segmentation=segmentation) if segmentation is not None else {}
        if contract_version == "1.2.0":
            response_type=ResearchResponseV12
            extension["research_verdict"]=compute_research_verdict(results,source_domain)
        return response_type(**extension,success=ok>0,status="ok" if ok==total else "partial" if ok else "unavailable",
            source_domain=source_domain,models=results,consensus=resolve_consensus(results,source_domain),
            heatmap_base64=heatmap,attribution=cam_record,image=ImageInfo(width=image.width,height=image.height),
            report_details=dict(Indication="Not assessed by these research classifiers.",Composition="Not assessed.",
                Findings="Individual research outputs only. No lesion size, grade, stage or tissue morphology was measured.",
                Impression=ABSTENTION,
                BI_RADS=f"Research model category {birads.prediction}; not a clinical assessment." if birads.status=="ok" else "BI-RADS model unavailable."))
