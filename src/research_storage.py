"""Research storage/review semantics; no connection to the production database."""
from __future__ import annotations
import os
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .research_inference import ABSTENTION, validate_research_response

REVIEW_KEYS = {"human_review", "review_history"}


def database_path(root: Path, environ=None) -> Path:
    env = os.environ if environ is None else environ
    production = (root / "lumina_database.db").resolve()
    if env.get("LUMINA_TEST_MODE") not in {None,"0","1"}:
        raise RuntimeError("LUMINA_TEST_MODE must be explicitly 0 or 1; refusing an ambiguous database mode")
    test = env.get("LUMINA_TEST_MODE") == "1"
    override = env.get("LUMINA_TEST_DB_PATH")
    if not test:
        if override is not None:
            raise RuntimeError("Test database override requires LUMINA_TEST_MODE=1")
        return production
    if not override or not Path(override).is_absolute():
        raise RuntimeError("Test mode requires an explicit absolute LUMINA_TEST_DB_PATH; no production fallback")
    path = Path(override).resolve()
    allowed = [(root / f"data/processed/lumina_integration_stage{stage}/databases").resolve() for stage in (2, 3)]
    allowed.append((root / "data/processed/lumina_ui_verdict_upgrade/databases").resolve())
    allowed.append((root / "data/processed/busi_runtime_diagnosis_20261026/databases").resolve())
    if not any(path.is_relative_to(folder) for folder in allowed) or path.suffix not in {".db", ".sqlite3"} or not path.parent.is_dir():
        raise RuntimeError("Test database must be in a dedicated existing integration databases directory")
    if path == production or (path.exists() and path.samefile(production)):
        raise RuntimeError("Test database cannot alias the production database")
    return path


class PatientUpsert(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, allow_inf_nan=False)
    id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=150)
    age: str = Field(min_length=1, max_length=20)
    date: str = Field(min_length=1, max_length=60)
    prediction: str
    confidence: float
    size_cm: float
    stage: str
    report_details: dict = Field(default_factory=dict)
    original_image: str = ""
    heatmap_image: str = ""
    multi_model_result: dict = Field(default_factory=dict)


class FeedbackSubmission(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, allow_inf_nan=False)
    case_id: str = Field(min_length=1, max_length=100)
    corrected_prediction: Literal["Normal", "Benign", "Malignant"] | None = None
    corrected_confidence: float | None = Field(default=None, ge=0, le=1)
    reasoning: str = Field(min_length=1, max_length=2000)
    feedback_type: Literal["reviewed", "independent_assessment", "uncertain", "pathology_confirmed", "clinically_overridden"] = "independent_assessment"
    doctor_id: str = Field(default="anonymous", min_length=1, max_length=100)

    @model_validator(mode="after")
    def assessment_required(self):
        if self.feedback_type == "reviewed" and self.corrected_prediction is not None:
            raise ValueError("Review-only submissions cannot include an assessment")
        if self.feedback_type in {"independent_assessment", "pathology_confirmed", "clinically_overridden"} and self.corrected_prediction is None:
            raise ValueError("An independent assessment requires a selected pathology label")
        return self


def frozen_analysis(raw: dict) -> dict:
    return {k:v for k,v in raw.items() if k not in REVIEW_KEYS}


def validate_new_analysis(payload: PatientUpsert):
    raw = payload.multi_model_result
    if raw.get("contract_version") is not None:
        if any(k in raw for k in REVIEW_KEYS):
            raise ValueError("Submit human review through /feedback/submit after saving the original analysis")
        validate_research_response(raw)
        if payload.prediction != ABSTENTION or payload.confidence != 0 or payload.size_cm != 0 or payload.stage != "N/A":
            raise ValueError("Versioned research storage requires abstention, a zero compatibility placeholder and unassessed size/stage")
        if payload.report_details != raw["report_details"] or payload.heatmap_image != raw["heatmap_base64"]:
            raise ValueError("Saved report/attribution must match the original inference response")


def workflow_statistics(patients: list[dict], feedback: list[dict]) -> dict:
    versioned = [p for p in patients if p.get("record_kind")=="versioned_research"]
    reviewed = [p for p in patients if isinstance(p["multi_model_result"].get("human_review"),dict) and p["multi_model_result"]["human_review"].get("status")=="recorded"]
    histories = {r["feedback_id"]:r for p in patients for r in (p["multi_model_result"].get("review_history") or []) if isinstance(r,dict) and r.get("feedback_id")}
    by_id = {p["id"]:p for p in patients}
    breakdown = Counter((r["original_prediction"],r["corrected_prediction"]) for r in feedback)
    unavailable = Counter()
    for p in versioned:
        for name,m in p["multi_model_result"].get("models",{}).items():
            if m.get("status") != "ok":unavailable[name] += 1
    recent=[]
    for f in feedback[:50]:
        review=histories.get(f["id"],{})
        p=by_id.get(f["case_id"],{})
        recent.append(dict(id=f["id"],case_id=f["case_id"],original_prediction=f["original_prediction"],
            human_assessment=None if f["corrected_prediction"]=="Not assessed" else f["corrected_prediction"],
            declared_confidence=review.get("declared_confidence"),reasoning=f["reasoning"],reviewer_id=f["doctor_id"],
            identity_verified=False,feedback_type=f["feedback_type"],created_at=f["created_at"],
            record_kind=p.get("record_kind","historical")))
    return dict(semantics_version="research-workflow-2.0",total_feedback=len(feedback),
        accuracy_percentage=None,total_predictions_reviewed=0,correct_predictions=None,recent_errors=[],
        diagnostic_accuracy_available=False,avg_model_confidence=None,
        note="Workflow counts only. Abstentions are not diagnostic errors; zero storage placeholders are not probabilities. Human identities and reference assessments are unverified.",
        counts=dict(total_records=len(patients),versioned_records=len(versioned),historical_records=len(patients)-len(versioned),
            abstentions=sum(p["multi_model_result"].get("consensus",{}).get("status")=="abstained" for p in versioned),
            reviewed_records=len(reviewed),unreviewed_records=len(patients)-len(reviewed),
            independent_assessments=sum(f["feedback_type"] in {"independent_assessment","pathology_confirmed","clinically_overridden"} for f in feedback),
            versioned_feedback=sum(by_id.get(f["case_id"],{}).get("record_kind")=="versioned_research" for f in feedback),
            unavailable_model_observations=dict(unavailable)),
        breakdown=[dict(model_said=a,actually=b,count=n,avg_model_confidence=None,avg_actual_confidence=None) for (a,b),n in sorted(breakdown.items())],
        recent_feedback=recent)
