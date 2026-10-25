import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.research_api import create_research_router
from src.research_inference import ABSTENTION, validate_research_response
from src.research_storage import (database_path, PatientUpsert, FeedbackSubmission, frozen_analysis, validate_new_analysis, workflow_statistics)

app = FastAPI()
ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = database_path(ROOT_DIR)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ] + (["http://127.0.0.1:5177"] if os.environ.get("LUMINA_TEST_MODE")=="1" else []),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _get_db()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age TEXT NOT NULL,
                date TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                size_cm REAL NOT NULL,
                stage TEXT NOT NULL,
                report_json TEXT NOT NULL,
                original_image TEXT NOT NULL,
                heatmap_image TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_logs (
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                doctor_id TEXT DEFAULT 'anonymous',
                original_prediction TEXT NOT NULL,
                original_confidence REAL NOT NULL,
                corrected_prediction TEXT NOT NULL,
                corrected_confidence REAL NOT NULL,
                reasoning TEXT,
                feedback_type TEXT DEFAULT 'pathology_confirmed',
                created_at TEXT NOT NULL,
                FOREIGN KEY (case_id) REFERENCES patients(id)
            )
            """
        )
        existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(patients)").fetchall()}
        if "multi_model_json" not in existing_columns:
            conn.execute("ALTER TABLE patients ADD COLUMN multi_model_json TEXT NOT NULL DEFAULT '{}'" )
        conn.commit()
    finally:
        conn.close()


# Preserve database initialization for server startup, without import-time writes.
app.add_event_handler("startup", init_db)
app.include_router(create_research_router())


@app.exception_handler(sqlite3.Error)
async def storage_error(_request, _exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=503, content={"detail":"Research storage unavailable; no result was confirmed saved."})


def _decode_record(row) -> dict:
    result = dict(row)
    warnings = []
    for source,target in (("report_json","report_details"),("multi_model_json","multi_model_result")):
        try:value = json.loads(result.pop(source) or "{}")
        except (ValueError,TypeError):value = {};warnings.append(f"Unreadable historical {target}; original stored text retained")
        if not isinstance(value,dict):warnings.append(f"Unsupported historical {target} structure; original stored text retained")
        result[target] = value if isinstance(value,dict) else {}
    raw = result["multi_model_result"]
    supported=False
    if raw.get("contract_version") is not None:
        try:validate_research_response(frozen_analysis(raw));supported=True
        except ValueError:warnings.append("Unsupported or invalid stored inference contract; retained as historical, not upgraded")
    result["record_kind"] = "versioned_research" if supported else "historical"
    result["review_status"] = "reviewed" if isinstance(raw.get("human_review"),dict) and raw["human_review"].get("status")=="recorded" else "unreviewed"
    result["storage_warnings"] = warnings
    return result


@app.get("/patients")
def list_patients() -> dict:
    conn = _get_db()
    try:
        return {"patients":[_decode_record(r) for r in conn.execute("SELECT * FROM patients ORDER BY date DESC").fetchall()]}
    finally:conn.close()


@app.post("/patients")
def upsert_patient(payload: PatientUpsert) -> dict:
    conn = _get_db()
    try:
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT * FROM patients WHERE id=?",(payload.id,)).fetchone()
            if existing is not None:
                prior = _decode_record(existing)
                # Original inference, images, narratives and review history are
                # immutable here. Reviews append through the feedback route.
                fields=("date","prediction","confidence","size_cm","stage","report_details","original_image","heatmap_image","multi_model_result")
                if any(getattr(payload,k)!=prior[k] for k in fields):
                    raise HTTPException(409,"Original analysis/review history cannot be replaced; submit an independent review")
                conn.execute("UPDATE patients SET name=?,age=? WHERE id=?",(payload.name,payload.age,payload.id))
            else:
                try:validate_new_analysis(payload)
                except ValueError as exc:raise HTTPException(422,str(exc)) from exc
                conn.execute("""INSERT INTO patients
                    (id,name,age,date,prediction,confidence,size_cm,stage,report_json,original_image,heatmap_image,multi_model_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (payload.id,payload.name,payload.age,payload.date,payload.prediction,payload.confidence,payload.size_cm,payload.stage,
                     json.dumps(payload.report_details,ensure_ascii=False),payload.original_image,payload.heatmap_image,
                     json.dumps(payload.multi_model_result,ensure_ascii=False)))
        return {"status":"ok","id":payload.id}
    finally:conn.close()


@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: str) -> dict:
    conn = _get_db()
    try:
        with conn:
            if conn.execute("SELECT 1 FROM patients WHERE id=?",(patient_id,)).fetchone() is None:
                raise HTTPException(404,"Research record not found")
            conn.execute("DELETE FROM feedback_logs WHERE case_id=?",(patient_id,))
            conn.execute("DELETE FROM patients WHERE id=?",(patient_id,))
        return {"status":"success","deleted_id":patient_id}
    finally:conn.close()


@app.post("/feedback/submit")
def submit_feedback(feedback: FeedbackSubmission) -> dict:
    """Append an explicitly unverified human review; never overwrite AI outputs."""
    conn = _get_db()
    try:
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            row=conn.execute("SELECT * FROM patients WHERE id=?",(feedback.case_id,)).fetchone()
            if row is None:raise HTTPException(404,"Research record not found")
            patient=_decode_record(row);raw=patient["multi_model_result"]
            if patient["storage_warnings"] or not isinstance(raw.get("review_history",[]),list):
                raise HTTPException(409,"Historical metadata cannot be safely appended; original record retained")
            feedback_id=str(uuid.uuid4());created_at=datetime.now(timezone.utc).isoformat()
            versioned=bool(raw.get("contract_version"))
            original_prediction=raw.get("consensus",{}).get("prediction",ABSTENTION) if versioned else patient["prediction"]
            review=dict(status="recorded",kind=feedback.feedback_type,prediction=feedback.corrected_prediction,
                notes=feedback.reasoning,reviewer_id=feedback.doctor_id,identity_verified=False,created_at=created_at,
                feedback_id=feedback_id,declared_confidence=feedback.corrected_confidence)
            history=list(raw.get("review_history",[]));history.append(review)
            updated=raw|dict(human_review=review,review_history=history)
            assert frozen_analysis(updated)==frozen_analysis(raw)
            # Both old confidence columns are NOT NULL. Placeholders are never
            # used as model probabilities/statistical accuracy denominators.
            conn.execute("""INSERT INTO feedback_logs
                (id,case_id,doctor_id,original_prediction,original_confidence,corrected_prediction,corrected_confidence,reasoning,feedback_type,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (feedback_id,feedback.case_id,feedback.doctor_id,original_prediction,0 if versioned else patient["confidence"],
                 feedback.corrected_prediction or "Not assessed",feedback.corrected_confidence if feedback.corrected_confidence is not None else 0,
                 feedback.reasoning,feedback.feedback_type,created_at))
            conn.execute("UPDATE patients SET multi_model_json=? WHERE id=?",(json.dumps(updated,ensure_ascii=False),feedback.case_id))
        return {"status":"ok","feedback_id":feedback_id,"review":review,"identity_verified":False}
    finally:conn.close()


@app.get("/feedback/stats")
def get_feedback_stats() -> dict:
    conn = _get_db()
    try:
        patients=[_decode_record(r) for r in conn.execute("SELECT * FROM patients").fetchall()]
        feedback=[dict(r) for r in conn.execute("SELECT * FROM feedback_logs ORDER BY created_at DESC,id DESC").fetchall()]
        return workflow_statistics(patients,feedback)
    finally:conn.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
