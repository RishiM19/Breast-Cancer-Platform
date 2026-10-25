"""The same database-free router is mounted by main.py and exercised in tests."""
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .research_inference import (
    MAX_UPLOAD_BYTES, InvalidImage, ResearchEngine, ResearchResponse, ResearchResponseV11, ResearchResponseV12, contract_descriptor,
)


def create_research_router(engine: ResearchEngine | None = None) -> APIRouter:
    engine = engine or ResearchEngine()
    router = APIRouter()

    @router.get("/inference-contract")
    def inference_contract(version: Literal["1.0.0", "1.1.0", "1.2.0"] = "1.0.0"):
        return contract_descriptor(version)

    @router.post("/predict", response_model=ResearchResponseV12 | ResearchResponseV11 | ResearchResponse)
    async def predict_image(file: UploadFile = File(...),
                            source_domain: Literal["unknown", "BUSI", "UCLM", "BUS-BRA"] = Form("unknown"),
                            contract_version: Literal["1.0.0", "1.1.0", "1.2.0"] = Form("1.0.0")):
        try:
            raw = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Image exceeds the 10 MiB upload limit")
            try:
                if contract_version == "1.0.0":
                    return await run_in_threadpool(engine.analyze, raw, source_domain)
                return await run_in_threadpool(engine.analyze, raw, source_domain, contract_version=contract_version)
            except InvalidImage as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            await file.close()

    return router
