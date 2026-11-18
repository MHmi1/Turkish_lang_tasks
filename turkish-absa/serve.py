"""FastAPI microservice for Turkish aspect extraction and polarity classification.
ABSA_ATE_DIR=work/export/ate_model ABSA_APC_DIR=work/export/apc_model uvicorn serve:app --port 8001
"""
import os
import threading
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="Turkish ABSA API", version="1.0")
_pipe = None
_load_lock = threading.Lock()


def pipe():
    global _pipe
    if _pipe is None:
        with _load_lock:
            if _pipe is None:
                from absa.infer import ABSAPipeline
                ate_dir = os.environ.get("ABSA_ATE_DIR", "work/export/ate_model")
                apc_dir = os.environ.get("ABSA_APC_DIR", "work/export/apc_model")
                dev = os.environ.get("ABSA_DEVICE", "cpu")
                _pipe = ABSAPipeline(ate_dir, apc_dir, device=dev)
    return _pipe


class Req(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    include_implicit_fallback: bool = True
    domain: Optional[str] = None

    @field_validator("text")
    @classmethod
    def non_blank_text(cls, value):
        if not value.strip():
            raise ValueError("text must contain a non-whitespace character")
        return value


class AspectOut(BaseModel):
    span: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    aspect: str
    polarity: str
    confidence: float
    implicit: bool


class Resp(BaseModel):
    text: str
    aspects: List[AspectOut]
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipe is not None}


@app.post("/analyze", response_model=Resp)
def analyze(r: Req):
    model = pipe()
    try:
        return model.analyze(r.text, include_implicit_fallback=r.include_implicit_fallback,
                             domain=r.domain)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
