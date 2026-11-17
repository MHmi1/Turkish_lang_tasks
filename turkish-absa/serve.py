"""FastAPI microservice for Turkish ABSA (explicit + implicit + sarcastic/idiomatic reviews).
ABSA_ATE_DIR=work/export/ate_model ABSA_APC_DIR=work/export/apc_model uvicorn serve:app --port 8001
"""
import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Turkish ABSA API (Figurative & Slang Aware)", version="1.0")
_pipe = None


def pipe():
    global _pipe
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
    return pipe().analyze(r.text, include_implicit_fallback=r.include_implicit_fallback)
