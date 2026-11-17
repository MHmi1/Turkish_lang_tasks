"""FastAPI service: POST /redact, POST /detect, GET /health.
PII_MODEL_DIR=work/export/model uvicorn serve:app --port 8000
"""
import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Turkish PII Redaction", version="1.0")
_pipe = None


def pipe():
    global _pipe
    if _pipe is None:
        from pii.infer import Pipeline
        _pipe = Pipeline(os.environ.get("PII_MODEL_DIR", "work/export/model"), device=os.environ.get("PII_DEVICE", "cpu"))
    return _pipe


class Req(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)
    style: str = Field("typed", pattern="^(typed|label|generic)$")


class Entity(BaseModel):
    start: int
    end: int
    label: str
    score: float
    source: str
    surface: Optional[str] = None


class Resp(BaseModel):
    redacted_text: str
    entities: List[Entity]
    needs_review: bool
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipe is not None}


@app.post("/redact", response_model=Resp)
def redact(r: Req):
    return pipe().redact(r.text, r.style)


@app.post("/detect")
def detect(r: Req):
    return {"entities": pipe().detect(r.text)}
