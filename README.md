# End-to-End-Sentimental-analysis
"""
Amazon Reviews Sentiment Analysis — FastAPI Backend
Run: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences      # type: ignore

from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.responses import FileResponse  # type: ignore
from pydantic import BaseModel, field_validator  # pydantic v2

# ── Paths ────────────────────────────────────────────────────
#  Structure on disk:
#  deployment/
#  ├── app.py
#  └── models/
#      ├── best_model_final.keras
#      ├── model_meta.json
#      ├── tokenizer_config.json
#      └── static/
#          └── index.html

BASE_DIR = Path(__file__).resolve().parent  
MODEL_PATH = BASE_DIR / "models" / "best_model_final.keras"
TOK_PATH   = BASE_DIR / "models" /"tokenizer_config.json"
META_PATH  = BASE_DIR /"models" / "model_meta.json"

STATIC_DIR = BASE_DIR / "static"

# ── Validate paths before loading ────────────────────────────
for p in [MODEL_PATH, TOK_PATH, META_PATH, STATIC_DIR]:
    if not p.exists():
        raise FileNotFoundError(f"Required path not found: {p}")

# ── Load model & tokeniser on startup ────────────────────────
print("Loading model & tokeniser ...")

with open(META_PATH, encoding="utf-8") as f:
    META = json.load(f)

with open(TOK_PATH, encoding="utf-8") as f:
    TOKENIZER = tokenizer_from_json(f.read())

MODEL = tf.keras.models.load_model(str(MODEL_PATH))

# Warm-up (first predict triggers XLA compilation)
_dummy = pad_sequences([[1, 2, 3]], maxlen=META["max_len"], padding="post")
MODEL.predict(_dummy, verbose=0)

print(
    f"Model loaded  : {META['best_model_name']}\n"
    f"Accuracy      : {META['test_accuracy']:.4f}\n"
    f"ROC-AUC       : {META['test_roc_auc']:.4f}"
)

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="Amazon Sentiment API",
    description="Deep Learning sentiment analysis on Amazon reviews",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas (pydantic v2) ────────────────────────────
class TextInput(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        return v


class BatchInput(BaseModel):
    texts: List[str]

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        cleaned = [t.strip() for t in v if t.strip()]
        if not cleaned:
            raise ValueError("texts list must not be empty")
        if len(cleaned) > 50:
            raise ValueError("maximum 50 texts per batch")
        return cleaned


# latency_ms is Optional — added after _predict() returns
class PredictionResult(BaseModel):
    text: str
    label: str
    confidence: float
    score: float
    latency_ms: Optional[float] = None


# ── Core inference ────────────────────────────────────────────
def _predict(texts: List[str]) -> List[dict]:
    seqs  = TOKENIZER.texts_to_sequences(texts)
    pads  = pad_sequences(seqs, maxlen=META["max_len"],
                          padding="post", truncating="post")
    probs = MODEL.predict(pads, verbose=0).flatten().tolist()

    results = []
    for txt, prob in zip(texts, probs):
        label = "Positive" if prob >= 0.5 else "Negative"
        conf  = prob if prob >= 0.5 else 1.0 - prob
        results.append({
            "text":       txt,
            "label":      label,
            "confidence": round(conf, 4),
            "score":      round(prob, 4),
        })
    return results


# ── API Routes  (must be declared BEFORE app.mount) ──────────
@app.get("/api/health")
def health():
    return {
        "status":   "ok",
        "model":    META["best_model_name"],
        "accuracy": META["test_accuracy"],
        "auc":      META["test_roc_auc"],
    }


@app.get("/api/model-info")
def model_info():
    return {
        "model_name":    META["best_model_name"],
        "test_accuracy": META["test_accuracy"],
        "test_roc_auc":  META["test_roc_auc"],
        "vocab_size":    META["vocab_size"],
        "max_len":       META["max_len"],
        "embed_dim":     META["embed_dim"],
        "dl_samples":    META["dl_samples"],
        "all_results":   META.get("all_results", {}),
    }


@app.post("/api/predict", response_model=PredictionResult)
def predict(body: TextInput):
    t0  = time.perf_counter()
    res = _predict([body.text])[0]
    res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return res


@app.post("/api/predict/batch")
def predict_batch(body: BatchInput):
    t0      = time.perf_counter()
    results = _predict(body.texts)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "results":    results,
        "count":      len(results),
        "latency_ms": elapsed,
    }


# ── Static files (AFTER API routes) ──────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/{full_path:path}")
def catch_all(full_path: str):
    """SPA fallback — serve index.html for every non-API path."""
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn      # type: ignore
    import webbrowser
    import threading
    import socket

    PORT = 8000

    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    local_ip = get_local_ip()

    def print_links():
        import time
        time.sleep(1.5)
        print("\n" + "=" * 55)
        print("  SentimentAI is running!")
        print("=" * 55)
        print(f"  Local    ->  http://localhost:{PORT}")
        print(f"  Network  ->  http://{local_ip}:{PORT}")
        print(f"  API Docs ->  http://localhost:{PORT}/docs")
        print("=" * 55 + "\n")

    threading.Thread(target=print_links, daemon=True).start()

    def open_browser():
        import time
        time.sleep(2)
        webbrowser.open(f"http://localhost:{PORT}")

    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)

    """
Amazon Reviews Sentiment Analysis — FastAPI Backend
Run: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences      # type: ignore

from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.responses import FileResponse  # type: ignore
from pydantic import BaseModel, field_validator  # pydantic v2

# ── Paths ────────────────────────────────────────────────────
#  Structure on disk:
#  deployment/
#  ├── app.py
#  └── models/
#      ├── best_model_final.keras
#      ├── model_meta.json
#      ├── tokenizer_config.json
#      └── static/
#          └── index.html

BASE_DIR = Path(__file__).resolve().parent  
MODEL_PATH = BASE_DIR / "models" / "best_model_final.keras"
TOK_PATH   = BASE_DIR / "models" /"tokenizer_config.json"
META_PATH  = BASE_DIR /"models" / "model_meta.json"

STATIC_DIR = BASE_DIR / "static"

# ── Validate paths before loading ────────────────────────────
for p in [MODEL_PATH, TOK_PATH, META_PATH, STATIC_DIR]:
    if not p.exists():
        raise FileNotFoundError(f"Required path not found: {p}")

# ── Load model & tokeniser on startup ────────────────────────
print("Loading model & tokeniser ...")

with open(META_PATH, encoding="utf-8") as f:
    META = json.load(f)

with open(TOK_PATH, encoding="utf-8") as f:
    TOKENIZER = tokenizer_from_json(f.read())

MODEL = tf.keras.models.load_model(str(MODEL_PATH))

# Warm-up (first predict triggers XLA compilation)
_dummy = pad_sequences([[1, 2, 3]], maxlen=META["max_len"], padding="post")
MODEL.predict(_dummy, verbose=0)

print(
    f"Model loaded  : {META['best_model_name']}\n"
    f"Accuracy      : {META['test_accuracy']:.4f}\n"
    f"ROC-AUC       : {META['test_roc_auc']:.4f}"
)

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="Amazon Sentiment API",
    description="Deep Learning sentiment analysis on Amazon reviews",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas (pydantic v2) ────────────────────────────
class TextInput(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        return v


class BatchInput(BaseModel):
    texts: List[str]

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        cleaned = [t.strip() for t in v if t.strip()]
        if not cleaned:
            raise ValueError("texts list must not be empty")
        if len(cleaned) > 50:
            raise ValueError("maximum 50 texts per batch")
        return cleaned


# latency_ms is Optional — added after _predict() returns
class PredictionResult(BaseModel):
    text: str
    label: str
    confidence: float
    score: float
    latency_ms: Optional[float] = None


# ── Core inference ────────────────────────────────────────────
def _predict(texts: List[str]) -> List[dict]:
    seqs  = TOKENIZER.texts_to_sequences(texts)
    pads  = pad_sequences(seqs, maxlen=META["max_len"],
                          padding="post", truncating="post")
    probs = MODEL.predict(pads, verbose=0).flatten().tolist()

    results = []
    for txt, prob in zip(texts, probs):
        label = "Positive" if prob >= 0.5 else "Negative"
        conf  = prob if prob >= 0.5 else 1.0 - prob
        results.append({
            "text":       txt,
            "label":      label,
            "confidence": round(conf, 4),
            "score":      round(prob, 4),
        })
    return results


# ── API Routes  (must be declared BEFORE app.mount) ──────────
@app.get("/api/health")
def health():
    return {
        "status":   "ok",
        "model":    META["best_model_name"],
        "accuracy": META["test_accuracy"],
        "auc":      META["test_roc_auc"],
    }


@app.get("/api/model-info")
def model_info():
    return {
        "model_name":    META["best_model_name"],
        "test_accuracy": META["test_accuracy"],
        "test_roc_auc":  META["test_roc_auc"],
        "vocab_size":    META["vocab_size"],
        "max_len":       META["max_len"],
        "embed_dim":     META["embed_dim"],
        "dl_samples":    META["dl_samples"],
        "all_results":   META.get("all_results", {}),
    }


@app.post("/api/predict", response_model=PredictionResult)
def predict(body: TextInput):
    t0  = time.perf_counter()
    res = _predict([body.text])[0]
    res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return res


@app.post("/api/predict/batch")
def predict_batch(body: BatchInput):
    t0      = time.perf_counter()
    results = _predict(body.texts)
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "results":    results,
        "count":      len(results),
        "latency_ms": elapsed,
    }


# ── Static files (AFTER API routes) ──────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/{full_path:path}")
def catch_all(full_path: str):
    """SPA fallback — serve index.html for every non-API path."""
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn      # type: ignore
    import webbrowser
    import threading
    import socket

    PORT = 8000

    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    local_ip = get_local_ip()

    def print_links():
        import time
        time.sleep(1.5)
        print("\n" + "=" * 55)
        print("  SentimentAI is running!")
        print("=" * 55)
        print(f"  Local    ->  http://localhost:{PORT}")
        print(f"  Network  ->  http://{local_ip}:{PORT}")
        print(f"  API Docs ->  http://localhost:{PORT}/docs")
        print("=" * 55 + "\n")

    threading.Thread(target=print_links, daemon=True).start()

    def open_browser():
        import time
        time.sleep(2)
        webbrowser.open(f"http://localhost:{PORT}")

    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
