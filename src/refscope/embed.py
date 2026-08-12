"""bge-m3 임베딩 (Ollama 로컬).

한국어 문장을 같은 의미 공간에 올린다. 두 군데에서 쓴다.
  - 섹션 이름 채점: 사람 "제품사용 Tip" ↔ 기계 "사용법·보관법" 은 문자열로 못 맞춘다
  - 카피 클러스터링: 카테고리 공통 메시지 유형 도출
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from . import config

OLLAMA_URL = "http://localhost:11434"
MODEL = "bge-m3"
CACHE = config.DERIVED_DIR / "embeddings.json"


def _load_cache() -> dict[str, list[float]]:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache), encoding="utf-8")


def _key(text: str, model: str) -> str:
    return hashlib.sha1(f"{model}::{text}".encode()).hexdigest()


def embed(texts: list[str], model: str = MODEL, use_cache: bool = True) -> np.ndarray:
    """문장 목록을 벡터로. 같은 문장은 다시 계산하지 않는다."""
    cache = _load_cache() if use_cache else {}
    out: list[list[float]] = []
    dirty = False
    for t in texts:
        k = _key(t, model)
        if k in cache:
            out.append(cache[k])
            continue
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embeddings",
            data=json.dumps({"model": model, "prompt": t}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                vec = json.loads(r.read())["embedding"]
        except (urllib.error.URLError, KeyError) as e:
            raise RuntimeError(f"임베딩 실패 ({model}): {e}") from e
        cache[k] = vec
        out.append(vec)
        dirty = True
    if dirty and use_cache:
        _save_cache(cache)
    arr = np.asarray(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-9, None)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """정규화된 벡터끼리의 코사인 유사도 행렬."""
    return a @ b.T


def available() -> bool:
    try:
        embed(["테스트"], use_cache=False)
        return True
    except Exception:
        return False
