"""이미지에 갇힌 글자를 꺼내는 여러 개의 눈.

엔진마다 성격이 다르다. 하나를 고르기 전에 같은 조각에 나란히 돌려 재는 것이
이 모듈의 목적이다. (모델 선정 근거 문서 참고)

  apple_vision — macOS 내장 Vision. 설치할 모델이 없고 빠르다. macOS 전용.
  paddleocr    — 고전적인 OCR 조립 라인(검출 → 인식). 어디서나 돈다.
  easyocr      — 같은 계열이지만 인식 모델이 다르다. torch 기반.
  qwen2.5vl    — VLM. 글자만이 아니라 "무엇이 보이는지"를 말할 수 있다.

공통 반환 형식은 Line 리스트다. 어떤 엔진을 쓰든 뒷단은 같은 것을 본다.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

OLLAMA_URL = "http://localhost:11434"


@dataclass
class Line:
    text: str
    conf: float
    # 조각 안에서의 좌표. y는 위에서부터. VLM은 좌표를 주지 않으므로 None일 수 있다.
    y: int | None = None
    h: int | None = None


@dataclass
class OcrResult:
    engine: str
    lines: list[Line] = field(default_factory=list)
    seconds: float = 0.0
    error: str = ""

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)

    @property
    def chars(self) -> int:
        return sum(len(ln.text) for ln in self.lines)


class Engine(Protocol):
    name: str

    def read(self, image_path: Path) -> OcrResult: ...


# ──────────────────────────────────────────────────────────────────────────
class AppleVision:
    """macOS Vision 프레임워크. 모델을 따로 받지 않고 OS가 가진 것을 쓴다."""

    name = "apple_vision"

    def __init__(self, languages: tuple[str, ...] = ("ko-KR", "en-US")):
        self.languages = list(languages)

    def read(self, image_path: Path) -> OcrResult:
        t0 = time.perf_counter()
        try:
            import Quartz
            import Vision
            from Foundation import NSURL
        except ImportError as e:
            return OcrResult(self.name, error=f"pyobjc 미설치: {e}")

        url = NSURL.fileURLWithPath_(str(image_path))
        src = Quartz.CGImageSourceCreateWithURL(url, None)
        if src is None:
            return OcrResult(self.name, error=f"이미지를 열 수 없음: {image_path}")
        cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
        height = Quartz.CGImageGetHeight(cg)

        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLanguages_(self.languages)
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setUsesLanguageCorrection_(True)
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
        ok, err = handler.performRequests_error_([req], None)
        if not ok:
            return OcrResult(self.name, error=f"Vision 실패: {err}")

        lines: list[Line] = []
        for obs in req.results() or []:
            cand = obs.topCandidates_(1)
            if not cand:
                continue
            top = cand[0]
            box = obs.boundingBox()  # 정규화 좌표, 원점은 좌하단
            y = int((1.0 - box.origin.y - box.size.height) * height)
            lines.append(
                Line(top.string(), float(top.confidence()), y, int(box.size.height * height))
            )
        lines.sort(key=lambda ln: (ln.y or 0))
        return OcrResult(self.name, lines, time.perf_counter() - t0)


# ──────────────────────────────────────────────────────────────────────────
class PaddleOcr:
    """고전적인 검출 → 인식 조립 라인. 커리큘럼 4-2절의 그 구조."""

    name = "paddleocr"
    _shared = None

    def __init__(self, lang: str = "korean"):
        self.lang = lang

    def _engine(self):
        if PaddleOcr._shared is None:
            import warnings

            warnings.filterwarnings("ignore")
            from paddleocr import PaddleOCR

            PaddleOcr._shared = PaddleOCR(
                lang=self.lang, show_log=False, use_angle_cls=False
            )
        return PaddleOcr._shared

    def read(self, image_path: Path) -> OcrResult:
        t0 = time.perf_counter()
        try:
            engine = self._engine()
        except Exception as e:
            return OcrResult(self.name, error=f"PaddleOCR 초기화 실패: {e}")
        try:
            raw = engine.ocr(str(image_path), cls=False)
        except Exception as e:
            return OcrResult(self.name, error=f"{type(e).__name__}: {e}")

        lines: list[Line] = []
        for entry in (raw[0] if raw and raw[0] else []):
            box, (text, conf) = entry
            ys = [p[1] for p in box]
            lines.append(Line(text, float(conf), int(min(ys)), int(max(ys) - min(ys))))
        lines.sort(key=lambda ln: (ln.y or 0))
        return OcrResult(self.name, lines, time.perf_counter() - t0)


# ──────────────────────────────────────────────────────────────────────────
class EasyOcr:
    """torch 기반 OCR. PaddleOCR과 같은 계열이지만 인식 모델이 다르다."""

    name = "easyocr"
    _shared = None

    def __init__(self, langs: tuple[str, ...] = ("ko", "en")):
        self.langs = list(langs)

    def _engine(self):
        if EasyOcr._shared is None:
            import easyocr

            EasyOcr._shared = easyocr.Reader(self.langs, gpu=False, verbose=False)
        return EasyOcr._shared

    def read(self, image_path: Path) -> OcrResult:
        t0 = time.perf_counter()
        try:
            engine = self._engine()
        except Exception as e:
            return OcrResult(self.name, error=f"EasyOCR 초기화 실패: {e}")
        try:
            raw = engine.readtext(str(image_path))
        except Exception as e:
            return OcrResult(self.name, error=f"{type(e).__name__}: {e}")

        lines: list[Line] = []
        for box, text, conf in raw:
            ys = [p[1] for p in box]
            lines.append(Line(text, float(conf), int(min(ys)), int(max(ys) - min(ys))))
        lines.sort(key=lambda ln: (ln.y or 0))
        return OcrResult(self.name, lines, time.perf_counter() - t0)


# ──────────────────────────────────────────────────────────────────────────
VLM_OCR_PROMPT = """이 이미지는 한국 온라인 쇼핑몰 상품 상세페이지의 일부입니다.
이미지 안에 보이는 **모든 한글·영문 텍스트**를 위에서 아래 순서로 그대로 옮겨 적으세요.

규칙:
- 한 줄에 하나씩, 다른 말 없이 텍스트만 출력합니다.
- 없는 글자를 지어내지 마세요. 안 보이면 적지 마세요.
- 번역하지 말고 보이는 그대로 적으세요.
- 설명·요약·머리말을 붙이지 마세요."""


class QwenVL:
    """VLM으로 읽기. 느리지만 글자 너머를 말할 수 있다.

    VLM은 입력 이미지를 정해진 해상도로 줄여서 본다. 3,494px짜리 세로 조각을
    그대로 주면 작은 글자가 뭉개져 버린다. 그래서 이 엔진만 내부에서 한 번 더
    쪼갠다 — 다른 엔진을 불리하게 만들지 않으면서 VLM에도 공정한 입력을 준다.
    """

    name = "qwen2.5vl"
    TILE_H = 1200
    TILE_OVERLAP = 100

    def __init__(self, model: str = "qwen2.5vl:7b", prompt: str = VLM_OCR_PROMPT):
        self.model = model
        self.prompt = prompt

    def _ask(self, png_bytes: bytes, prompt: str | None = None) -> tuple[str, str]:
        import urllib.error
        import urllib.request

        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt or self.prompt,
                "images": [base64.b64encode(png_bytes).decode()],
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 8192},
            }
        ).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                return json.loads(r.read()).get("response", ""), ""
        except urllib.error.URLError as e:
            return "", f"Ollama 연결 실패: {e}"
        except Exception as e:
            return "", f"{type(e).__name__}: {e}"

    def read(self, image_path: Path) -> OcrResult:
        import io

        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        t0 = time.perf_counter()
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        tiles: list[bytes] = []
        y = 0
        while y < h:
            y1 = min(y + self.TILE_H, h)
            buf = io.BytesIO()
            img.crop((0, y, w, y1)).save(buf, format="PNG")
            tiles.append(buf.getvalue())
            if y1 >= h:
                break
            y = y1 - self.TILE_OVERLAP

        lines: list[Line] = []
        for tile in tiles:
            text, err = self._ask(tile)
            if err:
                return OcrResult(self.name, lines, time.perf_counter() - t0, err)
            for s in text.splitlines():
                s = s.strip().lstrip("-•*# ").strip()
                if s:
                    lines.append(Line(s, 1.0))
        return OcrResult(self.name, lines, time.perf_counter() - t0)


ENGINES: dict[str, type] = {
    "apple_vision": AppleVision,
    "paddleocr": PaddleOcr,
    "easyocr": EasyOcr,
    "qwen2.5vl": QwenVL,
}


def get_engine(name: str) -> Engine:
    if name not in ENGINES:
        raise ValueError(f"모르는 엔진: {name}. 가능한 것: {sorted(ENGINES)}")
    return ENGINES[name]()
