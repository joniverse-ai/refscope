"""수집물을 리서치 카드로 바꾼다 — 노동을 넘기고 판단을 남기는 단계.

여기서 모델을 역할에 따라 나눠 쓴다. OCR 비교 실험에서 얻은 결론이 그대로
설계가 된다.

  글자를 읽는 일     → Apple Vision (VLM보다 91배 빠르고 정확도는 동등)
  글자를 고르는 일   → 규칙 (크기·위치. 모델이 필요 없다)
  구조를 나누는 일   → LLM, 텍스트만 (좌표 붙은 글자 목록이면 충분)
  분위기를 말하는 일 → VLM, 그림 (여기만 눈이 필요하다)

VLM에게 OCR을 시키지 않는 것이 이 설계의 핵심이다.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from . import config

Image.MAX_IMAGE_PIXELS = None
OLLAMA_URL = "http://localhost:11434"


# ── 카피 고르기 (모델 없이) ────────────────────────────────────────────────
NOISE_PATTERNS = re.compile(
    r"^(장바구니|구매하기|바로구매|관심상품|로그인|회원가입|검색|더보기|MORE|"
    r"배송|교환|반품|환불|고객센터|사업자|통신판매|이용약관|개인정보|COPYRIGHT|"
    r"\d[\d,\.]*원?|[\d\-\.\s]+|[A-Za-z0-9\-_/\.]+)$",
    re.IGNORECASE,
)


@dataclass
class Copy:
    text: str
    y: int
    size: float  # DOM은 font-size(px), OCR은 글자 높이(px)
    source: str  # dom | ocr


def is_noise(text: str) -> bool:
    t = text.strip()
    if len(t) < 2 or len(t) > 120:
        return True
    if NOISE_PATTERNS.match(t):
        return True
    # 한글이나 영문이 하나도 없으면 버린다 (치수·코드·기호 덩어리)
    return not re.search(r"[가-힣A-Za-z]", t)


def collect_copy(ref_dir: Path, engine: str = "apple_vision", top: int = 25) -> list[Copy]:
    """DOM 텍스트와 OCR 결과를 합쳐 '헤드라인급' 카피를 고른다.

    고르는 기준은 디자이너가 페이지를 훑을 때 쓰는 기준과 같다 — **큰 글자**다.
    상세페이지에서 크게 쓴 글자는 브랜드가 강조하고 싶은 말이다.
    """
    items: list[Copy] = []

    dom_path = ref_dir / "dom.json"
    if dom_path.exists():
        for d in json.loads(dom_path.read_text(encoding="utf-8")):
            if is_noise(d["text"]):
                continue
            items.append(Copy(d["text"].strip(), d["y"], d.get("font_size") or 0, "dom"))

    ocr_path = config.DERIVED_DIR / "ocr" / f"{ref_dir.name}__{engine}.json"
    if ocr_path.exists():
        data = json.loads(ocr_path.read_text(encoding="utf-8"))
        # 조각 정보에서 글자 높이를 가져온다. 없으면 0.
        for ln in data["lines"]:
            if is_noise(ln["text"]):
                continue
            # OCR이 주는 h는 글자 상자의 높이라 CSS font-size보다 크다(대략 1.4배).
            # 두 출처를 한 줄에 세워 정렬하려면 같은 자로 재야 한다.
            size = (ln.get("h") or 0) / 1.4
            items.append(Copy(ln["text"].strip(), ln.get("page_y") or 0, size, "ocr"))

    # 같은 글자가 DOM과 OCR 양쪽에서 나오면 하나만 남긴다
    seen: dict[str, Copy] = {}
    for c in items:
        key = re.sub(r"\s+", "", c.text)
        prev = seen.get(key)
        if prev is None or c.size > prev.size:
            seen[key] = c

    ranked = sorted(seen.values(), key=lambda c: (-c.size, c.y))
    return sorted(ranked[:top], key=lambda c: c.y)


# ── VLM / LLM 호출 ────────────────────────────────────────────────────────
def ask_ollama(
    model: str, prompt: str, images: list[bytes] | None = None, timeout: int = 900
) -> tuple[str, str]:
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    if images:
        payload["images"] = [base64.b64encode(b).decode() for b in images]
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("response", "").strip(), ""
    except urllib.error.URLError as e:
        return "", f"Ollama 연결 실패: {e}"
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


def contact_sheet(page_png: Path, strip_w: int = 300, n_strips: int = 6) -> bytes:
    """긴 페이지를 세로로 잘라 옆으로 나란히 붙인 '한 장'을 만든다.

    44,491px짜리 페이지를 VLM에 보여주는 방법. 그냥 축소하면 아무것도 안 보이고,
    조각내어 여러 번 물으면 전체 구성을 못 본다. 나란히 붙이면 한 번에 훑는다.
    """
    img = Image.open(page_png).convert("RGB")
    w, h = img.size
    scale = strip_w / w
    img = img.resize((strip_w, max(1, int(h * scale))), Image.LANCZOS)
    sh = -(-img.height // n_strips)
    sheet = Image.new("RGB", (strip_w * n_strips + 8 * (n_strips - 1), sh), "white")
    for i in range(n_strips):
        piece = img.crop((0, i * sh, strip_w, min((i + 1) * sh, img.height)))
        sheet.paste(piece, (i * (strip_w + 8), 0))
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()


TONE_PROMPT = """이 이미지는 한국 온라인 쇼핑몰 상품 상세페이지 한 장을 세로로 6등분해
왼쪽부터 오른쪽으로 나란히 붙인 것입니다. 즉 왼쪽 끝이 페이지 맨 위, 오른쪽 끝이 맨 아래입니다.

브랜드 디자이너의 눈으로 아래 네 가지만 한국어로 답하세요. 글자를 읽어 옮기지는 마세요.

1. 색·톤: 지배적인 색감과 명도 대비를 한 문장으로
2. 사진 스타일: 제품컷 위주인가 연출컷 위주인가, 배경 처리는 어떤가
3. 레이아웃: 여백이 넉넉한가 빽빽한가, 정보 밀도는
4. 전체 인상: 이 페이지가 주는 격(格)을 한 문장으로

각 항목을 "1. ", "2. " 로 시작하는 한 줄씩, 총 4줄로만 답하세요."""


# 처음에는 "텍스트 목록을 주고 섹션으로 묶어줘"라고 한 번에 시켰다. qwen2.5vl:7b는
# 묶지 않고 입력 20줄을 그대로 20개 섹션으로 되뱉었다. 나누는 일과 이름 짓는 일을
# 한꺼번에 시킨 게 잘못이었다.
#   → 나누기는 y좌표 간격으로 결정적으로 하고(아래 split_blocks),
#     LLM에게는 **이름 짓기만** 맡긴다. 모델에게 잘하는 일만 시킨다.
SECTION_PROMPT = """아래는 한국 쇼핑몰 상품 상세페이지를 위에서 아래로 {n}개 덩어리로 나눈 것입니다.
각 덩어리에 이름을 붙이세요.

쓸 만한 이름 예: 히어로/메인비주얼, 브랜드 스토리, 제품 소개, 구성품 안내, 성분·원산지,
사용법·보관법, 패키지 소개, 배송 안내, 교환·반품, 후기, Q&A

규칙:
- 정확히 {n}줄만 출력합니다. 덩어리 하나에 한 줄.
- 형식은 "번호 | 이름" 입니다. 이름은 12자 이내.
- 설명·머리말을 붙이지 마세요.

{blocks}"""


def split_blocks(copies: list[Copy], page_h: int) -> list[list[Copy]]:
    """세로 간격이 큰 곳을 경계로 삼아 덩어리로 나눈다.

    상세페이지에서 섹션이 바뀔 때는 반드시 여백이 생긴다. 그 여백을 재면
    모델 없이도 경계를 찾을 수 있다. 몇 개로 나눌지는 페이지 길이에 맞춘다.
    """
    if not copies:
        return []
    ordered = sorted(copies, key=lambda c: c.y)
    k = max(5, min(8, round(page_h / 3000)))
    if len(ordered) <= k:
        return [[c] for c in ordered]

    gaps = sorted(
        ((ordered[i + 1].y - ordered[i].y, i) for i in range(len(ordered) - 1)),
        reverse=True,
    )
    cuts = sorted(i for _, i in gaps[: k - 1])
    blocks, start = [], 0
    for c in cuts:
        blocks.append(ordered[start : c + 1])
        start = c + 1
    blocks.append(ordered[start:])
    return [b for b in blocks if b]


@dataclass
class Card:
    """한 레퍼런스의 리서치 카드. HTML 출력의 입력이 된다."""

    id: str
    name: str
    url: str
    group: str
    page_type: str
    page_height: int
    image_locked_ratio: float
    palette: list[dict] = field(default_factory=list)
    copy_lines: list[dict] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)
    tone: list[str] = field(default_factory=list)
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def build_card(
    ref: dict, engine: str = "apple_vision", model: str = "qwen2.5vl:7b"
) -> Card:
    t0 = time.perf_counter()
    ref_dir = config.REFS_DIR / ref["id"]
    meta = json.loads((ref_dir / "meta.json").read_text(encoding="utf-8"))
    regions = json.loads((ref_dir / "regions.json").read_text(encoding="utf-8"))

    card = Card(
        id=ref["id"],
        name=ref.get("name", ref["id"]),
        url=ref["url"],
        group=ref.get("group", ""),
        page_type=ref.get("page_type", ""),
        page_height=meta["page_height"],
        image_locked_ratio=regions["image_locked_ratio"],
    )

    palette_path = ref_dir / "palette.json"
    if palette_path.exists():
        card.palette = json.loads(palette_path.read_text(encoding="utf-8"))

    copies = collect_copy(ref_dir, engine=engine)
    card.copy_lines = [
        {"text": c.text, "y": c.y, "size": c.size, "source": c.source} for c in copies
    ]

    # 구조 나누기 — 경계는 좌표로 정하고, 이름만 모델에게 맡긴다
    blocks = split_blocks(copies, card.page_height)
    if blocks:
        listing = "\n\n".join(
            f"[{i + 1}] (y {b[0].y}~{b[-1].y})\n"
            + "\n".join(f"  {c.text}" for c in b[:6])
            for i, b in enumerate(blocks)
        )
        out, err = ask_ollama(
            model, SECTION_PROMPT.format(n=len(blocks), blocks=listing)
        )
        if err:
            card.errors.append(f"섹션 이름: {err}")
        names: dict[int, str] = {}
        for line in out.splitlines():
            m = re.match(r"\s*\[?(\d+)\]?\s*\|\s*(.+?)\s*$", line)
            if m:
                names[int(m.group(1))] = m.group(2)[:20]
        for i, b in enumerate(blocks, start=1):
            card.sections.append(
                {
                    "y": b[0].y,
                    "y_end": b[-1].y,
                    "name": names.get(i, "(이름 없음)"),
                    "n_lines": len(b),
                }
            )

    # 분위기 말하기 — 여기만 눈이 필요하다
    sheet = contact_sheet(ref_dir / "page.png")
    out, err = ask_ollama(model, TONE_PROMPT, images=[sheet])
    if err:
        card.errors.append(f"톤 서술: {err}")
    card.tone = [
        re.sub(r"^\s*\d+\.\s*", "", ln).strip()
        for ln in out.splitlines()
        if re.match(r"\s*\d+\.", ln)
    ]

    card.seconds = round(time.perf_counter() - t0, 1)
    return card
