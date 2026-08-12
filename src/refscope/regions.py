"""읽기 경로 판정 — 이 페이지의 어느 구간을 어느 눈으로 읽을 것인가.

파이프라인에서 모델보다 먼저 오는 '검사' 단계다. DOM에 글자가 있으면 OCR을
돌리지 않는다. 정확한 원본을 흐릿한 사본으로 바꾸는 짓이기 때문이다.

여기서 한 번 틀렸던 것을 기록해 둔다.
처음에는 **페이지 단위로** "DOM 글자 수 ≥ 800이면 dom 경로"라고 판정했다.
교동한과 상품 상세페이지(17,275px)에서 이 판정이 `dom`을 뱉었는데, 실제로는
y=2,000~13,300 구간 — 페이지의 64% — 에 DOM 텍스트가 **0자**였다. 그 자리는
500×8,555px짜리 통이미지 한 장이 차지하고 있었다. 페이지 전체 2,140자는 전부
헤더·푸터·약관 같은 상용구였다.

  → 교훈 1: 상세페이지에서 상용구는 항상 있다. 그것에 속지 않으려면
    **세로 구간(band) 단위로** 봐야 한다.

구간 단위로 바꾼 뒤에도 한 번 더 틀렸다. 피복률을 "이미지 넓이 ÷ 페이지 넓이"로
쟀더니 그 500×8,555 통이미지가 1,440px 페이지 폭의 35%밖에 못 덮어서 임계값에
걸리지 않았다. 세로로 긴 이미지는 **좁아도 그 구간의 전부**다.

  → 교훈 2: 척도는 가로 피복이 아니라 **세로 피복**이다.
    "이 구간의 높이 중 몇 %가 큰 이미지에 덮여 있는가"를 본다.

구간 단위 판정은 덤도 준다. OCR에 넣을 크롭 영역이 판정의 부산물로 나온다.
17,000px짜리 PNG를 통째로 OCR/VLM에 밀어 넣을 수는 없으니 어차피 필요했다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

BAND_H = 500  # 판정 단위 높이(px)
COVER_T = 0.5  # 구간 높이의 이만큼이 콘텐츠 이미지에 덮이면 '그림 구간'
TEXT_MIN = 25  # 구간에 이 글자 수 미만이면 '글자 없음'
MIN_REGION_H = 300  # 이보다 짧은 OCR 영역은 버린다 (배너·아이콘)
MAX_REGION_H = 4000  # OCR/VLM 한 번에 넣을 최대 높이 — 넘으면 쪼갠다

# '콘텐츠 이미지' — 글자가 구워져 있을 만한 크기. 로고·아이콘·썸네일을 걸러낸다.
CONTENT_IMG_MIN_H = 350
CONTENT_IMG_MIN_AREA = 150_000


@dataclass
class Band:
    y0: int
    y1: int
    dom_chars: int
    image_cover: float  # 구간 높이 대비 콘텐츠 이미지의 세로 피복률
    verdict: str  # dom | ocr | both | empty


@dataclass
class Region:
    """연속된 같은 판정의 구간을 묶은 것. OCR/VLM 입력 단위."""

    y0: int
    y1: int
    kind: str  # ocr | dom | both
    dom_chars: int
    image_cover: float

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def needs_ocr(self) -> bool:
        return self.kind in ("ocr", "both")


def is_content_image(im: dict) -> bool:
    """글자가 구워져 있을 만한 이미지인가. 로고·아이콘·썸네일은 뺀다."""
    return im["h"] >= CONTENT_IMG_MIN_H and im["w"] * im["h"] >= CONTENT_IMG_MIN_AREA


def _vertical_cover(images: list[dict], y0: int, y1: int) -> float:
    """구간 [y0,y1)의 높이 중 콘텐츠 이미지에 덮인 비율.

    가로 피복이 아니라 세로 피복인 이유는 모듈 상단 '교훈 2' 참고.
    겹치는 이미지를 이중으로 세지 않도록 구간 합집합을 쓴다.
    """
    spans = []
    for im in images:
        if not is_content_image(im):
            continue
        a, b = max(im["y"], y0), min(im["y"] + im["h"], y1)
        if b > a:
            spans.append((a, b))
    if not spans:
        return 0.0
    spans.sort()
    merged = 0
    cur_a, cur_b = spans[0]
    for a, b in spans[1:]:
        if a > cur_b:
            merged += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    merged += cur_b - cur_a
    return merged / (y1 - y0) if y1 > y0 else 0.0


def analyze_bands(
    dom: list[dict], images: list[dict], page_w: int, page_h: int, band_h: int = BAND_H
) -> list[Band]:
    """구간마다 두 가지를 독립적으로 묻는다: 글자가 있나 / 큰 그림이 있나.

    둘의 조합이 곧 판정이다. 둘 다 있으면 `both` — DOM으로 뼈대를 잡고
    같은 자리에 OCR도 돌린다 (히어로 배너처럼 글자가 픽셀에 구워진 경우).
    """
    bands: list[Band] = []
    for y0 in range(0, page_h, band_h):
        y1 = min(y0 + band_h, page_h)
        chars = sum(len(d["text"]) for d in dom if y0 <= d["y"] < y1)
        cover = _vertical_cover(images, y0, y1)
        has_text = chars >= TEXT_MIN
        has_image = cover >= COVER_T
        verdict = (
            "both" if (has_text and has_image)
            else "dom" if has_text
            else "ocr" if has_image
            else "empty"
        )
        bands.append(Band(y0, y1, chars, round(cover, 3), verdict))
    return bands


def merge_regions(bands: list[Band]) -> list[Region]:
    """연속된 같은 판정 구간을 잇는다. empty는 경계로 쓰되 영역에 넣지 않는다."""
    regions: list[Region] = []
    cur: list[Band] = []

    def flush() -> None:
        if not cur:
            return
        kind = cur[0].verdict
        y0, y1 = cur[0].y0, cur[-1].y1
        chars = sum(b.dom_chars for b in cur)
        cover = sum(b.image_cover for b in cur) / len(cur)
        if kind != "dom" and (y1 - y0) < MIN_REGION_H:
            cur.clear()
            return
        # 너무 길면 OCR/VLM이 감당할 크기로 쪼갠다
        span = y1 - y0
        if span > MAX_REGION_H:
            n = -(-span // MAX_REGION_H)  # 올림
            step = -(-span // n)
            for k in range(n):
                a = y0 + k * step
                b = min(a + step, y1)
                regions.append(Region(a, b, kind, chars // n, round(cover, 3)))
        else:
            regions.append(Region(y0, y1, kind, chars, round(cover, 3)))
        cur.clear()

    for b in bands:
        if b.verdict == "empty":
            flush()
            continue
        if cur and cur[-1].verdict != b.verdict:
            flush()
        cur.append(b)
    flush()
    return regions


def page_verdict(regions: list[Region], page_h: int) -> tuple[str, float]:
    """페이지 전체 성격과, 글자가 이미지에 갇힌 세로 비율.

    이 비율이 곧 "이 도메인에서 왜 OCR이 필요한가"의 증거다.
    """
    ocr_h = sum(r.height for r in regions if r.kind == "ocr")
    ratio = ocr_h / page_h if page_h else 0.0
    if ratio >= 0.5:
        return "image_page", round(ratio, 3)  # 통이미지 상세페이지
    if ratio >= 0.1:
        return "hybrid", round(ratio, 3)
    return "dom_page", round(ratio, 3)


def analyze_dir(ref_dir: Path, band_h: int = BAND_H) -> dict:
    """수집된 한 레퍼런스 폴더를 읽어 regions.json을 쓴다."""
    meta = json.loads((ref_dir / "meta.json").read_text(encoding="utf-8"))
    dom = json.loads((ref_dir / "dom.json").read_text(encoding="utf-8"))
    images = json.loads((ref_dir / "images.json").read_text(encoding="utf-8"))
    page_w, page_h = meta["page_width"], meta["page_height"]

    bands = analyze_bands(dom, images, page_w, page_h, band_h)
    regions = merge_regions(bands)
    verdict, ratio = page_verdict(regions, page_h)

    result = {
        "id": meta["id"],
        "page_width": page_w,
        "page_height": page_h,
        "band_h": band_h,
        "page_verdict": verdict,
        "image_locked_ratio": ratio,
        "dom_chars_total": sum(len(d["text"]) for d in dom),
        "dom_chars_in_ocr_regions": sum(
            r.dom_chars for r in regions if r.kind == "ocr"
        ),
        "n_ocr_regions": sum(1 for r in regions if r.kind == "ocr"),
        "n_both_regions": sum(1 for r in regions if r.kind == "both"),
        "n_dom_regions": sum(1 for r in regions if r.kind == "dom"),
        "regions": [asdict(r) for r in regions],
        "bands": [asdict(b) for b in bands],
    }
    (ref_dir / "regions.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return result
