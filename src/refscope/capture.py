"""레퍼런스 페이지 수집.

풀페이지 스크린샷 하나만 뜨는 게 아니라, 다음 단계가 "어느 눈으로 읽을지"를
고를 수 있도록 세 가지를 함께 남긴다.

  page.png    — 풀페이지 스크린샷 (OCR·VLM 입력)
  dom.json    — DOM 텍스트 노드 + 계산된 스타일 + 문서 좌표 (무손실 경로)
  images.json — 이미지 요소의 위치·크기 (통이미지 판정 근거)

수집 원칙(문제 정의서 8절)을 코드로 못 박아 둔다.
  - robots.txt를 확인하고, 막혀 있으면 건너뛴다
  - 요청 사이에 간격을 둔다
  - 페이지의 어떤 것도 클릭하지 않는다. 팝업은 CSS로 가리기만 한다
"""

from __future__ import annotations

import json
import time
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
)

VIEWPORT = {"width": 1440, "height": 900}

# 클릭하지 않는다. 화면을 가리는 레이어만 CSS로 숨긴다.
HIDE_OVERLAYS_CSS = """
[class*="cookie" i], [id*="cookie" i],
[class*="popup" i], [id*="popup" i],
[class*="modal" i], [id*="modal" i],
[class*="layer-pop" i], [class*="floating" i],
[class*="channel-talk" i], [id*="ch-plugin"],
[class*="consent" i], [id*="consent" i] { display: none !important; }
html { scroll-behavior: auto !important; }
* { animation: none !important; transition: none !important; }
"""

# 문서 좌표계로 텍스트 노드를 훑는다. getBoundingClientRect는 뷰포트 기준이라
# 스크롤 오프셋을 더해 문서 절대좌표로 바꾼다.
EXTRACT_DOM_JS = r"""
() => {
  const sx = window.scrollX, sy = window.scrollY;
  const out = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = walker.nextNode())) {
    const raw = n.nodeValue;
    if (!raw) continue;
    const text = raw.replace(/\s+/g, " ").trim();
    if (!text) continue;
    const el = n.parentElement;
    if (!el) continue;
    const tag = el.tagName.toLowerCase();
    if (tag === "script" || tag === "style" || tag === "noscript") continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === "hidden" || cs.display === "none" || parseFloat(cs.opacity) === 0) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue;
    out.push({
      text,
      tag,
      x: Math.round(r.left + sx), y: Math.round(r.top + sy),
      w: Math.round(r.width), h: Math.round(r.height),
      font_size: parseFloat(cs.fontSize) || null,
      font_weight: cs.fontWeight,
      font_family: cs.fontFamily,
      color: cs.color,
      background_color: cs.backgroundColor,
    });
  }
  return out;
}
"""

EXTRACT_IMAGES_JS = r"""
() => {
  const sx = window.scrollX, sy = window.scrollY;
  const out = [];
  for (const img of document.querySelectorAll("img")) {
    const r = img.getBoundingClientRect();
    if (r.width < 40 || r.height < 40) continue;
    out.push({
      kind: "img",
      src: img.currentSrc || img.src || null,
      alt: img.alt || "",
      x: Math.round(r.left + sx), y: Math.round(r.top + sy),
      w: Math.round(r.width), h: Math.round(r.height),
      natural_w: img.naturalWidth || null,
      natural_h: img.naturalHeight || null,
    });
  }
  for (const el of document.querySelectorAll("*")) {
    const bg = getComputedStyle(el).backgroundImage;
    if (!bg || bg === "none" || !bg.includes("url(")) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 120 || r.height < 120) continue;
    const m = bg.match(/url\(["']?([^"')]+)["']?\)/);
    out.push({
      kind: "background",
      src: m ? m[1] : null,
      alt: "",
      x: Math.round(r.left + sx), y: Math.round(r.top + sy),
      w: Math.round(r.width), h: Math.round(r.height),
      natural_w: null, natural_h: null,
    });
  }
  return out;
}
"""


@dataclass
class CaptureResult:
    ref_id: str
    url: str
    ok: bool
    reason: str = ""
    out_dir: Path | None = None
    page_height: int = 0
    dom_chars: int = 0
    long_image_area: int = 0
    verdict: str = ""
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)


_ROBOTS_CACHE: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def robots_allows(url: str, user_agent: str = "*") -> tuple[bool, str]:
    """robots.txt가 이 URL 수집을 허용하는지. 읽을 수 없으면 허용으로 본다."""
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin not in _ROBOTS_CACHE:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(urljoin(origin, "/robots.txt"))
        try:
            rp.read()
        except Exception:
            rp = None
        _ROBOTS_CACHE[origin] = rp
    rp = _ROBOTS_CACHE[origin]
    if rp is None:
        return True, "robots.txt를 읽지 못함 — 허용으로 간주"
    return bool(rp.can_fetch(user_agent, url)), "robots.txt 확인"


def _autoscroll(page, step: int = 700, pause_ms: int = 220, max_steps: int = 400) -> list[str]:
    """지연 로딩 이미지를 깨우려고 끝까지 내렸다가 맨 위로 돌아온다."""
    warnings: list[str] = []
    last_height, stable = 0, 0
    for i in range(max_steps):
        page.mouse.wheel(0, step)
        page.wait_for_timeout(pause_ms)
        height = page.evaluate("document.body.scrollHeight")
        pos = page.evaluate("window.scrollY + window.innerHeight")
        if height == last_height and pos >= height - 5:
            stable += 1
            if stable >= 2:
                break
        else:
            stable = 0
        last_height = height
        if i == max_steps - 1:
            warnings.append(f"스크롤 {max_steps}회로도 바닥에 닿지 못함 — 무한 스크롤 의심")
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(400)
    return warnings


def _screenshot(page, dest: Path) -> list[str]:
    """풀페이지 캡처. 너무 긴 페이지는 조각내어 이어 붙인다."""
    warnings: list[str] = []
    try:
        page.screenshot(path=str(dest), full_page=True, timeout=120_000)
        return warnings
    except (PlaywrightError, PlaywrightTimeout) as e:
        warnings.append(f"풀페이지 캡처 실패 → 분할 캡처로 전환 ({type(e).__name__})")

    from PIL import Image

    vh = VIEWPORT["height"]
    total = page.evaluate("document.body.scrollHeight")
    width = page.evaluate("document.body.scrollWidth")
    tiles, y = [], 0
    while y < total:
        page.evaluate(f"window.scrollTo(0, {y})")
        page.wait_for_timeout(200)
        buf = page.screenshot(timeout=60_000)
        tiles.append((y, buf))
        y += vh
    canvas = Image.new("RGB", (width, total), "white")
    for offset, buf in tiles:
        import io

        tile = Image.open(io.BytesIO(buf)).convert("RGB")
        canvas.paste(tile, (0, min(offset, max(total - tile.height, 0))))
    canvas.save(dest)
    return warnings


# 오류 페이지를 조용히 삼키지 않기 위한 방어.
# 처음 돌렸을 때 404 페이지 두 개를 정상 수집물로 받아들였다. 리서치 도구가
# 오류 화면을 데이터로 세면 "이 브랜드는 카피가 없다" 같은 헛된 결론이 나온다.
ERROR_PAGE_HINTS = (
    "페이지를 찾을 수 없",
    "찾을 수 없음",
    "사라졌거나 다른 페이지로",
    "주소를 다시 확인",
    "존재하지 않는 상품",
    "잘못된 접근",
    "page not found",
    "404 not found",
)
THIN_PAGE_H = 1800  # 이보다 짧으면서 오류 문구가 있으면 오류 페이지로 본다


def looks_like_error_page(title: str, dom: list[dict], page_h: int) -> str:
    """오류 페이지로 의심되면 그 이유를, 아니면 빈 문자열을 준다."""
    haystack = (title + " " + " ".join(d["text"] for d in dom[:60])).lower()
    for hint in ERROR_PAGE_HINTS:
        if hint.lower() in haystack:
            return f"오류 페이지 문구 감지: {hint!r}"
    if page_h <= 1000 and sum(len(d["text"]) for d in dom) < 200:
        return f"내용이 거의 없음 ({page_h}px, DOM 200자 미만)"
    return ""


def capture_one(browser, ref: dict, out_root: Path, timeout_ms: int = 60_000) -> CaptureResult:
    ref_id, url = ref["id"], ref["url"]
    started = time.perf_counter()
    res = CaptureResult(ref_id=ref_id, url=url, ok=False)

    allowed, why = robots_allows(url, UA)
    if not allowed:
        res.reason = f"robots.txt가 수집을 금지함 ({why})"
        return res

    out_dir = out_root / ref_id
    out_dir.mkdir(parents=True, exist_ok=True)
    res.out_dir = out_dir

    context = browser.new_context(
        viewport=VIEWPORT,
        user_agent=UA,
        locale="ko-KR",
        timezone_id="Asia/Seoul",
        device_scale_factor=1,
    )
    page = context.new_page()
    try:
        response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        status = response.status if response else 0
        if status >= 400:
            res.reason = f"HTTP {status} — 수집하지 않음"
            return res
        try:
            page.wait_for_load_state("networkidle", timeout=15_000)
        except PlaywrightTimeout:
            res.warnings.append("networkidle 미도달 — 배너·트래커가 계속 도는 사이트")
        page.add_style_tag(content=HIDE_OVERLAYS_CSS)
        page.keyboard.press("Escape")  # 클릭 대신 Esc — 아무것도 동의하지 않는다
        page.wait_for_timeout(300)

        res.warnings += _autoscroll(page)
        page.add_style_tag(content=HIDE_OVERLAYS_CSS)  # 스크롤 중 새로 뜬 레이어 대비

        dom = page.evaluate(EXTRACT_DOM_JS)
        images = page.evaluate(EXTRACT_IMAGES_JS)
        height = page.evaluate("document.body.scrollHeight")
        width = page.evaluate("document.body.scrollWidth")
        title = page.title()

        # HTTP 200으로 위장한 오류 페이지가 흔하다 (카페24·Shopify 모두 그렇다)
        if height <= THIN_PAGE_H:
            why_error = looks_like_error_page(title, dom, height)
            if why_error:
                res.reason = f"{why_error} — URL을 확인하세요"
                return res

        res.warnings += _screenshot(page, out_dir / "page.png")

        dom_chars = sum(len(d["text"]) for d in dom)
        long_area = sum(
            i["w"] * i["h"] for i in images if i["h"] >= 400 and i["h"] >= i["w"] * 0.8
        )

        (out_dir / "dom.json").write_text(
            json.dumps(dom, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (out_dir / "images.json").write_text(
            json.dumps(images, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (out_dir / "meta.json").write_text(
            json.dumps(
                {
                    "id": ref_id,
                    "name": ref.get("name", ref_id),
                    "group": ref.get("group", ""),
                    "url": url,
                    "http_status": status,
                    "title": title,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "viewport": VIEWPORT,
                    "page_width": width,
                    "page_height": height,
                    "dom_text_nodes": len(dom),
                    "dom_chars": dom_chars,
                    "image_elements": len(images),
                    "long_image_area": long_area,
                    "robots": why,
                    "warnings": res.warnings,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )

        res.ok = True
        res.page_height = height
        res.dom_chars = dom_chars
        res.long_image_area = long_area
    except Exception as e:  # 한 사이트가 죽어도 나머지는 계속 간다
        res.reason = f"{type(e).__name__}: {e}"
    finally:
        context.close()
        res.elapsed_s = time.perf_counter() - started
    return res


def capture_all(
    refs: list[dict], out_root: Path, delay_s: float = 2.0, headless: bool = True
) -> list[CaptureResult]:
    results: list[CaptureResult] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        try:
            for i, ref in enumerate(refs):
                r = capture_one(browser, ref, out_root)
                results.append(r)
                mark = "✓" if r.ok else "✗"
                detail = (
                    f"{r.page_height:>6}px  DOM {r.dom_chars:>6}자"
                    if r.ok
                    else r.reason[:70]
                )
                print(f"  {mark} {ref['id']:<14} {detail}  ({r.elapsed_s:.1f}s)")
                for w in r.warnings:
                    print(f"      ! {w}")
                if i < len(refs) - 1:
                    time.sleep(delay_s)  # 상대 서버에 대한 예의
        finally:
            browser.close()
    return results
