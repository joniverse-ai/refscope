"""리서치 카드를 한 장의 HTML로 만든다.

디자이너가 실제로 쓰는 물건이 되려면 표가 아니라 **보이는 것**이어야 한다.
컬러는 스와치로, 카피는 크기 순으로, 구성은 세로 흐름으로 보여준다.

원본 스크린샷은 저장소에 올리지 않으므로 HTML도 이미지를 링크하지 않는다.
페이지를 보고 싶으면 원문 URL로 간다.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path

CSS = """
:root{--bg:#fbfaf8;--fg:#1c1a17;--mut:#77706a;--line:#e3ddd5;--card:#fff;--accent:#8a6a3f}
@media(prefers-color-scheme:dark){:root{--bg:#171513;--fg:#efe9e2;--mut:#9c948c;
--line:#332e29;--card:#201d1a;--accent:#c9a468}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.65 -apple-system,BlinkMacSystemFont,"Pretendard","Apple SD Gothic Neo",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:48px 24px 96px}
h1{font-size:30px;letter-spacing:-.02em;margin:0 0 6px}
.sub{color:var(--mut);margin:0 0 40px;font-size:14px}
.stats{display:flex;flex-wrap:wrap;gap:28px;padding:20px 24px;background:var(--card);
border:1px solid var(--line);border-radius:12px;margin-bottom:40px}
.stat b{display:block;font-size:26px;letter-spacing:-.02em}
.stat span{color:var(--mut);font-size:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:26px 28px;margin-bottom:22px}
.card h2{font-size:19px;margin:0 0 4px;letter-spacing:-.01em}
.card h2 a{color:inherit;text-decoration:none;border-bottom:1px solid var(--line)}
.meta{color:var(--mut);font-size:12.5px;margin-bottom:20px}
.tag{display:inline-block;padding:1px 8px;border:1px solid var(--line);border-radius:99px;
margin-right:6px;font-size:11.5px}
.locked{color:var(--accent);font-weight:600}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:28px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.sec{font-size:12px;letter-spacing:.06em;color:var(--mut);text-transform:uppercase;
margin:0 0 10px;font-weight:600}
.sw{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:6px}
.sw div{width:52px;height:52px;border-radius:8px;border:1px solid var(--line);
position:relative}
.sw div::after{content:attr(data-hex);position:absolute;bottom:-17px;left:0;
font-size:9.5px;color:var(--mut);font-family:ui-monospace,monospace}
.sw.neutral div{opacity:.45}
ol.flow{list-style:none;padding:0;margin:0;counter-reset:s}
ol.flow li{counter-increment:s;padding:6px 0 6px 30px;position:relative;
border-bottom:1px dashed var(--line);font-size:14px}
ol.flow li:last-child{border:0}
ol.flow li::before{content:counter(s);position:absolute;left:0;top:7px;width:20px;
height:20px;border-radius:50%;background:var(--accent);color:#fff;font-size:11px;
display:grid;place-items:center}
ol.flow small{color:var(--mut);font-size:11px;margin-left:6px}
ul.copy{list-style:none;padding:0;margin:0}
ul.copy li{padding:5px 0;border-bottom:1px dashed var(--line)}
ul.copy li:last-child{border:0}
ul.copy .src{font-size:9.5px;color:var(--mut);border:1px solid var(--line);
border-radius:3px;padding:0 4px;margin-left:8px;vertical-align:middle}
.tone li{margin-bottom:7px;font-size:14px}
.tone{list-style:none;padding:0;margin:0}
.err{color:#b4472e;font-size:12.5px;margin-top:14px}
footer{color:var(--mut);font-size:12px;margin-top:56px;border-top:1px solid var(--line);
padding-top:20px}
"""


def _swatches(palette: list[dict], role: str) -> str:
    items = [p for p in palette if p.get("role") == role]
    if not items:
        return ""
    cls = "sw neutral" if role == "neutral" else "sw"
    inner = "".join(
        f'<div style="background:{html.escape(p["hex"])}" data-hex="{html.escape(p["hex"])}"'
        f' title="{p["ratio"]:.1%}"></div>'
        for p in items
    )
    return f'<div class="{cls}">{inner}</div>'


def _card_html(c: dict) -> str:
    copy_top = sorted(c["copy_lines"], key=lambda x: -x["size"])[:10]
    copy_html = "".join(
        f'<li>{html.escape(x["text"])}<span class="src">{x["source"]}</span></li>'
        for x in copy_top
    ) or '<li style="color:var(--mut)">추출된 카피 없음</li>'

    flow = "".join(
        f'<li>{html.escape(s["name"])}<small>y {s["y"]:,}</small></li>'
        for s in c["sections"]
    ) or '<li style="color:var(--mut)">구조 미분류</li>'

    tone = "".join(f"<li>{html.escape(t)}</li>" for t in c["tone"]) or (
        '<li style="color:var(--mut)">톤 서술 없음</li>'
    )

    locked = c["image_locked_ratio"]
    errs = (
        f'<div class="err">⚠ {html.escape(" / ".join(c["errors"]))}</div>'
        if c.get("errors")
        else ""
    )
    return f"""<div class="card">
  <h2><a href="{html.escape(c["url"])}" target="_blank" rel="noopener">{html.escape(c["name"])}</a></h2>
  <div class="meta">
    <span class="tag">{html.escape(c["group"])}</span>
    <span class="tag">{html.escape(c["page_type"])}</span>
    {c["page_height"]:,}px ·
    글자가 이미지에 갇힌 비율 <span class="locked">{locked:.0%}</span> ·
    처리 {c["seconds"]:.0f}초
  </div>
  <div class="grid">
    <div>
      <p class="sec">브랜드 컬러</p>
      {_swatches(c["palette"], "accent")}
      <p class="sec" style="margin-top:26px">무채색 · 배경</p>
      {_swatches(c["palette"], "neutral")}
      <p class="sec" style="margin-top:30px">페이지 구성</p>
      <ol class="flow">{flow}</ol>
    </div>
    <div>
      <p class="sec">헤드라인 · 카피 (큰 글자 순)</p>
      <ul class="copy">{copy_html}</ul>
      <p class="sec" style="margin-top:26px">비주얼 인상 (VLM)</p>
      <ul class="tone">{tone}</ul>
    </div>
  </div>
  {errs}
</div>"""


def render(cards: list, out_path: Path, title: str = "레퍼런스 리서치") -> Path:
    data = [c if isinstance(c, dict) else asdict(c) for c in cards]
    total_px = sum(c["page_height"] for c in data)
    total_copy = sum(len(c["copy_lines"]) for c in data)
    total_s = sum(c["seconds"] for c in data)
    avg_locked = (
        sum(c["image_locked_ratio"] for c in data) / len(data) if data else 0
    )

    body = "".join(_card_html(c) for c in data)
    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body><div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="sub">refscope가 자동 생성 · 원본 스크린샷은 저장소에 포함되지 않습니다</p>
<div class="stats">
  <div class="stat"><b>{len(data)}</b><span>레퍼런스</span></div>
  <div class="stat"><b>{total_px:,}</b><span>훑은 세로 픽셀</span></div>
  <div class="stat"><b>{total_copy:,}</b><span>추출 카피 줄</span></div>
  <div class="stat"><b>{avg_locked:.0%}</b><span>평균 이미지 잠김</span></div>
  <div class="stat"><b>{total_s:.0f}초</b><span>총 처리 시간</span></div>
</div>
{body}
<footer>Main Quest 2 · 레퍼런스 리서치 자동 정리기 —
Apple Vision(OCR) + Qwen2.5-VL(톤·구성) + k-means(컬러) 로컬 실행</footer>
</div></body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def dump_cards(cards: list, out_path: Path) -> Path:
    data = [c if isinstance(c, dict) else asdict(c) for c in cards]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return out_path
