"""카테고리 공통 메시지 패턴 — 리서치의 진짜 결론.

카드 12장은 아직 재료다. 디자이너가 실제로 답해야 하는 질문은 하나다.

    "이 바닥은 지금 소비자에게 무슨 말을 하고 있고, 아무도 안 하는 말은 무엇인가?"

브랜드별 카피를 전부 한 의미 공간에 올려 묶으면, 어떤 말이 이미 붐비는지와
어디가 비어 있는지가 보인다. 단정이 설 자리를 찾는 것이 이 단계의 목적이다.

묶음마다 이름은 LLM이 짓는다. 나누는 일(클러스터링)은 임베딩이 하고,
이름 짓는 일만 모델에게 맡긴다 — 이 저장소에서 반복되는 원칙이다.
"""

from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import config
from .embed import embed
from .synthesize import ask_ollama

MIN_LEN = 6  # 너무 짧은 카피는 의미가 안 잡힌다
MAX_LEN = 60


CLUSTER_NAME_PROMPT = """아래는 한국 프리미엄 선물·전통 디저트 브랜드들의 상세페이지에서 뽑은
카피를 의미가 비슷한 것끼리 묶은 것입니다. 각 묶음에 **소구점 이름**을 붙이세요.

좋은 이름의 예: 전통·장인정신, 건강·무첨가, 선물 적합성, 포장·격식, 가격·혜택,
배송·신선도, 원산지·재료, 감사·마음 전달

규칙:
- 정확히 {n}줄만 출력합니다.
- 형식은 "번호 | 이름" 입니다. 이름은 10자 이내의 명사구.
- 설명을 붙이지 마세요.

{blocks}"""


def gather_copy(cards: list[dict], groups: tuple[str, ...] = ("competitor", "tone")) -> list[dict]:
    """분석 대상 카피를 모은다. 내 브랜드(own)는 빼고 본다 — 비교 대상이니까."""
    rows = []
    seen: set[str] = set()
    for c in cards:
        if c["group"] not in groups:
            continue
        for x in c["copy_lines"]:
            t = re.sub(r"\s+", " ", x["text"]).strip()
            if not (MIN_LEN <= len(t) <= MAX_LEN):
                continue
            key = re.sub(r"\W", "", t)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": t, "brand": c["name"], "brand_id": c["id"]})
    return rows


def cluster(rows: list[dict], k: int = 8, seed: int = 0) -> list[dict]:
    from sklearn.cluster import KMeans

    vecs = embed([r["text"] for r in rows])
    k = min(k, max(2, len(rows) // 4))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(vecs)

    out = []
    for ci in range(k):
        idx = [i for i, lab in enumerate(labels) if lab == ci]
        if not idx:
            continue
        # 중심에 가까운 순서로 대표 카피를 뽑는다
        center = km.cluster_centers_[ci]
        dist = [(float(np.linalg.norm(vecs[i] - center)), i) for i in idx]
        dist.sort()
        members = [rows[i] for _, i in dist]
        brands = defaultdict(int)
        for m in members:
            brands[m["brand"]] += 1
        out.append(
            {
                "index": ci,
                "size": len(members),
                "brands": dict(sorted(brands.items(), key=lambda kv: -kv[1])),
                "n_brands": len(brands),
                "examples": [m["text"] for m in members[:6]],
                "members": members,
            }
        )
    out.sort(key=lambda c: -c["size"])
    return out


def name_clusters(clusters: list[dict], model: str = "qwen2.5vl:7b") -> None:
    blocks = "\n\n".join(
        f"[{i + 1}]\n" + "\n".join(f"  {t}" for t in c["examples"][:5])
        for i, c in enumerate(clusters)
    )
    out, err = ask_ollama(
        model, CLUSTER_NAME_PROMPT.format(n=len(clusters), blocks=blocks)
    )
    names: dict[int, str] = {}
    if not err:
        for line in out.splitlines():
            m = re.match(r"\s*\[?(\d+)\]?\s*\|\s*(.+?)\s*$", line)
            if m:
                names[int(m.group(1))] = m.group(2)[:16]
    for i, c in enumerate(clusters, start=1):
        c["name"] = names.get(i, f"묶음 {i}")
        c["name_error"] = err


def find_gaps(clusters: list[dict], own_cards: list[dict]) -> list[dict]:
    """내 브랜드가 아직 하지 않는 말은 무엇인가.

    단정의 카피를 같은 공간에 올려, 각 소구점 묶음과 얼마나 가까운지 잰다.
    가까운 묶음이 없으면 그 소구점은 내가 비워둔 자리다.
    """
    own_lines = [
        re.sub(r"\s+", " ", x["text"]).strip()
        for c in own_cards
        for x in c["copy_lines"]
        if MIN_LEN <= len(x["text"].strip()) <= MAX_LEN
    ]
    if not own_lines or not clusters:
        return []
    own_vecs = embed(own_lines)
    gaps = []
    for c in clusters:
        cv = embed(c["examples"][:5])
        sim = float((own_vecs @ cv.T).max())
        gaps.append(
            {
                "name": c["name"],
                "size": c["size"],
                "n_brands": c["n_brands"],
                "my_closest": round(sim, 3),
                "occupied": sim >= 0.60,
            }
        )
    return sorted(gaps, key=lambda g: g["my_closest"])


CSS = """
:root{--bg:#fbfaf8;--fg:#1c1a17;--mut:#77706a;--line:#e3ddd5;--card:#fff;--accent:#8a6a3f}
@media(prefers-color-scheme:dark){:root{--bg:#171513;--fg:#efe9e2;--mut:#9c948c;
--line:#332e29;--card:#201d1a;--accent:#c9a468}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.65 -apple-system,BlinkMacSystemFont,"Pretendard","Apple SD Gothic Neo",sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:48px 24px 96px}
h1{font-size:30px;letter-spacing:-.02em;margin:0 0 6px}
.sub{color:var(--mut);margin:0 0 40px;font-size:14px}
h2{font-size:18px;margin:44px 0 14px}
.cl{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:18px 22px;margin-bottom:14px}
.cl h3{margin:0 0 4px;font-size:16px}
.cl .m{color:var(--mut);font-size:12.5px;margin-bottom:10px}
.bar{height:6px;background:var(--line);border-radius:99px;overflow:hidden;margin-bottom:12px}
.bar i{display:block;height:100%;background:var(--accent)}
ul{margin:0;padding-left:18px}
li{font-size:13.5px;padding:1px 0}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.open{color:var(--accent);font-weight:600}
footer{color:var(--mut);font-size:12px;margin-top:56px;border-top:1px solid var(--line);
padding-top:20px}
"""


def render(clusters: list[dict], gaps: list[dict], out_path: Path, n_lines: int) -> Path:
    biggest = max((c["size"] for c in clusters), default=1)
    cl_html = "".join(
        f"""<div class="cl"><h3>{html.escape(c["name"])}</h3>
<div class="m">카피 {c["size"]}줄 · 브랜드 {c["n_brands"]}곳
({html.escape(", ".join(list(c["brands"])[:4]))})</div>
<div class="bar"><i style="width:{c["size"] / biggest * 100:.0f}%"></i></div>
<ul>{"".join(f"<li>{html.escape(t)}</li>" for t in c["examples"][:5])}</ul></div>"""
        for c in clusters
    )
    gap_rows = "".join(
        f"""<tr><td>{html.escape(g["name"])}</td><td>{g["size"]}줄 / {g["n_brands"]}곳</td>
<td>{g["my_closest"]:.2f}</td>
<td>{"이미 하고 있음" if g["occupied"] else '<span class="open">비어 있음</span>'}</td></tr>"""
        for g in gaps
    )
    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>카테고리 메시지 패턴</title><style>{CSS}</style></head><body><div class="wrap">
<h1>이 카테고리는 지금 무슨 말을 하고 있나</h1>
<p class="sub">경쟁·톤 레퍼런스에서 뽑은 카피 {n_lines}줄을 bge-m3로 임베딩해 묶었습니다</p>
<h2>소구점 묶음</h2>
{cl_html}
<h2>단정이 비워둔 자리</h2>
<p class="sub" style="margin-bottom:14px">단정의 카피를 같은 공간에 올려, 각 소구점과 얼마나 가까운지 쟀습니다.
가까운 것이 없으면 아직 하지 않은 말입니다.</p>
<table><tr><th>소구점</th><th>시장 점유</th><th>단정 최대 유사도</th><th></th></tr>
{gap_rows}</table>
<footer>Main Quest 2 · refscope — bge-m3 임베딩 + k-means, 전부 로컬 실행</footer>
</div></body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def run(k: int = 8) -> dict:
    cards = json.loads((config.DERIVED_DIR / "cards.json").read_text(encoding="utf-8"))
    rows = gather_copy(cards)
    clusters = cluster(rows, k=k)
    name_clusters(clusters)
    gaps = find_gaps(clusters, [c for c in cards if c["group"] == "own"])

    payload = {
        "n_lines": len(rows),
        "clusters": [{kk: v for kk, v in c.items() if kk != "members"} for c in clusters],
        "gaps": gaps,
    }
    (config.DERIVED_DIR / "patterns.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    render(clusters, gaps, config.OUT_DIR / "patterns.html", len(rows))
    return payload


if __name__ == "__main__":
    p = run()
    print(f"카피 {p['n_lines']}줄 → 묶음 {len(p['clusters'])}개\n")
    for c in p["clusters"]:
        print(f"  {c['name']:<14} {c['size']:>3}줄  브랜드 {c['n_brands']}곳")
        print(f"      예: {c['examples'][0][:52]}")
    print("\n단정이 비워둔 자리 (유사도 낮은 순):")
    for g in p["gaps"]:
        mark = "  " if g["occupied"] else "★ "
        print(f"  {mark}{g['name']:<14} 유사도 {g['my_closest']:.2f}  "
              f"{'이미 하고 있음' if g['occupied'] else '비어 있음'}")
