"""비교군 A / B / C 를 같은 자로 채점한다 (문제 정의서 6절).

  A 수작업      — 사람이 손으로 정리 (data/baseline/*.md). 정답 노릇도 겸한다
  B 순진한 AI   — 페이지를 VLM에 통째로 던지고 "정리해줘"
  C refscope    — 구간 판정 → 도구 갈아 끼우기 → 구조화

B와 C는 **같은 모델(qwen2.5vl:7b)** 을 쓴다. 다른 것은 무엇을 어떻게 보여주고
무엇을 시키느냐뿐이다. 그래서 B와 C의 차이는 곧 **설계가 만든 차이**다.

재는 것 넷:
  카피 재현율   — 사람이 적어둔 카피를 잡았는가
  카피 근거율   — 뱉은 카피가 실제로 페이지에 있는 글자인가 (지어내지 않았는가)
  컬러 정확도   — 사람이 스포이드로 찍은 색과의 CIEDE2000 색차
  소요 시간
"""

from __future__ import annotations

import json

from . import config
from .labels import load_all, normalize, recall
from .palette import score_palette
from .palette import Swatch

GROUND_ENGINE = "apple_vision"  # 근거율 판정에 쓰는 '페이지에 실제로 있는 글자'


def _page_text(ref_id: str) -> str:
    """그 페이지에서 실제로 읽힌 글자 전부 (DOM + OCR). 근거 판정의 기준."""
    parts: list[str] = []
    dom = config.REFS_DIR / ref_id / "dom.json"
    if dom.exists():
        parts += [d["text"] for d in json.loads(dom.read_text(encoding="utf-8"))]
    ocr = config.DERIVED_DIR / "ocr" / f"{ref_id}__{GROUND_ENGINE}.json"
    if ocr.exists():
        parts += [x["text"] for x in json.loads(ocr.read_text(encoding="utf-8"))["lines"]]
    return normalize(" ".join(parts))


def grounding_rate(lines: list[str], page_blob: str, n: int = 5) -> float:
    """뱉은 카피가 페이지에 실제로 박혀 있는 글자인가.

    문자 n-gram이 페이지 텍스트에 있는지로 본다. 지어낸 문장은 그럴듯해도
    페이지에 없다. 이 지표가 낮으면 "자신 있게 틀린" 출력이라는 뜻이다.
    """
    if not lines:
        return 0.0
    ok = 0
    for ln in lines:
        s = normalize(ln)
        if len(s) < n:
            ok += 1 if s and s in page_blob else 0
            continue
        grams = [s[i : i + n] for i in range(len(s) - n + 1)]
        hit = sum(1 for g in grams if g in page_blob)
        ok += 1 if hit / len(grams) >= 0.6 else 0
    return ok / len(lines)


def run(ref_ids: list[str]) -> dict:
    baselines = load_all(config.REPO_ROOT / "data" / "baseline")
    cards = {
        c["id"]: c
        for c in json.loads((config.DERIVED_DIR / "cards.json").read_text(encoding="utf-8"))
    }
    naive = json.loads((config.DERIVED_DIR / "naive_baseline.json").read_text(encoding="utf-8"))

    rows: dict[str, dict] = {}
    for rid in ref_ids:
        b = baselines[rid]
        blob = _page_text(rid)
        truth_copy = b.copy_lines

        # B — 순진한 AI
        nb = naive[rid]
        b_copy = nb["copy"]
        b_colors = nb["colors"]

        # C — refscope.
        # B가 "최대 10줄"을 요구받았으므로 공정하게 상위 10줄로 맞춰 비교한다.
        # 다만 C는 실제로 훨씬 많이 가지고 있으므로 전체 기준 값도 함께 낸다.
        card = cards[rid]
        c_all = [x["text"] for x in card["copy_lines"]]
        c_copy = [x["text"] for x in sorted(card["copy_lines"], key=lambda v: -v["size"])[:10]]
        c_colors = [p["hex"] for p in card["palette"] if p["role"] == "accent"]

        rows[rid] = {
            "A": {
                "seconds": (b.minutes or 0) * 60,
                "copy_n": len(truth_copy),
                "copy_recall": 1.0,
                "grounding": 1.0,
                "color": score_palette(b.brand_colors, [Swatch(h, 0, "accent") for h in b.brand_colors]),
            },
            "B": {
                "seconds": nb["seconds"],
                "copy_n": len(b_copy),
                "copy_recall": recall(truth_copy, b_copy)["recall"],
                "grounding": grounding_rate(b_copy, blob),
                "leaked": len(nb.get("leaked", [])),
                "color": score_palette(b.brand_colors, [Swatch(h, 0, "accent") for h in b_colors]),
            },
            "C": {
                "seconds": card["seconds"],
                "copy_n": len(card["copy_lines"]),
                "copy_recall": recall(truth_copy, c_copy)["recall"],
                "copy_recall_all": recall(truth_copy, c_all)["recall"],
                "grounding": grounding_rate(c_copy, blob),
                "color": score_palette(
                    b.brand_colors, [Swatch(h, 0, "accent") for h in c_colors]
                ),
            },
        }

    # 합산
    totals: dict[str, dict] = {}
    for g in ("A", "B", "C"):
        secs = sum(rows[r][g]["seconds"] for r in ref_ids)
        totals[g] = {
            "seconds": round(secs, 1),
            "copy_n": sum(rows[r][g]["copy_n"] for r in ref_ids),
            "copy_recall": sum(rows[r][g]["copy_recall"] for r in ref_ids) / len(ref_ids),
            "grounding": sum(rows[r][g]["grounding"] for r in ref_ids) / len(ref_ids),
            "color_hit": sum(rows[r][g]["color"]["matched"] for r in ref_ids),
            "color_total": sum(rows[r][g]["color"]["total"] for r in ref_ids),
        }
        if g == "C":
            totals[g]["copy_recall_all"] = sum(
                rows[r][g]["copy_recall_all"] for r in ref_ids
            ) / len(ref_ids)
        if g == "B":
            totals[g]["leaked"] = sum(rows[r][g]["leaked"] for r in ref_ids)
    result = {"per_ref": rows, "totals": totals}
    (config.DERIVED_DIR / "compare_abc.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return result


LABEL = {
    "A": "A 수작업",
    "B": "B 순진한 AI",
    "C": "C refscope",
}


def render(result: dict) -> str:
    t = result["totals"]
    out = [
        "",
        f"{'비교군':<13}{'소요':>8}{'카피':>7}{'재현율':>8}{'근거율':>8}{'컬러':>8}",
        "─" * 52,
    ]
    for g in ("A", "B", "C"):
        r = t[g]
        color = f"{r['color_hit']}/{r['color_total']}"
        out.append(
            f"{LABEL[g]:<13}{r['seconds']:>7.0f}s{r['copy_n']:>6}줄"
            f"{r['copy_recall']:>8.0%}{r['grounding']:>8.0%}{color:>8}"
        )
    out += [
        "─" * 52,
        "",
        "재현율 = 사람이 적어둔 카피를 잡았는가. A는 정의상 100%.",
        "         B는 최대 10줄을 요구받았으므로 C도 상위 10줄로 맞춰 쟀다.",
        f"         C를 전체 {t['C']['copy_n']}줄 기준으로 재면 {t['C']['copy_recall_all']:.0%}.",
        "근거율 = 뱉은 카피가 실제로 페이지에 박혀 있는 글자인가.",
        f"         B는 {1 - t['B']['grounding']:.0%}가 페이지에 없는 문장이다 (지어냄).",
        "컬러   = 사람이 스포이드로 찍은 3색 중 CIEDE2000 ΔE≤10 으로 맞춘 수.",
    ]
    if t["B"].get("leaked"):
        out.append(f"\n덧: B는 프롬프트 문장을 카피로 되뱉은 것이 {t['B']['leaked']}건 있었다.")
    return "\n".join(out)


if __name__ == "__main__":
    ids = ["gyodong_manwol", "hanpoom_99flower", "hangamall_dao"]
    print(render(run(ids)))
