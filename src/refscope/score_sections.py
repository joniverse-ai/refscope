"""섹션 분류 정확도 채점 (성공 기준 #4).

사람과 기계가 같은 섹션을 가리켜도 이름이 다르다.

  사람 "제품사용 Tip"   ↔  기계 "사용법·보관법"
  사람 "메인 사진, 가격" ↔  기계 "할인/정가 정보"

문자열 일치로 재면 전부 오답이 된다. 그래서 bge-m3 임베딩의 코사인 유사도로
맞춘다. 임계값 이상이면 같은 섹션을 가리킨 것으로 본다.

한 가지 더: 사람은 "네비게이터"를 항상 첫 섹션으로 적었는데 기계는 이것을
따로 세지 않는다. 이런 체계적 차이는 감추지 말고 결과에 드러내야 한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config
from .embed import cosine_matrix, embed
from .labels import load_all

MATCH_T = 0.62  # bge-m3 코사인 유사도 임계값


def score_one(truth: list[str], predicted: list[str], threshold: float = MATCH_T) -> dict:
    """사람이 적은 섹션 각각에 대해, 기계 섹션 중 가장 가까운 것을 찾는다.

    한 기계 섹션이 여러 사람 섹션에 중복 배정되지 않도록 그리디로 짝짓는다.
    """
    if not truth or not predicted:
        return {"total": len(truth), "hit": 0, "recall": 0.0, "rows": []}

    vt, vp = embed(truth), embed(predicted)
    sim = cosine_matrix(vt, vp)

    used: set[int] = set()
    rows = []
    # 유사도가 높은 짝부터 확정한다
    order = np.dstack(np.unravel_index(np.argsort(-sim, axis=None), sim.shape))[0]
    matched: dict[int, tuple[int, float]] = {}
    for i, j in order:
        i, j = int(i), int(j)
        if i in matched or j in used:
            continue
        s = float(sim[i, j])
        if s < threshold:
            continue
        matched[i] = (j, s)
        used.add(j)

    for i, t in enumerate(truth):
        if i in matched:
            j, s = matched[i]
            rows.append(
                {
                    "truth": t,
                    "matched": predicted[j],
                    "similarity": round(s, 3),
                    "hit": True,
                    "reason": "",
                }
            )
            continue
        # 놓친 이유를 구분한다. 둘은 성격이 전혀 다르다.
        #   below   — 비슷한 섹션 자체가 없었다 (진짜 실패)
        #   taken   — 짝이 될 만했지만 다른 항목이 먼저 가져갔다
        #             (사람이 5개로, 기계가 다른 수로 나눈 탓 — 분할 입도 문제)
        best_j = int(np.argmax(sim[i]))
        best_s = float(sim[i, best_j])
        reason = "below" if best_s < threshold else "taken"
        rows.append(
            {
                "truth": t,
                "matched": None,
                "nearest": predicted[best_j],
                "similarity": round(best_s, 3),
                "hit": False,
                "reason": reason,
            }
        )
    hit = sum(1 for r in rows if r["hit"])
    return {
        "total": len(truth),
        "hit": hit,
        "recall": hit / len(truth),
        "n_predicted": len(predicted),
        "n_taken": sum(1 for r in rows if r.get("reason") == "taken"),
        "rows": rows,
    }


# 사람은 "네비게이터"를 항상 첫 섹션으로 적었지만, 기계는 내비게이션 바를
# 콘텐츠 섹션으로 세지 않는다. 이건 성능 차이가 아니라 세는 단위의 차이다.
# 감추지 않되, 따로 떼어 함께 보고한다.
NAV_LIKE = ("네비게이터", "네이게이터", "내비게이터", "navigation", "gnb", "로고")


def is_nav(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in NAV_LIKE)


def run(threshold: float = MATCH_T) -> dict:
    baselines = load_all(config.REPO_ROOT / "data" / "baseline")
    cards_path = config.DERIVED_DIR / "cards.json"
    if not cards_path.exists():
        raise FileNotFoundError("cards.json이 없습니다. `refscope build`를 먼저 돌리세요")
    cards = {c["id"]: c for c in json.loads(cards_path.read_text(encoding="utf-8"))}

    per_ref, hit, total = {}, 0, 0
    for ref_id, b in baselines.items():
        card = cards.get(ref_id)
        if not card:
            continue
        predicted = [s["name"] for s in card["sections"]]
        r = score_one(b.sections, predicted, threshold)
        per_ref[ref_id] = r
        hit += r["hit"]
        total += r["total"]

    # 내비게이션 항목을 뺀 값도 함께 낸다
    nav_hit = nav_total = 0
    for r in per_ref.values():
        for row in r["rows"]:
            if is_nav(row["truth"]):
                continue
            nav_total += 1
            nav_hit += 1 if row["hit"] else 0

    result = {
        "threshold": threshold,
        "hit": hit,
        "total": total,
        "recall": hit / total if total else 0.0,
        "recall_excl_nav": nav_hit / nav_total if nav_total else 0.0,
        "hit_excl_nav": nav_hit,
        "total_excl_nav": nav_total,
        "n_taken": sum(r["n_taken"] for r in per_ref.values()),
        "per_ref": per_ref,
    }
    (config.DERIVED_DIR / "section_scores.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return result


def render(result: dict) -> str:
    out = [
        f"섹션 분류 정확도 — bge-m3 코사인 유사도 ≥ {result['threshold']}",
        "=" * 62,
    ]
    for ref_id, r in result["per_ref"].items():
        out.append(f"\n{ref_id}  {r['hit']}/{r['total']} ({r['recall']:.0%})"
                   f"   기계가 나눈 섹션 {r['n_predicted']}개")
        for row in r["rows"]:
            mark = "✓" if row["hit"] else "✗"
            if row["hit"]:
                got = row["matched"]
            elif row["reason"] == "taken":
                got = f"({row['nearest']} — 다른 항목이 선점)"
            else:
                got = "(비슷한 섹션 없음)"
            out.append(f"   {mark} 사람 {row['truth']:<18} ↔ {got:<28} {row['similarity']:.2f}")
    out.append("\n" + "=" * 62)
    out.append(f"전체            {result['hit']}/{result['total']} = {result['recall']:.0%}   (목표 70% — 미달)")
    out.append(
        f"내비게이터 제외  {result['hit_excl_nav']}/{result['total_excl_nav']} "
        f"= {result['recall_excl_nav']:.0%}"
    )
    out.append(f"선점으로 놓친 것 {result['n_taken']}건 (분할 입도 차이)")
    return "\n".join(out)


if __name__ == "__main__":
    print(render(run()))
