"""엔진 비교 결과를 채점해 표로 만든다. 이 표가 모델 선정의 근거다."""

from __future__ import annotations

import json
from pathlib import Path

from . import config
from .labels import Baseline, normalize, recall


def load_engine_result(ref_id: str, engine: str) -> dict | None:
    p = config.DERIVED_DIR / "ocr" / f"{ref_id}__{engine}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def garbage_ratio(lines: list[dict], conf_floor: float = 0.3) -> float:
    """신뢰도가 바닥인 줄의 비율.

    한국어 OCR이 무너질 때는 빈손으로 돌아오지 않고 **그럴듯한 헛것**을 만들어
    온다. EasyOCR은 "쌀 우미 속 백려조 오냐 녹독자"를 신뢰도 0.04로 뱉었다.
    글자 수만 세면 이런 엔진이 1등을 한다. 그래서 따로 센다.

    다만 이 지표는 엔진끼리 공정하지 않다. PaddleOCR은 `drop_score=0.5`로
    낮은 것을 **내부에서 이미 버리고** 주므로 0%가 나오는 것이 당연하고,
    Apple Vision은 신뢰도를 0.3/0.5/1.0으로 뭉뚱그려 준다. 엔진에 기대지 않는
    지표가 따로 필요해서 아래 corroboration_rate를 만들었다.
    """
    scored = [ln for ln in lines if ln.get("conf") is not None]
    if not scored:
        return 0.0
    return sum(1 for ln in scored if ln["conf"] < conf_floor) / len(scored)


def _ngrams(lines: list[dict], n: int = 3) -> set[str]:
    grams: set[str] = set()
    for ln in lines:
        s = normalize(ln["text"])
        for i in range(len(s) - n + 1):
            grams.add(s[i : i + n])
    return grams


def corroboration_rate(lines: list[dict], others: list[list[dict]], n: int = 3) -> float:
    """다른 엔진도 같은 글자를 봤는가.

    엔진이 만들어낸 헛것은 대체로 그 엔진에만 나타난다. 반대로 실제로 이미지에
    박혀 있는 글자는 여러 엔진이 함께 읽는다. 신뢰도 값과 달리 이 지표는
    엔진의 내부 임계값에 좌우되지 않는다.

    줄 단위로 비교하면 안 된다. 엔진마다 글자를 끊는 단위가 달라서, 잘게 쪼개는
    엔진일수록 "남의 문장에 포함될" 확률이 높아진다 (PaddleOCR은 줄당 6자,
    Apple Vision은 13.6자였다). 그래서 **문자 3-gram** 집합으로 비교한다 —
    어디서 끊었든 같은 글자를 읽었으면 같은 3-gram이 나온다.
    """
    mine = _ngrams(lines, n)
    if not mine:
        return 0.0
    theirs: set[str] = set()
    for o in others:
        theirs |= _ngrams(o, n)
    if not theirs:
        return 0.0
    return len(mine & theirs) / len(mine)


def score(ref_ids: list[str], engines: list[str], baselines: dict[str, Baseline]) -> dict:
    # 교차 검증을 위해 모든 엔진 결과를 미리 읽어둔다
    cache = {
        (r, e): load_engine_result(r, e) for r in ref_ids for e in engines
    }

    per_engine: dict[str, dict] = {}
    for engine in engines:
        hit = total = chars = lines_n = 0
        seconds = 0.0
        garbage: list[float] = []
        corrob: list[float] = []
        missing_refs: list[str] = []
        detail: dict[str, dict] = {}
        for ref_id in ref_ids:
            res = cache[(ref_id, engine)]
            if res is None:
                missing_refs.append(ref_id)
                continue
            others = [
                cache[(ref_id, e)]["lines"]
                for e in engines
                if e != engine and cache[(ref_id, e)] is not None
            ]
            corrob.append(corroboration_rate(res["lines"], others))
            texts = [ln["text"] for ln in res["lines"]]
            truth = baselines[ref_id].copy_lines if ref_id in baselines else []
            r = recall(truth, texts)
            hit += r["hit"]
            total += r["total"]
            chars += res["chars"]
            lines_n += res["n_lines"]
            seconds += res["seconds"]
            garbage.append(garbage_ratio(res["lines"]))
            detail[ref_id] = {
                "recall": r["recall"],
                "hit": r["hit"],
                "total": r["total"],
                "misses": r["misses"],
                "chars": res["chars"],
                "seconds": res["seconds"],
                "errors": res["errors"],
            }
        per_engine[engine] = {
            "recall": hit / total if total else 0.0,
            "hit": hit,
            "total": total,
            "chars": chars,
            "lines": lines_n,
            "seconds": round(seconds, 1),
            "garbage_ratio": round(sum(garbage) / len(garbage), 3) if garbage else None,
            "corroboration": round(sum(corrob) / len(corrob), 3) if corrob else None,
            "missing_refs": missing_refs,
            "per_ref": detail,
        }
    return per_engine


def human_summary(baselines: dict[str, Baseline], ref_ids: list[str]) -> dict:
    picked = [baselines[r] for r in ref_ids if r in baselines]
    minutes = sum(b.minutes or 0 for b in picked)
    return {
        "minutes": minutes,
        "seconds": minutes * 60,
        "copy_lines": sum(len(b.copy_lines) for b in picked),
        "chars": sum(len(normalize(c)) for b in picked for c in b.copy_lines),
    }


def render_table(per_engine: dict, human: dict) -> str:
    out = [
        "",
        f"{'엔진':<15}{'재현율':>12}{'추출 글자':>11}{'교차확인':>9}{'저신뢰':>8}{'소요':>9}{'사람 대비':>10}",
        "─" * 75,
    ]
    for name, e in sorted(per_engine.items(), key=lambda kv: (-kv[1]["recall"], kv[1]["seconds"])):
        if e["missing_refs"]:
            out.append(f"{name:<15}  (미실행: {', '.join(e['missing_refs'])})")
            continue
        g = f"{e['garbage_ratio']:.0%}" if e["garbage_ratio"] is not None else "—"
        c = f"{e['corroboration']:.0%}" if e["corroboration"] is not None else "—"
        speed = human["seconds"] / e["seconds"] if e["seconds"] else 0
        out.append(
            f"{name:<15}{e['hit']}/{e['total']} ({e['recall']:>4.0%}){e['chars']:>10,}자"
            f"{c:>9}{g:>8}{e['seconds']:>8.0f}s{speed:>9.0f}배"
        )
    out += [
        "─" * 75,
        f"{'사람(수작업)':<15}{human['copy_lines']}/{human['copy_lines']} (100%)"
        f"{human['chars']:>10,}자{'—':>9}{'—':>8}{human['seconds']:>8.0f}s{1:>9.0f}배",
        "",
        "재현율   = 사람이 손으로 적어둔 카피를 기계가 놓치지 않은 비율",
        "교차확인 = 그 줄을 다른 엔진도 읽었는가 (엔진 임계값에 좌우되지 않는 지표)",
        "저신뢰   = 신뢰도 0.3 미만인 줄의 비율 (엔진마다 기준이 달라 참고용)",
    ]
    return "\n".join(out)


def main(ref_ids: list[str], engines: list[str]) -> str:
    from .labels import load_all

    baselines = load_all(Path(config.REPO_ROOT / "data" / "baseline"))
    per_engine = score(ref_ids, engines, baselines)
    human = human_summary(baselines, ref_ids)
    config.DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    (config.DERIVED_DIR / "ocr_comparison.json").write_text(
        json.dumps({"engines": per_engine, "human": human}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return render_table(per_engine, human)
