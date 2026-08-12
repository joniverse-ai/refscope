"""같은 조각을 여러 엔진에 나란히 돌려 재는 실험.

커리큘럼 4-6절 「같은 PDF를 네 가지 눈으로 읽기」를 이 도메인에 옮긴 것이다.
여기서 나온 표가 모델 선정의 근거가 된다.

재는 것은 세 가지다.
  재현율 — 사람이 손으로 적어둔 카피를 놓치지 않았는가 (정답 대비)
  발견량 — 사람보다 얼마나 더 읽어냈는가 (사람은 지쳐서 3줄만 적었다)
  속도   — 사이트 하나를 처리하는 데 실제로 몇 초 걸리는가
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import config
from .labels import Baseline, recall
from .ocr import OcrResult, get_engine


def read_crops(engine_name: str, ref_dir: Path) -> tuple[list[str], float, list[str]]:
    """한 레퍼런스의 모든 조각을 한 엔진으로 읽는다."""
    manifest_path = ref_dir / "crops" / "manifest.json"
    if not manifest_path.exists():
        return [], 0.0, [f"{ref_dir.name}: 조각이 없습니다 (`refscope crop` 먼저)"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    engine = get_engine(engine_name)
    lines: list[str] = []
    errors: list[str] = []
    t0 = time.perf_counter()
    for item in manifest:
        res: OcrResult = engine.read(ref_dir / "crops" / item["file"])
        if res.error:
            errors.append(f"{item['file']}: {res.error}")
            continue
        lines.extend(ln.text for ln in res.lines if ln.text.strip())
    return lines, time.perf_counter() - t0, errors


def compare(
    ref_ids: list[str], engine_names: list[str], baselines: dict[str, Baseline]
) -> dict:
    results: dict = {"engines": engine_names, "refs": {}}
    for ref_id in ref_ids:
        ref_dir = config.REFS_DIR / ref_id
        truth = baselines[ref_id].copy_lines if ref_id in baselines else []
        per_engine = {}
        for name in engine_names:
            lines, secs, errors = read_crops(name, ref_dir)
            score = recall(truth, lines) if truth else None
            per_engine[name] = {
                "lines": lines,
                "n_lines": len(lines),
                "chars": sum(len(x) for x in lines),
                "seconds": round(secs, 1),
                "errors": errors,
                "recall": score,
            }
            hit = f"{score['hit']}/{score['total']}" if score else "—"
            print(
                f"  {name:<14} {len(lines):>4}줄 {sum(len(x) for x in lines):>6}자 "
                f"{secs:>6.1f}s  정답 {hit}"
            )
            for e in errors[:2]:
                print(f"        ! {e[:90]}")
        results["refs"][ref_id] = {"truth": truth, "engines": per_engine}
    return results


def summarize(results: dict, baselines: dict[str, Baseline]) -> str:
    """엔진별 합산 표. 이 표가 모델 선정 문서에 그대로 들어간다."""
    engines = results["engines"]
    rows = []
    for name in engines:
        hit = total = chars = 0
        secs = 0.0
        for ref_id, r in results["refs"].items():
            e = r["engines"][name]
            chars += e["chars"]
            secs += e["seconds"]
            if e["recall"]:
                hit += e["recall"]["hit"]
                total += e["recall"]["total"]
        rows.append(
            {
                "engine": name,
                "recall": hit / total if total else 0.0,
                "hit": hit,
                "total": total,
                "chars": chars,
                "seconds": round(secs, 1),
            }
        )

    human_min = sum(b.minutes or 0 for b in baselines.values())
    human_lines = sum(len(b.copy_lines) for b in baselines.values())

    out = [
        "",
        f"{'엔진':<16}{'재현율':>10}{'추출 글자수':>14}{'소요':>10}{'사람 대비 속도':>16}",
        "─" * 68,
    ]
    for r in sorted(rows, key=lambda x: (-x["recall"], x["seconds"])):
        speedup = (human_min * 60 / r["seconds"]) if r["seconds"] else 0
        out.append(
            f"{r['engine']:<16}{r['hit']}/{r['total']} ({r['recall']:>4.0%}){r['chars']:>13,}자"
            f"{r['seconds']:>9.0f}s{speedup:>14.0f}배"
        )
    out += [
        "",
        f"사람(베이스라인): {human_min:.0f}분 = {human_min * 60:.0f}s, 카피 {human_lines}줄 기록",
    ]
    return "\n".join(out)
