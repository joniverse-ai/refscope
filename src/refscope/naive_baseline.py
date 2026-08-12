"""대조군 B — 순진하게 AI를 쓰는 방법.

문제 정의서 6절이 약속한 비교군 셋 중 두 번째.

  (A) 수작업          — 사람이 손으로 정리
  (B) 순진한 AI       ← 여기. 페이지를 VLM에 통째로 던지고 "정리해줘"
  (C) refscope        — 구간을 판정하고 도구를 갈아 끼운 파이프라인

B를 두는 이유는 분명하다. B가 없으면 이 PoC의 성과가 "AI를 썼기 때문"인지
"설계 때문"인지 구분할 수 없다. B와 C는 같은 모델(qwen2.5vl:7b)을 쓰고
같은 페이지를 본다. 다른 것은 **어떻게 보여주고 무엇을 시키느냐** 뿐이다.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import config
from .synthesize import ask_ollama, contact_sheet

NAIVE_PROMPT = """이 이미지는 한국 온라인 쇼핑몰 상품 상세페이지 전체를 세로로 6등분해
왼쪽부터 오른쪽으로 나란히 붙인 것입니다.

브랜드 디자이너가 경쟁사 리서치를 한다고 생각하고, 이 페이지를 정리해 주세요.
아래 형식 그대로, 다른 말 없이 출력하세요.

카피:
- (페이지에 적힌 헤드라인·서브카피를 보이는 대로, 최대 10줄)

컬러:
- (주요 브랜드 컬러를 #rrggbb 형식으로 3개)

섹션:
- (페이지 구성을 위에서 아래 순서로 5개)

인상:
- (전체 인상 한 문장)"""


def _parse(text: str) -> dict:
    """느슨하게 판다.

    처음에는 `- ` 불릿으로 시작하는 줄만 받았다. 그랬더니 3개 중 2개가 "카피 0줄"로
    나왔는데, 원문을 열어보니 모델은 멀쩡히 내용을 냈고 다만 불릿 대신 맨 줄이나
    "1. " 번호를 썼을 뿐이었다. **대조군을 파서 결함 때문에 지게 만들면 비교
    자체가 거짓이 된다.** 그래서 세 가지 형식을 모두 받는다.

    프롬프트 문장이 그대로 답변에 섞여 나오는 것(프롬프트 누출)은 걸러낸다.
    그건 형식 문제가 아니라 모델이 실제로 실패한 것이므로 따로 센다.
    """
    out: dict[str, list[str]] = {"copy": [], "colors": [], "sections": [], "tone": []}
    leaked: list[str] = []
    key_map = {"카피": "copy", "컬러": "colors", "섹션": "sections", "인상": "tone"}
    prompt_marks = ("브랜드 디자이너가", "다른 말 없이", "형식 그대로", "최대 10줄", "이미지는 한국")

    cur: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        head = re.match(r"^[#*\s]*(카피|컬러|섹션|인상)\s*[:：]?\s*[*#]*\s*$", line)
        if head:
            cur = key_map[head.group(1)]
            continue
        if not cur:
            continue
        item = re.sub(r"^\s*(?:[-•*]|\d+[.)])\s*", "", line).strip()
        if not item:
            continue
        if any(m in item for m in prompt_marks):
            leaked.append(item)
            continue
        out[cur].append(item)

    hexes = []
    for c in out["colors"]:
        m = re.search(r"#[0-9a-fA-F]{6}", c)
        if m:
            hexes.append(m.group(0).lower())
    out["colors"] = hexes
    out["leaked"] = leaked  # type: ignore[assignment]
    return out


def run_one(ref: dict, model: str = "qwen2.5vl:7b") -> dict:
    ref_dir = config.REFS_DIR / ref["id"]
    t0 = time.perf_counter()
    sheet = contact_sheet(ref_dir / "page.png")
    text, err = ask_ollama(model, NAIVE_PROMPT, images=[sheet])
    parsed = _parse(text)
    return {
        "ref_id": ref["id"],
        "model": model,
        "seconds": round(time.perf_counter() - t0, 1),
        "error": err,
        "raw": text,
        **parsed,
    }


def run(ref_ids: list[str], model: str = "qwen2.5vl:7b") -> dict:
    from .config import load_refs

    refs = {r["id"]: r for r in load_refs()}
    results = {}
    for rid in ref_ids:
        r = run_one(refs[rid], model)
        results[rid] = r
        print(
            f"  {rid:<20} 카피 {len(r['copy']):>2}줄  컬러 {len(r['colors'])}개  "
            f"섹션 {len(r['sections'])}개  {r['seconds']:>5.1f}s"
            + (f"  ⚠ {r['error']}" if r["error"] else "")
        )
    out = config.DERIVED_DIR / "naive_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return results


if __name__ == "__main__":
    import sys

    ids = sys.argv[1:] or ["gyodong_manwol", "hanpoom_99flower", "hangamall_dao"]
    run(ids)
