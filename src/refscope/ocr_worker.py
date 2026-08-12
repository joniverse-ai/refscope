"""엔진 하나를 별도 프로세스에서 돌려 결과를 JSON으로 남긴다.

프로세스를 나누는 이유는 셋이다.

1. **엔진끼리 서로 망가뜨린다.** PaddleOCR과 EasyOCR을 한 프로세스에서 연달아
   돌렸더니 EasyOCR이 Paddle의 `Tensor holds no memory` 오류로 죽었다. 격리
   전에는 EasyOCR 재현율이 0%로 나왔는데, 단독으로 돌리니 멀쩡히 동작했다.
   비교 실험에서 이런 오염은 결론을 통째로 뒤집는다.
2. **결과를 캐시할 수 있다.** 느린 VLM을 매번 다시 돌리지 않아도 된다.
3. **하나가 죽어도 나머지는 남는다.**

사용: python -m refscope.ocr_worker <engine> <ref_id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import config
from .ocr import get_engine


def run(engine_name: str, ref_id: str, force: bool = False) -> dict:
    out_dir = config.DERIVED_DIR / "ocr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ref_id}__{engine_name}.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    ref_dir = config.REFS_DIR / ref_id
    manifest_path = ref_dir / "crops" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{ref_id}: 조각이 없습니다. `refscope crop`을 먼저 돌리세요")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    engine = get_engine(engine_name)

    lines: list[dict] = []
    errors: list[str] = []
    seconds = 0.0
    for item in manifest:
        res = engine.read(ref_dir / "crops" / item["file"])
        seconds += res.seconds
        if res.error:
            errors.append(f"{item['file']}: {res.error}")
        for ln in res.lines:
            if ln.text.strip():
                lines.append(
                    {
                        "text": ln.text.strip(),
                        "conf": round(ln.conf, 3),
                        # 조각 안 좌표를 페이지 절대좌표로 되돌린다
                        "page_y": (item["y0"] + ln.y) if ln.y is not None else None,
                        "crop": item["file"],
                    }
                )

    payload = {
        "ref_id": ref_id,
        "engine": engine_name,
        "n_crops": len(manifest),
        "n_lines": len(lines),
        "chars": sum(len(x["text"]) for x in lines),
        "seconds": round(seconds, 1),
        "errors": errors,
        "lines": lines,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 2:
        print("사용: python -m refscope.ocr_worker <engine> <ref_id> [--force]")
        return 2
    engine_name, ref_id = argv[0], argv[1]
    payload = run(engine_name, ref_id, force="--force" in argv)
    print(
        f"{ref_id:<20}{engine_name:<14}{payload['n_lines']:>5}줄 "
        f"{payload['chars']:>7,}자 {payload['seconds']:>7.1f}s"
        + (f"  오류 {len(payload['errors'])}건" if payload["errors"] else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
