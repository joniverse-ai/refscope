"""판정된 OCR 영역을 실제 이미지 조각으로 잘라낸다.

44,491px짜리 PNG를 통째로 OCR이나 VLM에 밀어 넣을 수는 없다. regions.py가
"어디를 읽어야 하는지" 판정했으니, 여기서는 그 좌표대로 자르기만 한다.

자를 때 두 가지를 신경 쓴다.
  - 겹침(overlap): 영역 경계에서 글자가 반으로 잘리면 OCR이 놓친다.
  - 최대 높이: 너무 길면 OCR 엔진이 축소해버려 작은 글자가 뭉갠다.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # 44,491px짜리 페이지는 PIL의 폭탄 경고에 걸린다

OVERLAP = 80  # 영역 경계에서 글자가 잘리지 않도록 위아래로 덧붙이는 여유(px)


def crop_regions(ref_dir: Path, kinds: tuple[str, ...] = ("ocr", "both")) -> list[dict]:
    """regions.json의 판정대로 page.png를 잘라 crops/에 저장한다."""
    regions_path = ref_dir / "regions.json"
    if not regions_path.exists():
        raise FileNotFoundError(f"{ref_dir.name}: 먼저 `refscope analyze`를 돌리세요")

    data = json.loads(regions_path.read_text(encoding="utf-8"))
    page = Image.open(ref_dir / "page.png").convert("RGB")
    pw, ph = page.size

    crops_dir = ref_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    for old in crops_dir.glob("*.png"):
        old.unlink()

    manifest: list[dict] = []
    for i, r in enumerate(data["regions"]):
        if r["kind"] not in kinds:
            continue
        y0 = max(0, r["y0"] - OVERLAP)
        y1 = min(ph, r["y1"] + OVERLAP)
        if y1 - y0 < 50:
            continue
        name = f"{i:02d}_{r['kind']}_{y0}-{y1}.png"
        page.crop((0, y0, pw, y1)).save(crops_dir / name)
        manifest.append(
            {
                "file": name,
                "region_index": i,
                "kind": r["kind"],
                "y0": y0,
                "y1": y1,
                "height": y1 - y0,
                "image_cover": r["image_cover"],
            }
        )

    (crops_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return manifest
