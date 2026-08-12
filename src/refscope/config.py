"""평가셋 정의(data/refs.yaml) 읽기."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REFS_YAML = REPO_ROOT / "data" / "refs.yaml"
REFS_DIR = REPO_ROOT / "data" / "refs"
DERIVED_DIR = REPO_ROOT / "data" / "derived"
OUT_DIR = REPO_ROOT / "out"


def load_refs(path: Path | None = None, only: list[str] | None = None) -> list[dict]:
    path = path or REFS_YAML
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    refs = data.get("refs", [])
    for r in refs:
        missing = {"id", "url"} - set(r)
        if missing:
            raise ValueError(f"refs.yaml 항목에 {missing}가 없습니다: {r}")
    if only:
        wanted = set(only)
        unknown = wanted - {r["id"] for r in refs}
        if unknown:
            raise ValueError(f"refs.yaml에 없는 id: {sorted(unknown)}")
        refs = [r for r in refs if r["id"] in wanted]
    return refs
