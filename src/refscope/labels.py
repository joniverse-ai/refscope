"""사람이 손으로 만든 정답(data/baseline/*.md) 읽기와 채점.

정답은 PoC 출력을 보기 전에 만들어졌다. 그래야 앵커링 없이 채점이 된다.

채점 방식에 대하여 — 이 정답은 '완전한 정답'이 아니다. 사람은 44,491px 페이지에서
카피를 3줄만 적었다. 지쳐서다. 그러니 여기서 재는 재현율은
**"사람이 적어둔 것을 기계가 놓치지 않았는가"** 이지 "전부 읽었는가"가 아니다.
기계가 사람보다 얼마나 더 찾았는지는 별도 지표(추가 발견량)로 따로 센다.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml

COPY_HEADING = "헤드라인"
# 사용자가 안내 문구를 지우지 않고 남겼을 때를 대비한 방어
INSTRUCTION_HINTS = (
    "평소 리서치할 때",
    "한 줄에 하나씩",
    "위에서 아래 순서로 옮겨 적는다",
    "정답이 된다",
)


@dataclass
class Baseline:
    id: str
    brand: str
    minutes: float | None
    brand_colors: list[str]
    sections: list[str]
    copy_lines: list[str]
    pain: str = ""

    @property
    def has_copy(self) -> bool:
        return bool(self.copy_lines)


def _split_front_matter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    return yaml.safe_load(parts[1]) or {}, parts[2]


def _section(body: str, heading_contains: str) -> str:
    """`## ...` 제목으로 나뉜 본문에서 한 섹션만 꺼낸다."""
    blocks = re.split(r"^##\s+", body, flags=re.MULTILINE)
    for b in blocks[1:]:
        title, _, rest = b.partition("\n")
        if heading_contains in title:
            return rest
    return ""


def parse_baseline(path: Path) -> Baseline:
    raw = path.read_text(encoding="utf-8")
    fm, body = _split_front_matter(raw)

    copy_lines: list[str] = []
    for line in _section(body, COPY_HEADING).splitlines():
        s = line.strip()
        if not s or not s[0] in "->":
            continue
        s = s.lstrip("->").strip()
        if not s or any(h in s for h in INSTRUCTION_HINTS):
            continue
        copy_lines.append(s)

    pain = " ".join(
        ln.strip().lstrip("->").strip()
        for ln in _section(body, "불편").splitlines()
        if ln.strip() and ln.strip()[0] in "->"
    )

    def clean_list(key: str) -> list[str]:
        vals = fm.get(key) or []
        return [
            str(v).strip()
            for v in vals
            if str(v).strip() and not str(v).strip().startswith(("<", "#_"))
        ]

    minutes = fm.get("minutes")
    minutes = float(minutes) if isinstance(minutes, (int, float)) else None

    return Baseline(
        id=str(fm.get("id", path.stem)),
        brand=str(fm.get("brand", "")),
        minutes=minutes,
        brand_colors=clean_list("brand_colors"),
        sections=clean_list("sections"),
        copy_lines=copy_lines,
        pain=pain,
    )


def load_all(baseline_dir: Path) -> dict[str, Baseline]:
    out: dict[str, Baseline] = {}
    for p in sorted(baseline_dir.glob("*.md")):
        if p.stem in ("README", "template"):
            continue
        b = parse_baseline(p)
        out[b.id] = b
    return out


# ── 채점 ──────────────────────────────────────────────────────────────────
def normalize(s: str) -> str:
    """공백·문장부호·전각을 지운다. OCR은 띄어쓰기에서 자주 갈리는데,
    디자이너가 카피를 알아보는 데에는 띄어쓰기가 결정적이지 않다."""
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"[\s\W_]+", "", s).lower()


def line_matches(truth: str, candidate: str, threshold: float = 0.75) -> bool:
    t, c = normalize(truth), normalize(candidate)
    if not t or not c:
        return False
    if t in c or c in t:
        return True
    return difflib.SequenceMatcher(None, t, c).ratio() >= threshold


def recall(truth_lines: list[str], found_lines: list[str], threshold: float = 0.75) -> dict:
    """사람이 적은 줄 중 기계가 잡아낸 비율.

    한 줄이 OCR에서 여러 조각으로 쪼개지는 일이 흔하므로, 붙여놓은 전체
    텍스트와도 대조한다. ("존경의 마음을 전하는" + "가장 확실한 방법")
    """
    joined = normalize(" ".join(found_lines))
    hits, misses = [], []
    for t in truth_lines:
        nt = normalize(t)
        ok = nt in joined or any(line_matches(t, f, threshold) for f in found_lines)
        (hits if ok else misses).append(t)
    total = len(truth_lines)
    return {
        "total": total,
        "hit": len(hits),
        "recall": len(hits) / total if total else 0.0,
        "misses": misses,
    }
