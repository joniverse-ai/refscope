"""브랜드 컬러 팔레트 추출 — 스포이드질을 대신한다.

디자이너가 레퍼런스를 볼 때 하는 일 중 하나가 스포이드로 색을 찍어 적는
것이다. 사람은 3색쯤 찍고 만다. 기계는 페이지 전체의 색 분포를 볼 수 있다.

두 가지를 조심한다.
  - **흰 배경이 전부를 차지한다.** 상세페이지의 절반 이상은 흰색·회색 여백이다.
    그대로 k-means를 돌리면 "이 브랜드의 색은 #ffffff"라는 쓸모없는 답이 나온다.
  - **사람이 찍은 색과 맞춰봐야 한다.** 채점은 CIEDE2000 색차로 한다. RGB
    거리로 재면 사람 눈이 느끼는 차이와 어긋난다.
"""

from __future__ import annotations

import colorsys
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SAMPLE_MAX_PX = 400_000  # 44,491px 페이지를 통째로 클러스터링하지 않는다


@dataclass
class Swatch:
    hex: str
    ratio: float  # 이 색이 차지하는 픽셀 비율
    role: str  # neutral | accent


def _to_hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(round(c)) for c in rgb))


def _is_neutral(rgb, sat_max: float = 0.12) -> bool:
    """무채색(흰·검·회)인가. 브랜드 컬러 후보에서 뺄 대상."""
    r, g, b = (c / 255 for c in rgb)
    _, _, s = colorsys.rgb_to_hsv(r, g, b)[0], colorsys.rgb_to_hsv(r, g, b)[2], colorsys.rgb_to_hsv(r, g, b)[1]
    return s <= sat_max


# ── 색차 (CIEDE2000) ──────────────────────────────────────────────────────
def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB(0-255) → CIELAB (D65)."""
    c = np.asarray(rgb, dtype=float) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = c @ m.T
    white = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz / white
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def delta_e_2000(rgb1, rgb2) -> float:
    """CIEDE2000 색차. 사람 눈이 느끼는 차이에 맞춘 척도. 2 이하면 거의 같은 색."""
    lab1, lab2 = _srgb_to_lab(np.array(rgb1)), _srgb_to_lab(np.array(rgb2))
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    avg_L = (L1 + L2) / 2
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    avg_C = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(avg_C**7 / (avg_C**7 + 25**7))) if avg_C > 0 else 0.0
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    avg_Cp = (C1p + C2p) / 2
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    else:
        dhp = h2p - h1p - 360 if h2p > h1p else h2p - h1p + 360
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)
    if C1p * C2p == 0:
        avg_hp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        avg_hp = (h1p + h2p) / 2
    else:
        avg_hp = (h1p + h2p + 360) / 2 if (h1p + h2p) < 360 else (h1p + h2p - 360) / 2
    T = (
        1
        - 0.17 * np.cos(np.radians(avg_hp - 30))
        + 0.24 * np.cos(np.radians(2 * avg_hp))
        + 0.32 * np.cos(np.radians(3 * avg_hp + 6))
        - 0.20 * np.cos(np.radians(4 * avg_hp - 63))
    )
    Sl = 1 + (0.015 * (avg_L - 50) ** 2) / np.sqrt(20 + (avg_L - 50) ** 2)
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    Rt = (
        -2
        * np.sqrt(avg_Cp**7 / (avg_Cp**7 + 25**7))
        * np.sin(np.radians(60 * np.exp(-(((avg_hp - 275) / 25) ** 2))))
        if avg_Cp > 0
        else 0.0
    )
    return float(
        np.sqrt(
            (dLp / Sl) ** 2
            + (dCp / Sc) ** 2
            + (dHp / Sh) ** 2
            + Rt * (dCp / Sc) * (dHp / Sh)
        )
    )


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.strip().lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


# ── 추출 ──────────────────────────────────────────────────────────────────
def extract_palette(page_png: Path, k: int = 12, top: int = 8) -> list[Swatch]:
    from sklearn.cluster import MiniBatchKMeans

    img = Image.open(page_png).convert("RGB")
    w, h = img.size
    scale = (SAMPLE_MAX_PX / (w * h)) ** 0.5
    if scale < 1:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    px = np.asarray(img).reshape(-1, 3)
    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=5, batch_size=4096)
    labels = km.fit_predict(px)
    counts = np.bincount(labels, minlength=k)
    total = counts.sum()

    swatches = [
        Swatch(
            hex=_to_hex(center),
            ratio=round(float(c / total), 4),
            role="neutral" if _is_neutral(center) else "accent",
        )
        for center, c in zip(km.cluster_centers_, counts)
    ]
    # 유채색을 먼저, 그 안에서 면적이 큰 순. 흰 배경이 1등을 차지하는 것을 막는다.
    swatches.sort(key=lambda s: (s.role == "neutral", -s.ratio))
    return swatches[:top]


def score_palette(truth_hexes: list[str], found: list[Swatch], tol: float = 10.0) -> dict:
    """사람이 스포이드로 찍은 색이 자동 팔레트 안에 있는가 (성공 기준 #3)."""
    rows = []
    for t in truth_hexes:
        try:
            trgb = hex_to_rgb(t)
        except (ValueError, IndexError):
            continue
        best = min(
            ((s, delta_e_2000(trgb, hex_to_rgb(s.hex))) for s in found),
            key=lambda x: x[1],
            default=(None, 999.0),
        )
        rows.append(
            {
                "truth": t,
                "nearest": best[0].hex if best[0] else None,
                "delta_e": round(best[1], 1),
                "matched": best[1] <= tol,
            }
        )
    return {
        "tolerance": tol,
        "matched": sum(1 for r in rows if r["matched"]),
        "total": len(rows),
        "rows": rows,
    }


def run(ref_dir: Path, k: int = 12, top: int = 8) -> list[Swatch]:
    sw = extract_palette(ref_dir / "page.png", k=k, top=top)
    (ref_dir / "palette.json").write_text(
        json.dumps([asdict(s) for s in sw], ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return sw
