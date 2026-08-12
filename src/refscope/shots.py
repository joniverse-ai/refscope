"""README에 넣을 시연 스크린샷을 만든다.

손으로 캡처하면 다시 만들 수 없다. 결과물이 바뀌면 이 스크립트를 다시 돌린다.
라이트·다크 두 벌을 찍는다 — 카드 HTML이 두 테마를 다 지원하기 때문이다.
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

from . import config

ASSETS = config.REPO_ROOT / "docs" / "assets"


def shoot(
    targets: list[tuple[str, Path, int]],
    scheme: str = "dark",
    width: int = 1200,
) -> list[Path]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": width, "height": 1000},
            color_scheme=scheme,
            device_scale_factor=2,
        )
        page = ctx.new_page()
        for name, src, height in targets:
            if not src.exists():
                print(f"  건너뜀 {name}: {src} 없음")
                continue
            page.goto(src.as_uri(), wait_until="load")
            page.wait_for_timeout(500)
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(300)
            dest = ASSETS / f"{name}-{scheme}.png"
            page.screenshot(path=str(dest))
            made.append(dest)
            print(f"  {dest.relative_to(config.REPO_ROOT)}")
        browser.close()
    return made


def run() -> list[Path]:
    targets = [
        ("research-cards", config.OUT_DIR / "research.html", 1500),
        ("patterns", config.OUT_DIR / "patterns.html", 1500),
    ]
    made = shoot(targets, scheme="dark")
    made += shoot(targets, scheme="light")
    return made


if __name__ == "__main__":
    run()
