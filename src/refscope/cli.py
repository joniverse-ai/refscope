"""refscope 명령줄 도구."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config


def cmd_collect(args: argparse.Namespace) -> int:
    from .capture import capture_all

    refs = config.load_refs(only=args.only)
    print(f"레퍼런스 {len(refs)}개 수집 시작 (원본은 저장소에 커밋되지 않습니다)\n")
    results = capture_all(
        refs, config.REFS_DIR, delay_s=args.delay, headless=not args.headed
    )
    ok = [r for r in results if r.ok]
    print(f"\n수집 성공 {len(ok)}/{len(results)}")
    for r in results:
        if not r.ok:
            print(f"  실패 {r.ref_id}: {r.reason}")
    if not ok:
        return 1
    return cmd_analyze(args)


def cmd_analyze(args: argparse.Namespace) -> int:
    """수집물을 다시 읽어 읽기 경로를 판정한다. 재수집 없이 몇 번이든 돌릴 수 있다."""
    from .regions import analyze_dir

    refs = config.load_refs(only=args.only)
    by_id = {r["id"]: r for r in refs}
    rows = []
    for ref_id in by_id:
        d = config.REFS_DIR / ref_id
        if not (d / "meta.json").exists():
            print(f"  건너뜀 {ref_id}: 아직 수집되지 않음")
            continue
        rows.append((by_id[ref_id], analyze_dir(d)))

    if not rows:
        print("분석할 수집물이 없습니다. 먼저 `refscope collect`를 돌리세요.")
        return 1

    w = max(20, max(len(r["id"]) for _, r in rows) + 2)
    print(f"\n{'id':<{w}}{'종류':<8}{'높이':>8}  {'판정':<11}{'이미지에갇힘':>10}  OCR영역")
    print("─" * (w + 54))
    for ref, r in sorted(rows, key=lambda x: (x[0].get("page_type", ""), x[0]["id"])):
        print(
            f"{r['id']:<{w}}{ref.get('page_type', '?'):<8}{r['page_height']:>7}px  "
            f"{r['page_verdict']:<11}{r['image_locked_ratio']:>9.0%}  {r['n_ocr_regions']}개"
        )

    for ptype in ("home", "detail"):
        group = [r for ref, r in rows if ref.get("page_type") == ptype]
        if group:
            avg = sum(r["image_locked_ratio"] for r in group) / len(group)
            print(f"\n{ptype:<7} {len(group)}개 평균 '글자가 이미지에 갇힌 비율': {avg:.0%}")
    return 0


def cmd_crop(args: argparse.Namespace) -> int:
    from .crops import crop_regions

    total = 0
    for ref in config.load_refs(only=args.only):
        d = config.REFS_DIR / ref["id"]
        if not (d / "regions.json").exists():
            print(f"  건너뜀 {ref['id']}: 먼저 `refscope analyze`를 돌리세요")
            continue
        m = crop_regions(d)
        total += len(m)
        print(f"  {ref['id']:<22} 조각 {len(m):>2}개  {sum(c['height'] for c in m):>7,}px")
    print(f"\n조각 {total}개")
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    from .ocr_worker import run

    for ref in config.load_refs(only=args.only):
        try:
            p = run(args.engine, ref["id"], force=args.force)
        except FileNotFoundError as e:
            print(f"  건너뜀 {ref['id']}: {e}")
            continue
        print(
            f"  {ref['id']:<22}{args.engine:<14}{p['n_lines']:>5}줄 "
            f"{p['chars']:>7,}자 {p['seconds']:>7.1f}s"
        )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """엔진들을 서로 다른 프로세스에서 돌린 뒤 채점한다.

    같은 프로세스에서 연달아 돌리면 엔진끼리 망가진다 (ocr_worker 참고).
    """
    import subprocess
    import sys

    from .report_ocr import main as report_main

    ref_ids = [r["id"] for r in config.load_refs(only=args.only)]
    for engine in args.engines:
        for ref_id in ref_ids:
            print(f"  {engine} × {ref_id} …", flush=True)
            subprocess.run(
                [sys.executable, "-m", "refscope.ocr_worker", engine, ref_id],
                capture_output=True,
                text=True,
            )
    print(report_main(ref_ids, args.engines))
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from .palette import run as palette_run
    from .render import dump_cards, render
    from .synthesize import build_card

    refs = config.load_refs(only=args.only)
    cards = []
    for ref in refs:
        d = config.REFS_DIR / ref["id"]
        if not (d / "regions.json").exists():
            print(f"  건너뜀 {ref['id']}: 수집·분석이 필요합니다")
            continue
        palette_run(d)
        card = build_card(ref, engine=args.engine, model=args.model)
        cards.append(card)
        print(
            f"  {card.id:<22} 카피 {len(card.copy_lines):>3}줄  섹션 {len(card.sections)}개"
            f"  컬러 {len(card.palette)}개  {card.seconds:>5.1f}s"
            + ("  ⚠ " + "; ".join(card.errors) if card.errors else "")
        )
    if not cards:
        print("만들 카드가 없습니다.")
        return 1
    out = Path(args.out) if args.out else config.OUT_DIR / "research.html"
    render(cards, out)
    dump_cards(cards, config.DERIVED_DIR / "cards.json")
    print(f"\n리서치 카드 {len(cards)}장 → {out}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """모양만 다시 찍는다. 디자인 반복용 — 모델을 부르지 않아 1초도 안 걸린다."""
    import time

    from .render import render_from_cache

    t0 = time.perf_counter()
    out = Path(args.out) if args.out else config.OUT_DIR / "research.html"
    css = Path(args.css) if args.css else None
    if css and not css.exists():
        print(f"CSS 파일이 없습니다: {css}")
        return 1
    try:
        render_from_cache(config.DERIVED_DIR / "cards.json", out, css)
    except FileNotFoundError as e:
        print(e)
        return 1
    print(f"{out}  ({time.perf_counter() - t0:.2f}s{', css=' + str(css) if css else ''})")
    return 0


def cmd_css(args: argparse.Namespace) -> int:
    """지금 쓰는 CSS를 파일로 꺼낸다. 여기서부터 고쳐 나가면 된다."""
    from .render import CSS

    dest = Path(args.out)
    if dest.exists() and not args.force:
        print(f"{dest} 가 이미 있습니다. 덮어쓰려면 --force")
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(CSS.strip() + "\n", encoding="utf-8")
    print(f"{dest} 로 꺼냈습니다. 고친 뒤:\n  uv run refscope render --css {dest}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    print("[1/5] 수집")
    if cmd_collect(args) != 0:
        return 1
    print("\n[3/5] 조각내기")
    cmd_crop(args)
    print("\n[4/5] 글자 꺼내기")
    cmd_read(args)
    print("\n[5/5] 카드 만들기")
    return cmd_build(args)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="refscope", description="레퍼런스 리서치 자동 정리기"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="레퍼런스 페이지를 캡처하고 DOM을 추출한다")
    c.add_argument("--only", nargs="*", help="특정 ref id만 수집")
    c.add_argument("--delay", type=float, default=2.0, help="요청 간 대기 초 (기본 2)")
    c.add_argument("--headed", action="store_true", help="브라우저 창을 보면서 수집")
    c.set_defaults(func=cmd_collect)

    a = sub.add_parser(
        "analyze", help="수집물의 읽기 경로를 판정한다 (재수집 없이 반복 가능)"
    )
    a.add_argument("--only", nargs="*", help="특정 ref id만 분석")
    a.set_defaults(func=cmd_analyze)

    c2 = sub.add_parser("crop", help="OCR 대상 구간을 이미지 조각으로 잘라낸다")
    c2.add_argument("--only", nargs="*")
    c2.set_defaults(func=cmd_crop)

    o = sub.add_parser("read", help="조각에서 글자를 꺼낸다 (엔진별 격리 실행)")
    o.add_argument("--only", nargs="*")
    o.add_argument("--engine", default="apple_vision", help="기본 apple_vision")
    o.add_argument("--force", action="store_true", help="캐시를 무시하고 다시 읽는다")
    o.set_defaults(func=cmd_read)

    cp = sub.add_parser("compare", help="여러 OCR 엔진을 같은 조각에 돌려 채점한다")
    cp.add_argument("--only", nargs="*")
    cp.add_argument(
        "--engines", nargs="*", default=["apple_vision", "paddleocr", "easyocr"]
    )
    cp.set_defaults(func=cmd_compare)

    b = sub.add_parser("build", help="리서치 카드 HTML을 만든다 (파이프라인 끝단)")
    b.add_argument("--only", nargs="*")
    b.add_argument("--engine", default="apple_vision")
    b.add_argument("--model", default="qwen2.5vl:7b")
    b.add_argument("--out", default=None, help="출력 HTML 경로")
    b.set_defaults(func=cmd_build)

    rd = sub.add_parser(
        "render", help="카드 HTML만 다시 찍는다 (모델 안 부름 — 디자인 반복용)"
    )
    rd.add_argument("--css", default=None, help="기본 CSS 대신 쓸 파일")
    rd.add_argument("--out", default=None, help="출력 HTML 경로")
    rd.set_defaults(func=cmd_render)

    cs = sub.add_parser("css", help="지금 쓰는 CSS를 파일로 꺼낸다 (고쳐 쓰기 시작점)")
    cs.add_argument("--out", default="themes/custom.css")
    cs.add_argument("--force", action="store_true")
    cs.set_defaults(func=cmd_css)

    r = sub.add_parser("run", help="수집부터 카드까지 한 번에")
    r.add_argument("--only", nargs="*")
    r.add_argument("--delay", type=float, default=2.0)
    r.add_argument("--headed", action="store_true")
    r.add_argument("--engine", default="apple_vision")
    r.add_argument("--model", default="qwen2.5vl:7b")
    r.add_argument("--force", action="store_true")
    r.add_argument("--out", default=None)
    r.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
