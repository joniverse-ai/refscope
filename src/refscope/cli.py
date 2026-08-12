"""refscope 명령줄 도구."""

from __future__ import annotations

import argparse
import sys

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

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
