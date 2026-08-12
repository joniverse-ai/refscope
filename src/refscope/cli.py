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
    print(f"\n성공 {len(ok)}/{len(results)}")
    if ok:
        paths = {}
        for r in ok:
            paths[r.verdict] = paths.get(r.verdict, 0) + 1
        summary = ", ".join(f"{k} {v}개" for k, v in sorted(paths.items()))
        print(f"읽기 경로 판정: {summary}")
    for r in results:
        if not r.ok:
            print(f"  실패 {r.ref_id}: {r.reason}")
    return 0 if ok else 1


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

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
