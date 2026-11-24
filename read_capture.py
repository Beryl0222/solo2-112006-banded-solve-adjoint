import argparse
import json
from pathlib import Path


def read_capture(path):
    """读取一次实验交接采集，保留原始事件顺序。"""
    with Path(path).open(encoding="utf-8") as stream:
        capture = json.load(stream)
    events = capture.get("events") if isinstance(capture, dict) else None
    if not isinstance(events, list) or any(
        not isinstance(event, list) or not event or not isinstance(event[0], str)
        for event in events
    ):
        raise ValueError("采集资料含有无效的事件列表")
    return capture


def replay_capture(path, **kwargs):
    """按采集事件回放带状求解/发证/反传，返回 :class:`band_solver.Session`。"""
    from band_solver import replay
    return replay(read_capture(path), **kwargs)


# 便捷再导出：允许 ``from read_capture import Session, replay``。
try:  # pragma: no cover - 导入失败不应影响基础采集读取
    from band_solver import (  # noqa: F401
        Session, Solver, Scheduler, Report, replay, json_dumps,
        create_session, run_events, solve_banded, certificate,
    )
except Exception:  # pragma: no cover
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查看交接资料的事件顺序")
    parser.add_argument("capture", type=Path)
    parser.add_argument(
        "--replay", "--run", action="store_true",
        help="实际受理 solve/publish/back 事件并输出 JSON 数值报告")
    parser.add_argument("--out", type=Path, default=None,
                        help="配合 --replay，把 JSON 报告写入文件")
    args = parser.parse_args()
    capture = read_capture(args.capture)

    if args.replay:
        from band_solver import replay, json_dumps
        session = replay(capture)
        report = session.report()
        text = json_dumps(report)
        if args.out is not None:
            args.out.write_text(text, encoding="utf-8")
        print(text)
    else:
        for index, event in enumerate(capture["events"], start=1):
            print(f"{index:02} {event[0]}")
