"""Runnable demo.

    python -m robot_runtime.cli                    # scripted visit
    python -m robot_runtime.cli --mode manual      # type p / l / q
    python -m robot_runtime.cli --motion-failure-rate 0.5
    python -m robot_runtime.cli --mode camera      # needs requirements-camera.txt
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .perception.detectors.simulated import ManualDetector, ScriptedDetector, Segment
from .presence_filter.filter import PresenceConfig
from .runtime.app import AppConfig, RobotApp
from .runtime.clock import RealClock

SCENARIO = [
    Segment(start_s=0.0, present=False),
    Segment(start_s=2.0, present=True),
    Segment(start_s=4.0, present=False),  # a brief occlusion, must NOT end the visit
    Segment(start_s=4.3, present=True),
    Segment(start_s=9.0, present=False),  # a real departure
    Segment(start_s=13.0, present=True),  # the same visitor returns: two behaviors now
    Segment(start_s=18.0, present=False),  # compete, so the policy has a real decision
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive robot runtime demo")
    parser.add_argument("--mode", choices=("scripted", "manual", "camera"), default="scripted")
    parser.add_argument("--duration", type=float, default=22.0, help="seconds (scripted mode)")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--motion-failure-rate", type=float, default=0.0)
    parser.add_argument("--motion-slow-rate", type=float, default=0.0)
    parser.add_argument("--sensor-noise", type=float, default=0.02, help="flicker rate (scripted mode)")
    parser.add_argument("--vlm-latency", type=float, default=0.12)
    parser.add_argument("--vlm-invalid-rate", type=float, default=0.0)
    parser.add_argument("--vlm-error-rate", type=float, default=0.0)
    parser.add_argument("--decision-deadline", type=float, default=0.25)
    parser.add_argument("--no-vlm", action="store_true", help="use the deterministic rule policy only")
    parser.add_argument("--trace-file", type=str, default=None, help="write JSONL traces here")
    parser.add_argument("--robot-config", type=str, default=None, help="path to the robot model YAML")
    parser.add_argument("--seed", type=int, default=7)
    return parser


async def _run(args: argparse.Namespace) -> int:
    clock = RealClock()
    config = AppConfig(
        fps=args.fps,
        presence=PresenceConfig(enter_frames=3, exit_frames=15),
        robot_config=args.robot_config,
        motion_failure_rate=args.motion_failure_rate,
        motion_slow_rate=args.motion_slow_rate,
        use_vlm=not args.no_vlm,
        vlm_latency_s=args.vlm_latency,
        vlm_invalid_rate=args.vlm_invalid_rate,
        vlm_error_rate=args.vlm_error_rate,
        decision_deadline_s=args.decision_deadline,
        seed=args.seed,
    )

    trace_stream = open(args.trace_file, "w", encoding="utf-8") if args.trace_file else None
    manual: ManualDetector | None = None

    if args.mode == "scripted":
        detector = ScriptedDetector(SCENARIO, clock=clock, noise=args.sensor_noise, seed=args.seed)
    elif args.mode == "manual":
        manual = ManualDetector()
        detector = manual
    else:
        from .perception.detectors import load_yolo_detector

        detector = load_yolo_detector()

    app = RobotApp(detector=detector, clock=clock, config=config, trace_stream=trace_stream)
    _banner(args, app)

    await app.start()
    try:
        if manual is not None:
            await _manual_loop(manual)
        else:
            await asyncio.sleep(args.duration)
    finally:
        await app.stop()
        if trace_stream is not None:
            trace_stream.close()
            print(f"\nTraces written to {args.trace_file}")

    print(f"\nFinal state: {app.state.value}   frames: {app.perception.frames}")
    return 0


async def _manual_loop(detector: ManualDetector) -> None:
    print("Commands:  p = person enters   l = person leaves   q = quit\n")
    loop = asyncio.get_running_loop()
    while True:
        line = (await loop.run_in_executor(None, sys.stdin.readline)).strip().lower()
        if line in ("q", "quit", "exit", ""):
            return
        if line == "p":
            detector.set_present(True)
        elif line == "l":
            detector.set_present(False)
        else:
            print("  (expected p, l or q)")


def _banner(args: argparse.Namespace, app: RobotApp) -> None:
    presence = app.presence.config
    print("=" * 78)
    print(f"  robot runtime  |  mode={args.mode}  policy={app.policy.name}  fps={args.fps:g}")
    print(f"  model: {app.model.name}  limbs: {', '.join(sorted(app.model.limbs))}")
    print(f"  presence filter: enter={presence.enter_frames} frames, exit={presence.exit_frames} frames")
    print(f"  behaviors: {', '.join(app.behavior.registry.names())}")
    print(f"  primitives: {', '.join(app.primitives.names())}")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
