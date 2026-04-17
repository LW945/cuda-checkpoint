#!/usr/bin/env python3

"""Time CUDA checkpoint operations for a single process."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CommandResult:
    name: str
    argv: list[str]
    duration_ms: float
    exit_code: int
    stdout: str
    stderr: str


class StepFailed(RuntimeError):
    def __init__(self, step: str, result: CommandResult):
        super().__init__(f"{step} failed with exit code {result.exit_code}")
        self.step = step
        self.result = result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure CUDA checkpoint timings for a running process. "
            "The tool executes: lock -> checkpoint -> restore -> unlock."
        )
    )
    parser.add_argument("--pid", type=int, required=True, help="PID of the target CUDA process")
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=30000,
        help="Lock timeout in milliseconds. Use 0 to wait indefinitely. Default: 30000",
    )
    parser.add_argument(
        "--device-map",
        help=(
            "Optional GPU remap passed to restore in the format "
            "oldUuid=newUuid,oldUuid=newUuid,..."
        ),
    )
    parser.add_argument(
        "--cuda-checkpoint",
        dest="cuda_checkpoint",
        help="Path to the cuda-checkpoint binary. Defaults to the repo copy or PATH.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable summary.",
    )
    return parser.parse_args()


def resolve_cuda_checkpoint(explicit: str | None) -> str:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.is_file():
            return str(path)
        raise FileNotFoundError(f"cuda-checkpoint not found at {path}")

    repo_candidate = Path(__file__).resolve().parents[1] / "bin" / "x86_64_Linux" / "cuda-checkpoint"
    if repo_candidate.is_file():
        return str(repo_candidate)

    which_result = shutil.which("cuda-checkpoint")
    if which_result:
        return which_result

    raise FileNotFoundError("cuda-checkpoint binary not found in the repo or PATH")


def run_command(name: str, argv: list[str]) -> CommandResult:
    start_ns = time.monotonic_ns()
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    duration_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
    return CommandResult(
        name=name,
        argv=argv,
        duration_ms=duration_ms,
        exit_code=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )


def get_state(cuda_checkpoint: str, pid: int) -> str:
    result = run_command(
        "get-state", [cuda_checkpoint, "--get-state", "--pid", str(pid)]
    )
    if result.exit_code != 0:
        raise RuntimeError(format_command_failure(result))
    return result.stdout.strip().lower()


def safe_get_state(cuda_checkpoint: str, pid: int) -> str | None:
    try:
        return get_state(cuda_checkpoint, pid)
    except RuntimeError:
        return None


def get_restore_tid(cuda_checkpoint: str, pid: int) -> int | None:
    result = run_command(
        "get-restore-tid", [cuda_checkpoint, "--get-restore-tid", "--pid", str(pid)]
    )
    if result.exit_code != 0 or not result.stdout:
        return None
    try:
        return int(result.stdout)
    except ValueError:
        return None


def format_command_failure(result: CommandResult) -> str:
    parts = [f"{result.name} exited with code {result.exit_code}"]
    if result.stdout:
        parts.append(f"stdout: {result.stdout}")
    if result.stderr:
        parts.append(f"stderr: {result.stderr}")
    return "\n".join(parts)


def run_step(
    step_name: str,
    cuda_checkpoint: str,
    pid: int,
    timeout_ms: int | None = None,
    device_map: str | None = None,
) -> CommandResult:
    argv = [cuda_checkpoint, "--action", step_name, "--pid", str(pid)]
    if step_name == "lock" and timeout_ms is not None:
        argv.extend(["--timeout", str(timeout_ms)])
    if step_name == "restore" and device_map:
        argv.extend(["--device-map", device_map])
    result = run_command(step_name, argv)
    if result.exit_code != 0:
        raise StepFailed(step_name, result)
    return result


def cleanup_process(
    cuda_checkpoint: str,
    pid: int,
    device_map: str | None,
) -> tuple[str | None, list[CommandResult]]:
    cleanup_steps: list[CommandResult] = []
    state = safe_get_state(cuda_checkpoint, pid)

    if state == "checkpointed":
        restore_result = run_command(
            "cleanup-restore",
            [
                cuda_checkpoint,
                "--action",
                "restore",
                "--pid",
                str(pid),
                *(
                    ["--device-map", device_map]
                    if device_map
                    else []
                ),
            ],
        )
        cleanup_steps.append(restore_result)
        state = safe_get_state(cuda_checkpoint, pid)

    if state == "locked":
        unlock_result = run_command(
            "cleanup-unlock",
            [
                cuda_checkpoint,
                "--action",
                "unlock",
                "--pid",
                str(pid),
            ],
        )
        cleanup_steps.append(unlock_result)
        state = safe_get_state(cuda_checkpoint, pid)

    return state, cleanup_steps


def build_report(
    *,
    pid: int,
    cuda_checkpoint: str,
    timeout_ms: int,
    device_map: str | None,
    initial_state: str,
    final_state: str | None,
    initial_restore_tid: int | None,
    success: bool,
    total_ms: float,
    steps: list[CommandResult],
    cleanup_steps: list[CommandResult],
    failed_step: str | None,
    error: str | None,
) -> dict:
    return {
        "pid": pid,
        "cuda_checkpoint": cuda_checkpoint,
        "timeout_ms": timeout_ms,
        "device_map": device_map,
        "initial_state": initial_state,
        "final_state": final_state,
        "initial_restore_tid": initial_restore_tid,
        "success": success,
        "failed_step": failed_step,
        "error": error,
        "timings_ms": {
            step.name: round(step.duration_ms, 3) for step in steps
        },
        "total_ms": round(total_ms, 3),
        "steps": [asdict(step) for step in steps],
        "cleanup_steps": [asdict(step) for step in cleanup_steps],
    }


def print_human_report(report: dict) -> None:
    print(f"PID: {report['pid']}")
    print(f"cuda-checkpoint: {report['cuda_checkpoint']}")
    print(f"Initial CUDA state: {report['initial_state']}")
    if report["initial_restore_tid"] is not None:
        print(f"Restore thread ID: {report['initial_restore_tid']}")
    print(f"Final CUDA state: {report['final_state']}")
    print(f"Success: {report['success']}")

    if report["failed_step"]:
        print(f"Failed step: {report['failed_step']}")
    if report["error"]:
        print("Error:")
        print(report["error"])

    if report["steps"]:
        print("Timings:")
        for step in report["steps"]:
            print(
                f"  {step['name']}: {step['duration_ms']:.3f} ms "
                f"(exit {step['exit_code']})"
            )
        print(f"  total: {report['total_ms']:.3f} ms")

    if report["cleanup_steps"]:
        print("Cleanup:")
        for step in report["cleanup_steps"]:
            print(
                f"  {step['name']}: {step['duration_ms']:.3f} ms "
                f"(exit {step['exit_code']})"
            )


def main(args: argparse.Namespace) -> int:
    if args.pid <= 0:
        raise ValueError("--pid must be a positive integer")
    if args.timeout_ms < 0:
        raise ValueError("--timeout-ms must be greater than or equal to 0")
    if not Path(f"/proc/{args.pid}").exists():
        raise FileNotFoundError(f"PID {args.pid} does not exist")

    cuda_checkpoint = resolve_cuda_checkpoint(args.cuda_checkpoint)
    initial_state = get_state(cuda_checkpoint, args.pid)
    initial_restore_tid = get_restore_tid(cuda_checkpoint, args.pid)

    if initial_state != "running":
        raise RuntimeError(
            f"PID {args.pid} must be in the running CUDA state before timing begins. "
            f"Current state: {initial_state}"
        )

    steps: list[CommandResult] = []
    cleanup_steps: list[CommandResult] = []
    failed_step: str | None = None
    error: str | None = None
    total_start_ns = time.monotonic_ns()

    try:
        steps.append(run_step("lock", cuda_checkpoint, args.pid, timeout_ms=args.timeout_ms))
        steps.append(run_step("checkpoint", cuda_checkpoint, args.pid))
        steps.append(
            run_step(
                "restore",
                cuda_checkpoint,
                args.pid,
                device_map=args.device_map,
            )
        )
        steps.append(run_step("unlock", cuda_checkpoint, args.pid))
    except StepFailed as exc:
        failed_step = exc.step
        error = format_command_failure(exc.result)
        final_state, cleanup_steps = cleanup_process(
            cuda_checkpoint,
            args.pid,
            args.device_map,
        )
        total_ms = (time.monotonic_ns() - total_start_ns) / 1_000_000.0
        report = build_report(
            pid=args.pid,
            cuda_checkpoint=cuda_checkpoint,
            timeout_ms=args.timeout_ms,
            device_map=args.device_map,
            initial_state=initial_state,
            final_state=final_state,
            initial_restore_tid=initial_restore_tid,
            success=False,
            total_ms=total_ms,
            steps=steps + [exc.result],
            cleanup_steps=cleanup_steps,
            failed_step=failed_step,
            error=error,
        )
        emit_report(report, args.json)
        return 1

    total_ms = (time.monotonic_ns() - total_start_ns) / 1_000_000.0
    final_state = get_state(cuda_checkpoint, args.pid)
    report = build_report(
        pid=args.pid,
        cuda_checkpoint=cuda_checkpoint,
        timeout_ms=args.timeout_ms,
        device_map=args.device_map,
        initial_state=initial_state,
        final_state=final_state,
        initial_restore_tid=initial_restore_tid,
        success=final_state == "running",
        total_ms=total_ms,
        steps=steps,
        cleanup_steps=[],
        failed_step=None,
        error=None if final_state == "running" else f"Unexpected final state: {final_state}",
    )
    emit_report(report, args.json)
    return 0 if report["success"] else 1


def emit_report(report: dict, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print_human_report(report)


def emit_error(error: str, json_output: bool) -> None:
    if json_output:
        print(json.dumps({"success": False, "error": error}, indent=2, sort_keys=True))
        return
    print(f"error: {error}", file=sys.stderr)


if __name__ == "__main__":
    arguments = parse_args()
    try:
        sys.exit(main(arguments))
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        emit_error(str(exc), arguments.json)
        sys.exit(2)
