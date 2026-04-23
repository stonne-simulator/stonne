import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

try:
    from .config_parser import load_config
    from .expander import expand
    from .command_builder import build_command
except ImportError:
    from config_parser import load_config
    from expander import expand
    from command_builder import build_command


def _timestamp():
    return datetime.now().isoformat(timespec="seconds")


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _write_text(path, content):
    with open(path, "w") as f:
        f.write(content)


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _detect_output_files(output_dir):
    stats_file = None
    counters_file = None
    all_files = []

    runner_files = {"run_config.json", "status.json", "command.txt", "stdout.log", "stderr.log"}

    for entry in sorted(os.listdir(output_dir)):
        full_path = os.path.join(output_dir, entry)
        if not os.path.isfile(full_path):
            continue
        all_files.append(entry)
        if entry in runner_files:
            continue
        if entry.endswith(".txt") and stats_file is None:
            stats_file = full_path
        if "counter" in entry.lower() and counters_file is None:
            counters_file = full_path

    return {"stats_file": stats_file, "counters_file": counters_file, "all_files": all_files}


def run_one(binary, run_spec):
    output_dir = run_spec["output_dir"]
    _ensure_dir(output_dir)

    cmd = build_command(binary, run_spec)

    _write_json(os.path.join(output_dir, "run_config.json"), run_spec)
    _write_text(os.path.join(output_dir, "command.txt"), " ".join(cmd) + "\n")

    env = os.environ.copy()
    env["OUTPUT_DIR"] = output_dir
    started_at = _timestamp()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        _write_text(os.path.join(output_dir, "stdout.log"), result.stdout)
        _write_text(os.path.join(output_dir, "stderr.log"), result.stderr)

        detected = _detect_output_files(output_dir)
        success = result.returncode == 0

        status = {
            "run_id": run_spec["run_id"],
            "run_name": run_spec["run_name"],
            "operation": run_spec["operation"],
            "params": run_spec["params"],
            "command": cmd,
            "return_code": result.returncode,
            "success": success,
            "started_at": started_at,
            "finished_at": _timestamp(),
            "output_dir": output_dir,
            "stats_json": detected["stats_file"],
            "counters_file": detected["counters_file"],
            "all_files": detected["all_files"],
        }

    except FileNotFoundError as e:
        _write_text(os.path.join(output_dir, "stdout.log"), "")
        _write_text(os.path.join(output_dir, "stderr.log"), str(e) + "\n")
        status = {
            "run_id": run_spec["run_id"],
            "run_name": run_spec["run_name"],
            "operation": run_spec["operation"],
            "params": run_spec["params"],
            "command": cmd,
            "return_code": None,
            "success": False,
            "started_at": started_at,
            "finished_at": _timestamp(),
            "output_dir": output_dir,
            "stats_json": None,
            "counters_file": None,
            "all_files": [],
            "error": f"Binary not found: {e}",
        }

    _write_json(os.path.join(output_dir, "status.json"), status)
    return status


def run_all(config):
    binary = config["global"]["binary"]
    runs = expand(config)
    statuses = []

    for run_spec in runs:
        print(f"Running {run_spec['run_name']} ...")
        status = run_one(binary, run_spec)
        statuses.append(status)
        if status["success"]:
            print(f"  done: return_code={status['return_code']}")
        else:
            print(f"  failed: return_code={status['return_code']}")

    return statuses


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m sweep.runner <config.yaml>")
        sys.exit(1)

    try:
        config = load_config(sys.argv[1])
        results = run_all(config)
        total = len(results)
        passed = sum(1 for r in results if r["success"])
        print(f"\nRun summary")
        print(f"  total  : {total}")
        print(f"  passed : {passed}")
        print(f"  failed : {total - passed}")
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
