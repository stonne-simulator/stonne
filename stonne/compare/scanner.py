import json
import os
import sys


def scan_runs(root_dir):
    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a directory")
        return []

    records = []

    for entry in sorted(os.listdir(root_dir)):
        run_path = os.path.join(root_dir, entry)
        if not os.path.isdir(run_path):
            continue

        record = {
            "run_name": entry,
            "run_path": run_path,
            "stats_file": None,
            "counters_file": None,
            "params": {},
            "success": None,
            "return_code": None,
            "status_file": None,
        }

        status_path = os.path.join(run_path, "status.json")
        run_config_path = os.path.join(run_path, "run_config.json")

        if os.path.isfile(status_path):
            record["status_file"] = status_path
            try:
                with open(status_path, "r") as f:
                    status_data = json.load(f)
                record["success"] = status_data.get("success")
                record["return_code"] = status_data.get("return_code")
                record["params"] = status_data.get("params", {}) or {}
                record["stats_file"] = status_data.get("stats_json")
                record["counters_file"] = status_data.get("counters_file")
            except (json.JSONDecodeError, OSError):
                pass

        if not record["stats_file"] or not record["counters_file"]:
            for filename in os.listdir(run_path):
                filepath = os.path.join(run_path, filename)
                if not os.path.isfile(filepath):
                    continue
                if not record["stats_file"] and filename.endswith(".txt") and filename.startswith("output_stats"):
                    record["stats_file"] = filepath
                elif not record["counters_file"] and filename.endswith(".counters"):
                    record["counters_file"] = filepath

        if not record["params"] and os.path.isfile(run_config_path):
            try:
                with open(run_config_path, "r") as f:
                    run_config = json.load(f)
                record["params"] = run_config.get("params", {}) or {}
            except (json.JSONDecodeError, OSError):
                pass

        if not record["params"]:
            record["params"] = parse_run_name(entry)

        records.append(record)

    return records


def parse_run_name(run_name):
    params = {}
    parts = run_name.split("_")
    for part in parts:
        for key in ["TM", "TN", "TK"]:
            if part.startswith(key) and part[len(key):].isdigit():
                params[key] = int(part[len(key):])
    return params


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "./experiment_runs"
    runs = scan_runs(root)
    print(f"Found {len(runs)} run(s) in {root}\n")
    for r in runs:
        print(f"  Run        : {r['run_name']}")
        print(f"  Success    : {r['success']}")
        print(f"  ReturnCode : {r['return_code']}")
        print(f"  Params     : {r['params']}")
        print(f"  Stats      : {r['stats_file']}")
        print(f"  Counters   : {r['counters_file']}")
        print()
