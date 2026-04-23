import os
import sys

COLUMN_ORDER = [
    "run_name", "status",
    "TM", "TN", "TK",
    "num_ms", "dn_bw", "rn_bw",
    "cycles", "layer_type",
    "CB_WIRE_WRITE", "CB_WIRE_READ",
    "FIFO_PUSH", "FIFO_POP",
    "energy",
    "stats_file", "counters_file", "energy_file",
]


def aggregate_run(record, stats, counters, energy):
    row = {col: None for col in COLUMN_ORDER}
    row["run_name"] = record.get("run_name")

    params = record.get("params") or {}
    row["TM"] = params.get("T_M", params.get("TM"))
    row["TN"] = params.get("T_N", params.get("TN"))
    row["TK"] = params.get("T_K", params.get("TK"))

    if stats:
        for key in ("cycles", "num_ms", "dn_bw", "rn_bw", "layer_type"):
            if key in stats:
                row[key] = stats[key]

    if counters:
        for key in ("CB_WIRE_WRITE", "CB_WIRE_READ", "FIFO_PUSH", "FIFO_POP"):
            if key in counters:
                row[key] = counters[key]

    if energy:
        row["energy"] = energy.get("energy")
        row["energy_file"] = energy.get("energy_file")

    success = record.get("success")
    if success is True:
        row["status"] = "success"
    elif success is False:
        row["status"] = "failed"
    else:
        row["status"] = "unknown"

    if row["status"] == "success" and energy and not energy.get("success"):
        row["status"] = "energy_failed"

    row["stats_file"] = record.get("stats_file")
    row["counters_file"] = record.get("counters_file")
    return row


def aggregate_all(records, stonne_root="."):
    try:
        from .stats_parser import parse_stats, parse_counters
        from .energy_wrapper import run_energy
    except ImportError:
        from stats_parser import parse_stats, parse_counters
        from energy_wrapper import run_energy

    rows = []
    for record in records:
        stats = parse_stats(record.get("stats_file"))
        counters = parse_counters(record.get("counters_file"))
        energy = None
        counters_file = record.get("counters_file")
        if counters_file and os.path.isfile(counters_file):
            energy = run_energy(counters_file, record["run_path"], stonne_root)
        rows.append(aggregate_run(record, stats, counters, energy))

    return rows


if __name__ == "__main__":
    try:
        from .scanner import scan_runs
    except ImportError:
        from scanner import scan_runs

    root = sys.argv[1] if len(sys.argv) > 1 else "./experiment_runs"
    stonne_root = sys.argv[2] if len(sys.argv) > 2 else "."

    records = scan_runs(root)
    rows = aggregate_all(records, stonne_root)

    print(f"Aggregated {len(rows)} run(s) from {root}\n")
    for row in rows:
        print(f"  {row['run_name']}  ({row['status']})")
        for col in COLUMN_ORDER:
            if row[col] is not None and col not in ("stats_file", "counters_file", "energy_file", "run_name", "status"):
                print(f"    {col}: {row[col]}")
        print()
