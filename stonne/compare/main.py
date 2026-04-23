import sys

try:
    from .scanner import scan_runs
    from .aggregator import aggregate_all, COLUMN_ORDER
    from .csv_export import write_csv
except ImportError:
    from scanner import scan_runs
    from aggregator import aggregate_all, COLUMN_ORDER
    from csv_export import write_csv


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "./experiment_runs"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else f"{root}/summary.csv"
    stonne_root = sys.argv[3] if len(sys.argv) > 3 else "."

    records = scan_runs(root)
    rows = aggregate_all(records, stonne_root)
    write_csv(rows, output_csv, COLUMN_ORDER)

    print(f"Wrote {len(rows)} row(s) to {output_csv}")


if __name__ == "__main__":
    main()
