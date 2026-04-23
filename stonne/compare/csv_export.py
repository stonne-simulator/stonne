import csv


def write_csv(rows, output_file, column_order=None):
    if not rows:
        return
    fieldnames = column_order or list(rows[0].keys())
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
