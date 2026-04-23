from itertools import product


def _run_name_key(key):
    return key.replace("_", "")


def expand(config):
    fixed = config["fixed"]
    sweep = config["sweep"]
    output_root = config["global"]["output_root"]
    operation = config["global"]["operation"]

    sweep_keys = list(sweep.keys())
    sweep_values = [sweep[k] for k in sweep_keys]

    runs = []
    for i, combo in enumerate(product(*sweep_values), start=1):
        sweep_combo = dict(zip(sweep_keys, combo))
        params = {**fixed, **sweep_combo}
        sweep_parts = [f"{_run_name_key(k)}{v}" for k, v in sweep_combo.items()]
        run_id = f"run_{i:04d}"
        run_name = "_".join([run_id] + sweep_parts)
        runs.append({
            "run_id": run_id,
            "operation": operation,
            "params": params,
            "run_name": run_name,
            "output_dir": f"{output_root}/{run_name}",
        })

    return runs


if __name__ == "__main__":
    import sys
    try:
        from .config_parser import load_config
    except ImportError:
        from config_parser import load_config

    if len(sys.argv) < 2:
        print("Usage: python -m sweep.expander <config.yaml>")
        sys.exit(1)

    config = load_config(sys.argv[1])
    runs = expand(config)

    print(f"Total runs: {len(runs)}\n")
    for run in runs:
        print(f"  {run['run_name']}")
        print(f"    params     : {run['params']}")
        print(f"    output_dir : {run['output_dir']}")
        print()
