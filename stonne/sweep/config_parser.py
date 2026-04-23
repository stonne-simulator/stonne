import yaml
import os

try:
    from .validator import validate_config
except ImportError:
    from validator import validate_config

SUPPORTED_OPERATIONS = ["DenseGEMM"]

REQUIRED_TOP_LEVEL_KEYS = ["binary", "output_root", "operation", "fixed", "sweep"]

INT_PARAMS = {
    "M", "N", "K",
    "num_ms", "dn_bw", "rn_bw",
    "T_M", "T_N", "T_K",
    "print_stats",
}


def _coerce_value(key, value):
    if key in INT_PARAMS:
        try:
            return int(value)
        except (ValueError, TypeError):
            raise ValueError(f"Parameter '{key}' must be an integer, got: {value!r}")
    return value


def _normalize_fixed(d):
    if not isinstance(d, dict):
        raise ValueError("'fixed' must be a dictionary.")
    return {k.strip(): _coerce_value(k.strip(), v) for k, v in d.items()}


def _normalize_sweep(d):
    if not isinstance(d, dict):
        raise ValueError("'sweep' must be a dictionary.")
    result = {}
    for k, v in d.items():
        key = k.strip()
        if not isinstance(v, list):
            raise ValueError(f"Sweep parameter '{key}' must be a list.")
        if len(v) == 0:
            raise ValueError(f"Sweep parameter '{key}' cannot be empty.")
        result[key] = [_coerce_value(key, item) for item in v]
    return result


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping.")

    raw = {k.strip(): v for k, v in raw.items()}

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in raw:
            raise ValueError(f"Config missing required key: '{key}'")

    operation = str(raw["operation"]).strip()
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(f"Unsupported operation '{operation}'. Supported: {SUPPORTED_OPERATIONS}")

    if not isinstance(raw["fixed"], dict):
        raise ValueError("'fixed' must be a dictionary.")

    fixed = _normalize_fixed(raw["fixed"])
    sweep = _normalize_sweep(raw["sweep"])

    global_settings = {
        "binary": str(raw["binary"]).strip(),
        "output_root": str(raw["output_root"]).strip(),
        "operation": operation,
    }
    if "run_name_template" in raw:
        global_settings["run_name_template"] = str(raw["run_name_template"]).strip()

    config = {"global": global_settings, "fixed": fixed, "sweep": sweep}
    validate_config(config)
    return config


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m sweep.config_parser <config.yaml>")
        sys.exit(1)

    try:
        config = load_config(sys.argv[1])
        g = config["global"]
        print("Config loaded successfully!")
        print(f"  Operation : {g['operation']}")
        print(f"  Binary    : {g['binary']}")
        print(f"  Output    : {g['output_root']}")
        print(f"  Fixed     : {config['fixed']}")
        print(f"  Sweep     : {config['sweep']}")
        total = 1
        for values in config["sweep"].values():
            total *= len(values)
        print(f"  Total runs: {total}")
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
