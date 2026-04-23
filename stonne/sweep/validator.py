import os

SUPPORTED_OPERATIONS = ["DenseGEMM"]
DENSEGEMM_FIXED = {"M", "N", "K", "num_ms", "dn_bw", "rn_bw"}
DENSEGEMM_TILES = {"T_M", "T_N", "T_K"}


def _is_power_of_2(n):
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def validate_global(global_settings):
    operation = global_settings["operation"]
    output_root = global_settings["output_root"]

    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(f"Unsupported operation '{operation}'. Supported: {SUPPORTED_OPERATIONS}")

    try:
        os.makedirs(output_root, exist_ok=True)
    except OSError as e:
        raise ValueError(f"Could not create output_root '{output_root}': {e}")

    if not os.access(output_root, os.W_OK):
        raise ValueError(f"output_root '{output_root}' is not writable.")


def validate_hardware(fixed):
    for param in ["num_ms", "dn_bw", "rn_bw"]:
        if param not in fixed:
            raise ValueError(f"Hardware parameter '{param}' is required but missing.")
        val = fixed[param]
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"Hardware parameter '{param}' must be a positive integer, got: {val!r}")
        if not _is_power_of_2(val):
            raise ValueError(f"Hardware parameter '{param}' must be a power of 2, got: {val}")


def validate_operation(fixed, sweep):
    all_keys = set(fixed.keys()) | set(sweep.keys())

    for param in DENSEGEMM_FIXED:
        if param not in fixed:
            raise ValueError(f"DenseGEMM requires '{param}' in 'fixed' but it is missing.")

    for param in DENSEGEMM_TILES:
        if param not in all_keys:
            raise ValueError(f"DenseGEMM requires '{param}' in 'fixed' or 'sweep' but it is missing.")

    for dim in ("M", "N", "K"):
        val = fixed[dim]
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"DenseGEMM requires '{dim}' to be a positive integer, got: {val!r}")


def validate_sweep(sweep):
    if not sweep:
        raise ValueError("'sweep' section is empty. At least one parameter must be swept.")

    total = 1
    for key, values in sweep.items():
        if not isinstance(values, list):
            raise ValueError(f"Sweep parameter '{key}' must be a list.")
        if len(values) == 0:
            raise ValueError(f"Sweep parameter '{key}' cannot be empty.")
        total *= len(values)

    if total == 0:
        raise ValueError("Sweep Cartesian product is zero-sized.")


def validate_config(config):
    validate_global(config["global"])
    validate_hardware(config["fixed"])
    validate_operation(config["fixed"], config["sweep"])
    validate_sweep(config["sweep"])


if __name__ == "__main__":
    import sys
    try:
        from .config_parser import load_config
    except ImportError:
        from config_parser import load_config

    if len(sys.argv) < 2:
        print("Usage: python -m sweep.validator <config.yaml>")
        sys.exit(1)

    try:
        config = load_config(sys.argv[1])
        validate_config(config)
        print("Validation passed.")
    except (FileNotFoundError, ValueError) as e:
        print(f"[VALIDATION ERROR] {e}")
        sys.exit(1)
