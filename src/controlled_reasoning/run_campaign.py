"""Reproduce final logical-sentence conditions sequentially; dry-run by default."""

# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import argparse
import os
from pathlib import Path
import subprocess
import sys

from src.common.experiment_paths import artifact_dir
from conditions import run_name


def commands(args):
    source = Path(__file__).resolve().parent
    root = artifact_dir(__file__)
    for n in args.conclusions:
        if args.stage == "extension" and n != 6:
            continue
        for layers in args.layers:
            for seed in args.seeds:
                for kind in args.conditions:
                    model_kind = "discrete" if kind == "discrete" else "sentence"
                    prompt_mode = "chunk" if kind in ("chunked_prompt", "chunked_prompt_fixed_rates") else "token"
                    fixed_rates = kind in ("fixed_rates", "chunked_prompt_fixed_rates")
                    name = run_name(
                        model_kind, n, seed, layers, kind == "nope", prompt_mode,
                        fixed_rates,
                    )
                    out = (
                        root
                        / ("extended_full" if args.stage == "extension" else "full")
                        / name
                    )
                    if out.exists():
                        raise FileExistsError(f"Existing run is protected: {out}")
                    script = (
                        "continue_full.py" if args.stage == "extension" else "full.py"
                    )
                    cmd = [
                        sys.executable,
                        str(source / script),
                        "--kind",
                        model_kind,
                        "--conclusions",
                        str(n),
                        "--layers",
                        str(layers),
                        "--seed",
                        str(seed),
                    ]
                    if kind == "nope":
                        cmd.append("--no-token-pe")
                    if prompt_mode == "chunk":
                        cmd.extend(["--prompt-mode", "chunk"])
                    if fixed_rates:
                        cmd.append("--fixed-rotation-rates")
                    yield cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["initial", "extension"], default="initial")
    p.add_argument("--conclusions", nargs="+", type=int, choices=[3, 6], default=[3, 6])
    p.add_argument("--layers", nargs="+", type=int, choices=[2, 4], default=[2, 4])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(101, 106)))
    p.add_argument(
        "--conditions",
        nargs="+",
        choices=["discrete", "sentence", "nope", "chunked_prompt",
                 "fixed_rates", "chunked_prompt_fixed_rates"],
        default=["discrete", "sentence", "nope"],
    )
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    jobs = list(commands(args))  # Validate every destination before launching anything.
    import shlex

    for cmd in jobs:
        print(shlex.join(cmd), flush=True)
        if args.execute:
            subprocess.run(
                cmd, check=True, env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0"}
            )


if __name__ == "__main__":
    main()
