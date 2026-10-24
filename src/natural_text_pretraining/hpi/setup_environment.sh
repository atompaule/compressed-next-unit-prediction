#!/bin/bash
# Called inside the manually submitted compute job; no resource submission here.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" && "$(hostname -s)" != lx* ]] || {
  echo "Environment setup requires a Slurm compute job" >&2; exit 1;
}
source_dir=$(realpath "$1")
environment=$(realpath -m "$2")
uv_bin="${UV_BIN:-$HOME/.local/bin/uv}"
[[ -x "$uv_bin" ]] || { echo "Missing uv: $uv_bin" >&2; exit 1; }
mkdir -p "$environment/tmp"
export UV_PYTHON_INSTALL_DIR="$environment/python"
export UV_PYTHON_PREFERENCE=only-managed
export TMPDIR="$environment/tmp"
if [[ ! -x "$environment/venv/bin/python" ]]; then
  "$uv_bin" python install 3.11
  "$uv_bin" venv --python 3.11 "$environment/venv"
fi
"$uv_bin" pip sync --no-cache --python "$environment/venv/bin/python" \
  "$source_dir/requirements-linux.txt"
"$uv_bin" pip freeze --python "$environment/venv/bin/python" > "$environment/installed.txt"
"$environment/venv/bin/python" - <<'PY'
import torch
import skops.io
import spacy
from wtpsplit import SaT
from transformers import AutoTokenizer
assert torch.cuda.is_available(), "CUDA unavailable in allocated job"
x = torch.ones(32, device="cuda")
assert float((x @ x).cpu()) == 32
print({"torch": torch.__version__, "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name()}, flush=True)
nlp = spacy.load("en_core_web_sm", disable=["ner"])
doc = nlp("This sentence checks the dependency parser.")
assert doc.has_annotation("DEP") and doc.has_annotation("POS")
print("Segmentation imports and spaCy parser check passed", flush=True)
PY
