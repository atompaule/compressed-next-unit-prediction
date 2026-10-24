"""Probe whether handcrafted concept embeddings collapse toward their argmax path.

The analysis mirrors the token-level intervention in Rizvi-Martel et al. (2026):
for an otherwise identical prompt, replace the continuous input with its argmax token
embedding and compare the two paths over transformer depth. It additionally records
the top-10 Logit-Lens outputs after decoder layers 1, L/2, and L.

Outputs are written as JSON, a compact text report, a two-column PDF/PNG figure, and
a LaTeX appendix containing the selected-layer top-10 tables.
"""

from __future__ import annotations

# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.preliminary_experiments.mixture_experiments.concept_descriptions import (
    EXPERIMENTS,
    EXP_CFG,
    build_logits,
    build_vectors,
    single_token_id,
)

DEFAULT_CASES = tuple(experiment["name"] for experiment in EXPERIMENTS)
CONDITIONS = ("Soft Thinking", "Centroid", "Geometric optimizer")
TOP_K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/preliminary_experiments/convergence"),
        help="Directory for JSON, text, PDF, and PNG outputs.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(DEFAULT_CASES),
        help="Exact experiment names from concept_descriptions.py.",
    )
    parser.add_argument(
        "--plot-only-json",
        type=Path,
        default=None,
        help="Regenerate reports and figures from an existing analysis JSON without loading the model.",
    )
    return parser.parse_args()


def display_token(tokenizer, token_id: int) -> str:
    token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    return token.replace("\n", "[newline]").replace("\t", "[tab]")


def selected_experiments(names: list[str]) -> list[dict[str, Any]]:
    by_name = {experiment["name"]: experiment for experiment in EXPERIMENTS}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown cases: {missing}. Available: {sorted(by_name)}")
    return [by_name[name] for name in names]


def build_prompt_embeddings(
    model,
    tokenizer,
    prefix: str,
    suffix: str,
    vector: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    suffix_ids = tokenizer(
        suffix, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    embedding_layer = model.get_input_embeddings()
    concept_position = prefix_ids.shape[1]
    inputs = torch.cat(
        [
            embedding_layer(prefix_ids),
            vector.unsqueeze(0).to(device=device, dtype=model.dtype),
            embedding_layer(suffix_ids),
        ],
        dim=1,
    )
    return inputs, concept_position


def collect_layer_states(
    model,
    inputs_embeds: torch.Tensor,
    concept_position: int,
) -> torch.Tensor:
    """Return concept-position states at input and after each layer: [L+1, D]."""
    captured: list[torch.Tensor | None] = [None] * len(model.model.layers)
    handles = []

    for layer_index, layer in enumerate(model.model.layers):

        def capture(_module, _inputs, output, index=layer_index):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[index] = hidden[0, concept_position].detach().float().cpu()

        handles.append(layer.register_forward_hook(capture))

    attention_mask = torch.ones(
        inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
    )
    try:
        with torch.no_grad():
            model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    if any(state is None for state in captured):
        missing = [i for i, state in enumerate(captured) if state is None]
        raise RuntimeError(f"Failed to capture decoder layers {missing}")

    input_states = inputs_embeds[0, concept_position].detach().float().cpu()
    return torch.stack([input_states, *captured])


def logit_lens_log_probs(
    model, states: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Apply Llama's final RMSNorm and unembedding to all probed states."""
    flat = states.reshape(-1, states.shape[-1]).to(device=device, dtype=model.dtype)
    with torch.no_grad():
        normalized = model.model.norm(flat)
        logits = model.lm_head(normalized).float()
        log_probs = F.log_softmax(logits, dim=-1).cpu()
    return log_probs.reshape(*states.shape[:-1], -1)


def entropy_from_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def token_snapshot(
    tokenizer, log_probs: torch.Tensor, k: int = 5
) -> list[dict[str, Any]]:
    values, indices = torch.topk(log_probs, k=k)
    return [
        {
            "token_id": int(token_id),
            "token": display_token(tokenizer, int(token_id)),
            "probability": float(log_probability.exp()),
        }
        for log_probability, token_id in zip(values, indices)
    ]


def compare_paths(
    argmax_states: torch.Tensor,
    condition_states: torch.Tensor,
    argmax_log_probs: torch.Tensor,
    condition_log_probs: torch.Tensor,
) -> dict[str, Any]:
    cosine = F.cosine_similarity(condition_states, argmax_states, dim=-1)
    condition_probs = condition_log_probs.exp()
    kl = (condition_probs * (condition_log_probs - argmax_log_probs)).sum(dim=-1)
    entropy_difference = entropy_from_log_probs(
        condition_log_probs
    ) - entropy_from_log_probs(argmax_log_probs)
    argmax_top = torch.topk(argmax_log_probs, k=TOP_K, dim=-1).indices
    condition_top = torch.topk(condition_log_probs, k=TOP_K, dim=-1).indices
    top_k_overlap = torch.stack(
        [
            torch.isin(condition_top[layer], argmax_top[layer]).float().mean()
            for layer in range(condition_top.shape[0])
        ]
    )

    layers = list(range(argmax_states.shape[0]))
    return {
        "layers": layers,
        "hidden_cosine": cosine.tolist(),
        "kl_condition_to_argmax_nats": kl.tolist(),
        "entropy_difference_nats": entropy_difference.tolist(),
        "top10_overlap": top_k_overlap.tolist(),
    }


def run_case(model, tokenizer, vocab_embs, experiment, device) -> dict[str, Any]:
    target_ids = [single_token_id(tokenizer, word) for word in experiment["words"]]
    logit_values = torch.tensor(experiment["logits"], device=device)
    full_logits = build_logits(target_ids, logit_values, vocab_embs.shape[0], device)

    torch.manual_seed(EXP_CFG.seed)
    vectors = build_vectors(full_logits, target_ids, vocab_embs)
    condition_data: dict[str, dict[str, torch.Tensor]] = {}

    for condition_name, vector in vectors.items():
        inputs, concept_position = build_prompt_embeddings(
            model,
            tokenizer,
            experiment["prefix"],
            experiment["suffix"],
            vector,
            device,
        )
        states = collect_layer_states(model, inputs, concept_position)
        log_probs = logit_lens_log_probs(model, states, device)
        condition_data[condition_name] = {
            "states": states,
            "log_probs": log_probs,
        }

    argmax_data = condition_data["Argmax"]
    comparisons = {
        condition: compare_paths(
            argmax_data["states"],
            condition_data[condition]["states"],
            argmax_data["log_probs"],
            condition_data[condition]["log_probs"],
        )
        for condition in CONDITIONS
    }
    number_of_layers = len(model.model.layers)
    probe_layers = [1, number_of_layers // 2, number_of_layers]
    top10_by_layer = {
        str(layer): {
            condition: token_snapshot(
                tokenizer, condition_data[condition]["log_probs"][layer], k=TOP_K
            )
            for condition in ("Argmax", *CONDITIONS)
        }
        for layer in probe_layers
    }

    input_probs = F.softmax(full_logits, dim=-1)[target_ids].detach().cpu()
    return {
        "name": experiment["name"],
        "support": [
            {
                "token_id": token_id,
                "token": display_token(tokenizer, token_id),
                "probability": float(probability),
            }
            for token_id, probability in zip(target_ids, input_probs)
        ],
        "concept_position": concept_position,
        "probe_layers": probe_layers,
        "logit_lens_top10": top10_by_layer,
        "comparisons": comparisons,
    }


def write_text_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "CONCEPT-EMBEDDING SUPERPOSITION-COLLAPSE ANALYSIS",
        f"Model: {payload['model_id']}",
        "Comparison: each continuous condition versus the discrete argmax embedding",
        "Layer 0 is the input embedding; layers 1--32 are decoder-layer outputs.",
        "KL direction: D_KL(p_condition || p_argmax), in nats.",
        "",
    ]
    for case in payload["cases"]:
        support = ", ".join(
            f"{item['token'].strip()} ({item['probability']:.4f})"
            for item in case["support"]
        )
        lines.extend(["=" * 100, case["name"], f"Support: {support}"])
        for condition, comparison in case["comparisons"].items():
            lines.append(f"\n{condition}")
            final_cosine = comparison["hidden_cosine"][-1]
            final_kl = comparison["kl_condition_to_argmax_nats"][-1]
            final_entropy = comparison["entropy_difference_nats"][-1]
            final_overlap = comparison["top10_overlap"][-1]
            max_kl = max(comparison["kl_condition_to_argmax_nats"])
            max_kl_layer = comparison["kl_condition_to_argmax_nats"].index(max_kl)
            lines.append(
                f"  final cosine={final_cosine:.6f}, final KL={final_kl:.6g}, "
                f"entropy delta={final_entropy:+.6g}, "
                f"top-10 overlap={final_overlap:.2f}; "
                f"max KL={max_kl:.6g} at layer {max_kl_layer}"
            )
        lines.append("\nSELECTED-LAYER LOGIT-LENS TOP-10")
        for layer in case["probe_layers"]:
            lines.append(f"  Layer {layer}")
            for condition in ("Argmax", *CONDITIONS):
                tokens = ", ".join(
                    f"{entry['token']!r}:{entry['probability']:.4f}"
                    for entry in case["logit_lens_top10"][str(layer)][condition]
                )
                lines.append(f"    {condition}: {tokens}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "%": r"\%",
        "_": r"\_",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def latex_token_cell(entry: dict[str, Any]) -> str:
    raw_token = entry["token"].replace("\r", "[carriage-return]")
    if any(ord(character) > 126 or ord(character) < 32 for character in raw_token):
        raw_token = f"id:{entry['token_id']}"
    elif len(raw_token) > 18:
        raw_token = f"{raw_token[:12]}...[id:{entry['token_id']}]"
    token = latex_escape(raw_token).replace(" ", r"\textvisiblespace{}")
    return rf"\texttt{{{token}}} ({entry['probability']:.3f})"


def write_latex_tables(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "% Generated by analyze_concept_superposition_collapse.py.",
        "% Top-10 Logit-Lens outputs at the injected concept position.",
    ]
    for case_index, case in enumerate(payload["cases"], start=1):
        label = f"tab:concept-logit-lens-{case_index}"
        lines.extend(
            [
                r"\begin{table}[p]",
                r"    \centering",
                rf"    \caption{{Top-10 Logit-Lens outputs for {latex_escape(case['name'])} at the injected concept position. Probabilities are shown in parentheses.}}",
                rf"    \label{{{label}}}",
                r"    \scriptsize",
                r"    \setlength{\tabcolsep}{2.5pt}",
                r"    \renewcommand{\arraystretch}{0.92}",
                r"    \resizebox{\textwidth}{!}{%",
                r"    \begin{tabular}{rllll}",
                r"        \toprule",
                r"        Rank & Argmax & Soft Thinking & Centroid & Geometric optimizer \\",
            ]
        )
        for layer in case["probe_layers"]:
            layer_outputs = case["logit_lens_top10"][str(layer)]
            lines.extend(
                [
                    r"        \midrule",
                    rf"        \multicolumn{{5}}{{c}}{{\textbf{{Decoder layer {layer}}}}} \\",
                    r"        \midrule",
                ]
            )
            for rank in range(TOP_K):
                cells = [
                    latex_token_cell(layer_outputs[condition][rank])
                    for condition in ("Argmax", *CONDITIONS)
                ]
                lines.append(f"        {rank + 1} & " + " & ".join(cells) + r" \\")
        lines.extend(
            [
                r"        \bottomrule",
                r"    \end{tabular}",
                r"    }",
                r"\end{table}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_results(path_pdf: Path, path_png: Path, payload: dict[str, Any]) -> None:
    styles = {
        "Ambiguity: river bank": ("#4D4D4D", "-", "o"),
        "City: Las Vegas": ("#0072B2", "-", "s"),
        "City: Amsterdam": ("#56B4E9", "-", "P"),
        "Mountain: huge": ("#D55E00", "-", "^"),
        "Mountain: tiny": ("#E69F00", "-", "v"),
        "Singer: Amy Winehouse": ("#CC79A7", "-", "D"),
        "Singer: Bob Marley": ("#009E73", "-", "X"),
    }
    cases = payload["cases"]
    maximum_kl = max(
        value
        for case in cases
        for value in case["comparisons"]["Soft Thinking"]["kl_condition_to_argmax_nats"]
    )
    figure, (cosine_axis, kl_axis) = plt.subplots(
        1, 2, figsize=(7.25, 3.65), sharex=True
    )

    for case in cases:
        comparison = case["comparisons"]["Soft Thinking"]
        layers = comparison["layers"]
        color, linestyle, marker = styles[case["name"]]
        shared = {
            "color": color,
            "linestyle": linestyle,
            "linewidth": 1.55,
            "marker": marker,
            "markevery": 4,
            "markersize": 3.2,
            "markerfacecolor": "white",
            "markeredgewidth": 0.8,
        }
        cosine_axis.plot(
            layers,
            comparison["hidden_cosine"],
            label=case["name"],
            **shared,
        )
        kl_axis.plot(
            layers,
            comparison["kl_condition_to_argmax_nats"],
            **shared,
        )

    maximum_layer = max(cases[0]["comparisons"]["Soft Thinking"]["layers"])
    for axis in (cosine_axis, kl_axis):
        axis.set_xlim(0, maximum_layer)
        axis.xaxis.set_major_locator(MultipleLocator(4))
        axis.grid(True, color="#D8DDE3", linewidth=0.55, alpha=0.85)
        axis.tick_params(labelsize=7.5)
        axis.set_xlabel("Input embedding (0) and decoder layer", fontsize=8.5)

    cosine_axis.set_title("Hidden-state cosine similarity", fontsize=9.5)
    cosine_axis.set_ylabel("Cosine similarity", fontsize=8.5)
    cosine_axis.set_ylim(0.0, 1.02)
    cosine_axis.yaxis.set_major_locator(MultipleLocator(0.1))
    cosine_axis.axhspan(0.99, 1.0, color="#B8BDC5", alpha=0.22, zorder=0)
    cosine_axis.text(
        31.6,
        0.986,
        "IOSP: >.99",
        fontsize=6.6,
        color="#555B63",
        ha="right",
        va="top",
    )

    kl_axis.set_title("Logit-Lens distribution difference", fontsize=9.5)
    kl_axis.set_ylabel(
        r"$D_{\mathrm{KL}}(p_{\mathrm{soft}}\,\Vert\,p_{\mathrm{argmax}})$ [nats]",
        fontsize=8.5,
    )
    kl_axis.set_ylim(0.0, max(7.0, maximum_kl * 1.05))
    kl_axis.yaxis.set_major_locator(MultipleLocator(1.0))
    kl_axis.axhline(0.1, color="#6B7078", linewidth=0.9, linestyle=":", zorder=0)
    kl_axis.text(
        31.6,
        0.17,
        "IOSP: at most ~.1",
        fontsize=6.6,
        color="#555B63",
        ha="right",
        va="bottom",
    )

    handles, legend_labels = cosine_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
        fontsize=6.9,
        handlelength=2.6,
        columnspacing=1.2,
    )
    figure.tight_layout(rect=(0.0, 0.17, 1.0, 1.0))
    figure.savefig(path_pdf, bbox_inches="tight")
    figure.savefig(path_png, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    stem = "llama_8b_concept_superposition_collapse"
    json_path = output_dir / f"{stem}.json"
    text_path = output_dir / f"{stem}.txt"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    tex_path = output_dir / f"{stem}_top10.tex"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_text_report(text_path, payload)
    plot_results(pdf_path, png_path, payload)
    write_latex_tables(tex_path, payload)
    print(f"Saved {json_path}")
    print(f"Saved {text_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    print(f"Saved {tex_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.plot_only_json is not None:
        payload = json.loads(args.plot_only_json.read_text(encoding="utf-8"))
        write_outputs(args.output_dir, payload)
        return

    if not torch.backends.mps.is_available():
        raise RuntimeError("This experiment requires an available MPS device.")
    device = torch.device("mps")
    torch.manual_seed(EXP_CFG.seed)

    print(f"Loading {EXP_CFG.model_id} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(
        EXP_CFG.model_id,
        local_files_only=True,
        clean_up_tokenization_spaces=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        EXP_CFG.model_id,
        dtype=torch.float16,
        local_files_only=True,
    ).to(device)
    model.eval()
    vocab_embs = model.get_input_embeddings().weight.detach().float()

    payload = {
        "model_id": EXP_CFG.model_id,
        "device": str(device),
        "seed": EXP_CFG.seed,
        "metric_definition": {
            "hidden_cosine": "cos(h_condition, h_argmax)",
            "kl": "D_KL(p_condition || p_argmax), nats",
            "entropy_difference": "H(p_condition) - H(p_argmax), nats",
            "top10_overlap": "intersection size divided by 10",
            "layer_zero": "input embedding before the first decoder layer",
        },
        "cases": [],
    }
    for experiment in selected_experiments(args.cases):
        print(f"Probing {experiment['name']}")
        payload["cases"].append(
            run_case(model, tokenizer, vocab_embs, experiment, device)
        )

    write_outputs(args.output_dir, payload)


if __name__ == "__main__":
    main()
