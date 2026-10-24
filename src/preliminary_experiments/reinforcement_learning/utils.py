import os
import re

import torch


def log(seq_log, s: str = ""):
    print(s)
    seq_log.write(s + "\n")
    seq_log.flush()


def log_rollouts(seq_log, tokenizer, rollouts, rewards, question, answer, step):
    SEP = "─" * 80

    gt_match = re.search(r"####\s*([\d,]+)", answer)
    gt_str = gt_match.group(1) if gt_match else (str(answer).strip() or "?")
    emoji_row = "  ".join("✅" if r > 0.0 else "❌" for r in rewards)

    log(seq_log, f"\n{SEP}")
    log(seq_log, f"step {step:5d}  |  GT: {gt_str}  |  {emoji_row}")
    log(seq_log, f"Q: {question[:200].replace(chr(10), ' ')}")

    for i, r in enumerate(rollouts):
        tag = "✅" if rewards[i] > 0.0 else "❌"
        n = len(r["concept_vecs"])
        log(
            seq_log,
            f"\n  [{i}] {tag}  Sequence Length: {n}  (Halted by: {r['stop_reason']})",
        )

        prompt_ids = r.get("prompt_ids")
        if prompt_ids is not None:
            full_prompt_text = tokenizer.decode(
                prompt_ids[0].tolist(), skip_special_tokens=False
            )
            disp = full_prompt_text.replace("\n", "↵\n")
            indented = "\n           ".join(disp.splitlines())
            log(
                seq_log,
                f"    📥 Full prompt (raw, special tokens included):\n           {indented}",
            )

        if n > 0:
            log(seq_log, f"    🧠 Concept path ({n} steps):")
            line = "       "
            for step_idx, (ids, probs) in enumerate(
                zip(r["sample_ids_list"], r["sample_probs_list"])
            ):
                # for approaches without discrete sampled tokens; fall back to argmax
                if ids is None:
                    argmax_ids = r.get("argmax_ids_list") or []
                    if step_idx < len(argmax_ids):
                        line += (
                            tokenizer.decode([argmax_ids[step_idx]])
                            .replace("\n", "↵")
                            .replace("\t", "⇥")
                        )
                    continue
                top_tok = (
                    tokenizer.decode([ids[0].item()])
                    .replace("\n", "↵")
                    .replace("\t", "⇥")
                )
                alts = [
                    tokenizer.decode([tid]).replace("\n", "↵").replace("\t", "⇥")
                    for tid in ids[1:].tolist()
                ]
                alt_str = "(" + "|".join(alts) + ")" if alts else ""
                line += top_tok + alt_str
            log(seq_log, line)
        else:
            log(seq_log, f"    🧠 Concept path: (no steps taken)")

        prefix_ids = r.get("prefix_ids")
        if prefix_ids is not None:
            prefix_text = tokenizer.decode(
                prefix_ids[0].tolist(), skip_special_tokens=True
            )
        else:
            prefix_text = ""
        gen_text = tokenizer.decode(r.get("answer_ids", []), skip_special_tokens=True)

        prefix_disp = prefix_text.replace("\n", "↵")
        marked = f"«FORCED:{prefix_disp}»{gen_text}" if prefix_text else gen_text
        marked = marked.strip() if marked.strip() else "(empty)"
        indented = "\n           ".join(marked.splitlines())
        log(
            seq_log,
            f"    🗣️  Answer (stop_reason={r['stop_reason']}, «FORCED:...» = injected prefix):\n           {indented}",
        )

    log(seq_log, SEP)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m:02d}m"


def _latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    steps = [
        d
        for d in os.listdir(ckpt_dir)
        if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_dir, d))
    ]
    if not steps:
        return None
    latest = max(steps, key=lambda s: int(s.split("_")[1]))
    return os.path.join(ckpt_dir, latest)
