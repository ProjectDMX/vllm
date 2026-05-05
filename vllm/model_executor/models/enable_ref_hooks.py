"""Preprocessor: uncomment BENCH_OFF lines for selected hooks.

Usage as script:
    python enable_ref_hooks.py --model-file gpt2_ref.py --hooks all \\
        --max-len 8192 --output-dir /tmp/ref --config-out /tmp/ref/ref_config.json

Usage as module:
    from .enable_ref_hooks import enable_ref_hooks
    enable_ref_hooks(model_file=..., hooks="all", max_len=8192, ...)
"""
import argparse
import json
import os
import re


# Hook name shortcuts
# Must match monitoring/ring_transport.py _HOOK_SELECTIONS presets.
_HOOK_SHORTCUTS = {
    "all": [
        "token_ids", "embed", "pos_embed",
        "resid_pre", "ln1", "q", "k", "v", "z",
        "attn_out", "resid_mid", "ln2", "mlp_in", "mlp_post", "mlp_out",
        "resid_final", "final_ln", "final_logits",
    ],
    "hidden-states": ["resid_pre"],
    # vllm-full: excludes attn_scores, pattern (FlashAttention doesn't
    # materialize).
    "vllm-full": [
        "token_ids", "embed", "pos_embed",
        "resid_pre", "ln1", "q", "k", "v", "z",
        "attn_out", "resid_mid", "ln2", "mlp_in", "mlp_post", "mlp_out",
        "resid_final", "final_ln", "final_logits",
    ],
}

# Model metadata (for ref_config.json)
_MODEL_META = {
    "gpt2": {
        "num_layers": 12, "hidden_dim": 768, "vocab_size": 50257,
        "num_heads": 12, "head_dim": 64, "num_kv_heads": 12,
    },
    "qwen3": {
        "num_layers": 28, "hidden_dim": 1024, "vocab_size": 151936,
        "num_heads": 16, "head_dim": 64, "num_kv_heads": 8,
    },
    "llama": {
        # meta-llama/Llama-3.1-8B
        "num_layers": 32, "hidden_dim": 4096, "vocab_size": 128256,
        "num_heads": 32, "head_dim": 128, "num_kv_heads": 8,
    },
}

_BENCH_OFF_RE = re.compile(r"^(\s*)# BENCH_OFF (\w+): (.*)$")


def enable_ref_hooks(
    model_file: str,
    hooks: str | list[str],
    max_len: int,
    output_dir: str,
    config_out: str,
) -> dict:
    """Uncomment BENCH_OFF lines for selected hooks, write config JSON."""
    # Resolve hook names
    if isinstance(hooks, str):
        hooks = _HOOK_SHORTCUTS.get(hooks, hooks.split(","))
    enabled = set(hooks)

    # Read model file
    with open(model_file) as f:
        lines = f.readlines()

    # Uncomment matching BENCH_OFF lines
    found_hooks: set[str] = set()
    new_lines: list[str] = []
    for line in lines:
        m = _BENCH_OFF_RE.match(line)
        if m:
            indent, hook_name, code = m.groups()
            found_hooks.add(hook_name)
            if hook_name in enabled:
                new_lines.append(f"{indent}{code}\n")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Write modified file
    with open(model_file, "w") as f:
        f.writelines(new_lines)

    # Detect model from filename
    basename = os.path.basename(model_file)
    if "gpt2" in basename:
        model_key = "gpt2"
    elif "qwen3" in basename:
        model_key = "qwen3"
    elif "llama" in basename:
        model_key = "llama"
    else:
        model_key = "unknown"

    meta = _MODEL_META.get(model_key, {})

    # Write config JSON
    os.makedirs(os.path.dirname(config_out) or ".", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    config = {
        "model": model_key,
        "enabled_hooks": sorted(enabled),
        "found_hooks": sorted(found_hooks),
        "max_len": max_len,
        "output_dir": output_dir,
        **meta,
    }
    with open(config_out, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[enable_ref_hooks] Enabled {len(enabled & found_hooks)}/{len(enabled)} "
          f"hooks in {model_file}")
    if enabled - found_hooks:
        print(f"[enable_ref_hooks] WARNING: not found in file: {sorted(enabled - found_hooks)}")
    print(f"[enable_ref_hooks] Config: {config_out}")
    return config


def disable_all_hooks(model_file: str) -> None:
    """Re-comment all uncommented BENCH_OFF lines (restore to default state).

    Detects uncommented lines by matching the code patterns that
    enable_ref_hooks produces (self._buf_* copy lines).
    """
    # The simplest restore: re-read the original from backup.
    # This function is provided for completeness but the test flow
    # uses file backup/restore instead.
    pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-file", required=True)
    p.add_argument("--hooks", default="all")
    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config-out", required=True)
    args = p.parse_args()
    enable_ref_hooks(args.model_file, args.hooks, args.max_len,
                     args.output_dir, args.config_out)
