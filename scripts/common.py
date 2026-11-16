# Block E shared helpers: ONE post-processing / prompt / validity function used by rewards, eval and baselines.
import os, re, json
W = os.environ.get("RLW", os.path.expandvars("$WORK_ROOT/rlhead"))   # set RLW, or WORK_ROOT (parent of rlhead/)
SYSTEM = "You are a news editor."
USER = "Write one headline for the article below. Reply with the headline only.\n\nARTICLE:\n{article}"

def messages(article):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(article=article)}]

_QUOTES = "\"'“”‘’`*"
def postprocess(text):
    t = text.replace("<|im_end|>", "").strip()
    t = t.strip(_QUOTES + " ")
    t = re.sub(r"^(\*\*)?\s*(headline|title)\s*:\s*(\*\*)?\s*", "", t, flags=re.I)
    t = t.strip().strip(_QUOTES + " ")
    t = t.rstrip()
    return t

def is_valid(h):
    """Shared shaping rule: 4–20 words, no newline, non-empty, non-ASCII ratio <= 20%."""
    if not h or "\n" in h: return False
    n = len(h.split())
    if n < 4 or n > 20: return False
    if sum(ord(c) > 127 for c in h) / max(1, len(h)) > 0.20: return False
    return True

def model_path(role):
    if role == "policy" and os.environ.get("POLICY_PATH"): return os.environ["POLICY_PATH"]   # exploratory scale/family runs
    return os.path.expandvars(json.load(open(f"{W}/data/models.json"))[role]["path"])

def fix_pad(tok):
    """Llama-3.1 has no pad token: use its reserved pad id, never EOS. No-op for Qwen (pad = <|endoftext|>)."""
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>" if "<|finetune_right_pad_id|>" in tok.get_vocab() else tok.eos_token
    return tok

def assert_a100():
    import torch
    name = torch.cuda.get_device_name(0)
    print("GPU:", name, flush=True)
    if "A100" not in name:
        raise SystemExit(f"ABORT: not an A100 ({name})")
    return name
