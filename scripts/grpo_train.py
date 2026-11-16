# Block E GRPO trainer (identical recipe across arms). Resumable; wall-clock guard exits 0; pre-registered collapse stop.
import argparse, glob, json, os, re, sys, time
START = time.time()
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from common import fix_pad, W, model_path, messages, assert_a100
from rewards import ArmReward

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--name", required=True)
ap.add_argument("--stop_at", type=int, default=0, help="pilot: simulate the guard at this step")
ap.add_argument("--guard_min", type=float, default=105.0); ap.add_argument("--no_gc", action="store_true")
a = ap.parse_args()
gpu = assert_a100()
OUT = f"{W}/runs/grpo/{a.name}"; os.makedirs(OUT, exist_ok=True)
def ck_step(p): return int(re.search(r"checkpoint-(\d+)$", p).group(1))
cks = sorted(glob.glob(f"{OUT}/checkpoint-*"), key=ck_step)
last = cks[-1] if cks else None
if os.path.exists(f"{OUT}/COLLAPSED"): print("COLLAPSED marker present; nothing to do"); sys.exit(0)
if last and ck_step(last) >= a.steps: print(f"TARGET_REACHED {last}"); sys.exit(0)
print("resume from", last, flush=True)

arts = __import__("pandas").read_parquet(f"{W}/data/articles.parquet"); tr = arts[arts.split == "train"]
ds = Dataset.from_dict({"prompt": [messages(x) for x in tr.article_trunc], "article": list(tr.article_trunc)})
p = model_path("policy")
tok = AutoTokenizer.from_pretrained(p); tok.padding_side = "left"; fix_pad(tok)
model = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16)
peft_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
R = ArmReward(a.arm, f"{OUT}/reward_log.jsonl")
mb, acc = (8, 8) if a.no_gc else (16, 4)
cfg_kw = dict(output_dir=OUT, per_device_train_batch_size=mb, gradient_accumulation_steps=acc, steps_per_generation=acc,
              num_generations=8, max_completion_length=32, temperature=1.0, top_p=1.0, top_k=0,
              learning_rate=a.lr, lr_scheduler_type="constant_with_warmup", warmup_steps=10, beta=0.04, loss_type="dapo",
              scale_rewards="group", num_iterations=1, mask_truncated_completions=False, max_steps=a.steps, save_steps=50,
              logging_steps=1, bf16=True, gradient_checkpointing=not a.no_gc, seed=a.seed, report_to="none",
              generation_kwargs={"use_cache": True}, remove_unused_columns=False, shuffle_dataset=True, save_strategy="steps")
cfg = GRPOConfig(**cfg_kw)
json.dump({**{k: (v if isinstance(v, (int, float, str, bool, type(None), dict, list)) else str(v)) for k, v in cfg_kw.items()},
           "generation_batch_size": cfg.generation_batch_size, "steps_per_generation_resolved": cfg.steps_per_generation,
           "gpu": gpu, "arm": a.arm, "args": vars(a)}, open(f"{OUT}/config_{int(START)}.json", "w"), indent=1, default=str)
print("generation_batch_size", cfg.generation_batch_size, "steps_per_generation", cfg.steps_per_generation, flush=True)

timing = open(f"{OUT}/timing.jsonl", "a")
class Ctl(TrainerCallback):
    def __init__(self): self.t = None; self.bad = 0; self.win = []
    def on_step_begin(self, args, state, control, **kw): self.t = time.time(); R.cur_step = state.global_step + 1
    def on_step_end(self, args, state, control, **kw):
        s = state.global_step
        rec = {"step": s, "step_s": round(time.time() - self.t, 3), "gen_score_s": round(getattr(tr_, "_gs_time", 0.0), 3),
               "reward_s": round(R.t, 3), "elapsed_min": round((time.time() - START) / 60, 2)}
        rec["gen_s"] = round(rec["gen_score_s"] - rec["reward_s"], 3); rec["train_s"] = round(rec["step_s"] - rec["gen_score_s"], 3)
        timing.write(json.dumps(rec) + "\n"); timing.flush()
        last = R.last
        self.win.append((last["valid_rate"], last["distinct_per_group_min"], last["distinct_per_group"]))
        if s % 10 == 0:
            v = sum(x[0] for x in self.win) / len(self.win); d = sum(x[2] for x in self.win) / len(self.win)
            self.bad = self.bad + 1 if (v < 0.8 or d < 2) else 0; self.win = []
            print(f"WINDOW step {s} valid {v:.3f} distinct {d:.2f} bad_windows {self.bad}", flush=True)
            if self.bad >= 3:
                open(f"{OUT}/COLLAPSED", "w").write(f"collapsed at step {s}\n"); control.should_save = True; control.should_training_stop = True
        if (a.stop_at and s == a.stop_at) or (time.time() - START) / 60 > a.guard_min:
            print(f"GUARD_STOP at step {s} elapsed {(time.time()-START)/60:.1f} min", flush=True)
            open(f"{OUT}/guard_stops.txt", "a").write(f"{s}\n"); control.should_save = True; control.should_training_stop = True
        return control

tr_ = GRPOTrainer(model=model, reward_funcs=[R], args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_config, callbacks=[Ctl()])
_orig = tr_._generate_and_score_completions
def _timed(*x, **k):
    t = time.time(); r = _orig(*x, **k); tr_._gs_time = time.time() - t; return r
tr_._generate_and_score_completions = _timed
print("trainable params", sum(p.numel() for p in tr_.model.parameters() if p.requires_grad), flush=True)
tr_.train(resume_from_checkpoint=last)
gs = tr_.state.global_step
if gs >= a.steps and not glob.glob(f"{OUT}/checkpoint-{gs}"): tr_.save_model(f"{OUT}/checkpoint-{gs}")
print(f"END step {gs} / {a.steps} elapsed {(time.time()-START)/60:.1f} min peak_mem_GB {torch.cuda.max_memory_allocated()/1e9:.1f}", flush=True)
if gs >= a.steps: open(f"{OUT}/DONE", "w").write(f"{gs}\n")
