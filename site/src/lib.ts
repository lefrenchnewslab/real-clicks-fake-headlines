// Everything the page draws comes from data.json, built from the project's saved results
// (training reward logs, checkpoint evaluations, final test outputs, reference labels).
import raw from "./data.json";

export const D = raw as any;

export type Arm = "a" | "b" | "c";
export type Seed = 0 | 1;
export const ARMS: { key: Arm; name: string; short: string; hint: string }[] = [
  { key: "a", name: "Click predictor", short: "Clicks", hint: "A DeBERTa model fitted to 47,932 pairs of headlines from randomised Upworthy tests" },
  { key: "b", name: "Simulated audience", short: "Audience", hint: "Qwen2.5-7B asked “Would you click it?” as four reader personas" },
  { key: "c", name: "Clicks + faithfulness penalty", short: "Penalty", hint: "The click predictor minus a penalty when an NLI model finds the headline unsupported by the article" },
];
export const armName = (a: Arm) => ARMS.find((x) => x.key === a)!.name;

/** One hue, as Ledger allows: the click predictor sits at --deep, the audience at --late, the penalty between. */
export const ARM_COLOR: Record<Arm | "base", string> = {
  a: "var(--deep)",
  b: "var(--late)",
  c: "color-mix(in oklch, var(--deep) 45%, var(--wash2))",
  base: "var(--ink3)",
};
export const run = (a: Arm, s: Seed) => `main_${a}_s${s}`;

/** Reaction words of the paper's lexicon (lemmas), matched loosely on word starts. */
const REACT = ["controvers", "debate", "backlash", "outrage", "uproar", "slam", "blast", "spark", "fury", "critici", "viral"];
export type Seg = { t: string; kind: "" | "num" | "react" };

/** Splits a headline into plain text, numbers and reaction words, for marking. */
export function marks(h: string, nums: string[] | null = null): Seg[] {
  const out: Seg[] = [];
  const re = /(\d[\d,.]*(?:-[a-z]+)?|[A-Za-z’']+)/g;
  let last = 0, m: RegExpExecArray | null;
  while ((m = re.exec(h))) {
    if (m.index > last) out.push({ t: h.slice(last, m.index), kind: "" });
    const w = m[0];
    let kind: Seg["kind"] = "";
    if (/^\d/.test(w)) {
      const core = w.replace(/-[a-z]+$/, "").replace(/[,.]$/, "");
      if (nums === null || nums.some((n) => core === n || core.replace(/,/g, "") === String(n).replace(/,/g, ""))) kind = "num";
    } else if (REACT.some((r) => w.toLowerCase().startsWith(r))) kind = "react";
    out.push({ t: w, kind });
    last = m.index + w.length;
  }
  if (last < h.length) out.push({ t: h.slice(last), kind: "" });
  return out;
}

export const pct = (x: number, d = 0) => `${(x * 100).toFixed(d)}%`;
export const fmt = (x: number, d = 2) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);

/** Training reward log for one run: [step, reward_mean, distinct_per_group, valid_rate, examples]. */
export type TrainRow = [number, number, number, number, string[]];
export const train = (a: Arm, s: Seed): TrainRow[] => D.train[run(a, s)];

/** Checkpoint evaluations (300 test articles × 2 headlines). */
export type Traj = { run: string; arm: string; seed: number; step: number; kl_seq: number; click_model_z: number; audience_z: number; distinct_per_600: number; mean_words: number; invented_reactions_pct: number; invented_numbers_pct: number; off_article_pct: number; entity_check_pct: number };
export const traj = (r: string): Traj[] => (D.traj as Traj[]).filter((t) => t.run === r).sort((x, y) => x.step - y.step);

/** Depth ramp along training: violet at step 0, cyan at the last step. */
export const stepTint = (step: number, max = 350) => `color-mix(in oklch, var(--deep) ${Math.round((1 - step / max) * 100)}%, var(--late))`;
