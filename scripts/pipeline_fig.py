# Block E: simple pipeline figure (Upworthy -> RM; cc_news -> GRPO arms -> metrics).
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import W
fig, ax = plt.subplots(figsize=(10, 3.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
def box(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="#333", lw=1)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5)
def arr(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.1, color="#333"))
box(0.1, 2.8, 2.0, 0.9, "Upworthy A/B tests\n(real clicks, 2013–15)", "#e8eef7")
box(2.6, 2.8, 2.0, 0.9, "Reward model\n(headline only)", "#e8eef7")
box(0.1, 0.4, 2.0, 0.9, "News articles\n(cc_news, 2017–19)", "#eef7e8")
box(2.6, 0.4, 2.0, 0.9, "Simulated audience\nQwen2.5-7B, 4 personas", "#f7e8e8")
box(5.1, 1.3, 2.1, 1.6, "GRPO headline writer\nQwen2.5-3B + LoRA\n\n(a) real-click RM\n(b) simulated audience\n(c) RM + NLI penalty", "#fff7e0")
box(7.7, 1.3, 2.2, 1.6, "Fabrication check\nvs. the article\n\nentities & numbers,\nNLI, LLM probe,\nhand labels", "#f0f0f0")
arr(2.1, 3.25, 2.6, 3.25); arr(4.6, 3.25, 5.4, 2.9); arr(4.6, 0.85, 5.4, 1.3); arr(2.1, 0.85, 5.1, 2.0); arr(7.2, 2.1, 7.7, 2.1)
for e in ("pdf", "png"): fig.savefig(f"{W}/figs/pipeline.{e}", dpi=200, bbox_inches="tight")
print("PIPELINE_FIG_DONE")
