"""
08_example_sentences.py
Figure panels showing example quotes from each contrast group,
laid out as clean typographic quote cards.
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(__file__).parent
DATA_IN = ROOT / "data" / "merged_clean.csv"
DATA_OUT = ROOT / "data"

MISSING = {"", "nan", "none", "0.0", "1.0", "888", "888.0", "999", "999.0"}
clean = lambda v: "" if str(v).strip().lower() in MISSING else str(v).strip()

df = pd.read_csv(DATA_IN)
for col in ["T1_YSR_Total", "T1_YSR_Internalizing", "T2_YSR_Total", "T2_YSR_Internalizing"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace({888: np.nan, 999: np.nan})

# ── Helpers ────────────────────────────────────────────────────────────────
social_pat  = [r"social", r"anxi", r"crowd", r"public", r"confrontation",
               r"infidelity", r"financ", r"unemploy", r"death", r"harm",
               r"alone", r"worry", r"violence", r"police", r"exploit",
               r"judg", r"embarrass", r"reject", r"panic", r"stress"]
external_pat = [r"spider", r"snake", r"dog", r"bug", r"rat", r"horse",
                r"balloon", r"height", r"water", r"driv", r"lightning",
                r"fire", r"tight", r"new place", r"getting lost",
                r"blood", r"needle", r"vomit", r"emetophob"]
maint_pat   = [r"wake", r"wak", r"middle.*night", r"stay asleep",
               r"through.*night", r"back asleep", r"back to sleep"]

def wrap(text, width=58):
    """Wrap text to width characters."""
    words = text.split()
    lines, line = [], []
    for w in words:
        if sum(len(x)+1 for x in line) + len(w) > width:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)

def truncate(text, maxlen=120):
    return text if len(text) <= maxlen else text[:maxlen].rsplit(" ", 1)[0] + "…"

# ══════════════════════════════════════════════════════════════════════════
# Collect quotes
# ══════════════════════════════════════════════════════════════════════════

# --- Fears: high vs low YSR ---
fears = df[["asr_29_text.T1", "T1_YSR_Total"]].copy()
fears["text"] = fears["asr_29_text.T1"].apply(clean)
fears = fears[fears["text"].str.len() > 2].dropna(subset=["T1_YSR_Total"])
med = fears["T1_YSR_Total"].median()

fears_hi = (fears[fears["T1_YSR_Total"] >= med]
            .sort_values("T1_YSR_Total", ascending=False)
            ["text"].tolist())
fears_lo = (fears[fears["T1_YSR_Total"] < med]
            .sort_values("T1_YSR_Total")
            ["text"].tolist())

# --- Fears: social vs external ---
fears_all = df[["asr_29_text.T1"]].copy()
fears_all["text"] = fears_all["asr_29_text.T1"].apply(clean)
fears_all = fears_all[fears_all["text"].str.len() > 2]
fears_all["is_social"]   = fears_all["text"].str.lower().apply(
    lambda t: any(re.search(p, t) for p in social_pat))
fears_all["is_external"] = fears_all["text"].str.lower().apply(
    lambda t: any(re.search(p, t) for p in external_pat))

social_quotes   = fears_all[fears_all["is_social"]]["text"].tolist()
external_quotes = fears_all[fears_all["is_external"] & ~fears_all["is_social"]]["text"].tolist()

# --- Sleep: maintenance vs other ---
sleep = df[["asr_100_text.T1", "T1_YSR_Internalizing"]].copy()
sleep["text"] = sleep["asr_100_text.T1"].apply(clean)
sleep = sleep[sleep["text"].str.len() > 2]
sleep["maintenance"] = sleep["text"].str.lower().apply(
    lambda t: any(re.search(p, t) for p in maint_pat))

sleep_maint = sleep[sleep["maintenance"]]["text"].tolist()
sleep_other = sleep[~sleep["maintenance"]]["text"].tolist()

# ══════════════════════════════════════════════════════════════════════════
# Figure 1: Fears — High vs Low child YSR
# ══════════════════════════════════════════════════════════════════════════
def quote_panel(ax, quotes, title, color, n_show=8, max_chars=110):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title bar
    ax.add_patch(plt.Rectangle((0, 0.93), 1, 0.07, color=color, transform=ax.transAxes, clip_on=False))
    ax.text(0.5, 0.965, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white", transform=ax.transAxes)

    selected = [q for q in quotes if len(q) > 10][:n_show]
    n = len(selected)
    row_h = 0.9 / max(n, 1)

    for i, q in enumerate(selected):
        y = 0.90 - i * row_h - row_h * 0.5
        # bullet
        ax.text(0.02, y, "\u2022", ha="left", va="center",
                fontsize=11, color=color, transform=ax.transAxes)
        # quote text
        txt = truncate(q, max_chars)
        ax.text(0.06, y, f'"{txt}"', ha="left", va="center",
                fontsize=8.5, color="#222222", transform=ax.transAxes,
                wrap=True, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.15", fc="#f9f9f9", ec="none", alpha=0.0))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
quote_panel(ax1, fears_hi, "Fears — parents of HIGHER-problem children  (YSR ≥ median)", "#d62728")
quote_panel(ax2, fears_lo, "Fears — parents of LOWER-problem children   (YSR < median)", "#1f77b4")
fig1.suptitle("Parent free-text: 'I am afraid of certain animals, situations, or places (describe)'\n"
              "ASR item 29  ·  grouped by child's T1 YSR Total score",
              fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout(rect=[0, 0, 1, 0.98])
out1 = DATA_OUT / "quotes_fears_hilo.png"
fig1.savefig(out1, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ══════════════════════════════════════════════════════════════════════════
# Figure 2: Social/internal fears vs External phobias
# ══════════════════════════════════════════════════════════════════════════
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 7))
quote_panel(ax3, social_quotes,
            "Social / internal fears  (predict child outcomes at T2–T4)",
            "#7b2d8b", n_show=9)
quote_panel(ax4, external_quotes,
            "External / object phobias  (do not predict child outcomes)",
            "#2ca02c", n_show=9)
fig2.suptitle("Two types of fear language in parent free-text\n"
              "Social/internal fears → child problems years later; object phobias do not",
              fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout(rect=[0, 0, 1, 0.98])
out2 = DATA_OUT / "quotes_fears_type.png"
fig2.savefig(out2, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ══════════════════════════════════════════════════════════════════════════
# Figure 3: Sleep — maintenance vs other
# ══════════════════════════════════════════════════════════════════════════
fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 6))
quote_panel(ax5, sleep_maint,
            f"Sleep maintenance problems  (n = {len(sleep_maint)})",
            "#8c564b", n_show=8)
quote_panel(ax6, sleep_other,
            f"Other sleep descriptions  (onset, physical, vague)  (n = {len(sleep_other)})",
            "#7f7f7f", n_show=8)
fig3.suptitle("Parent free-text: 'I have trouble sleeping (describe)'\n"
              "Sleep maintenance descriptions trend toward higher child internalizing",
              fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout(rect=[0, 0, 1, 0.98])
out3 = DATA_OUT / "quotes_sleep.png"
fig3.savefig(out3, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")

# ══════════════════════════════════════════════════════════════════════════
# Also print to console for copy-paste
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SOCIAL / INTERNAL FEARS (predict child outcomes)")
print("="*70)
for q in social_quotes:
    print(f'  • "{q}"')

print("\n" + "="*70)
print("EXTERNAL / OBJECT PHOBIAS (don't predict child outcomes)")
print("="*70)
for q in external_quotes:
    print(f'  • "{q}"')

print("\n" + "="*70)
print("SLEEP MAINTENANCE")
print("="*70)
for q in sleep_maint[:12]:
    print(f'  • "{q}"')

print("\n" + "="*70)
print("FEARS — HIGH-PROBLEM CHILDREN (YSR ≥ median)")
print("="*70)
for q in fears_hi[:10]:
    print(f'  • "{q}"')

print("\n" + "="*70)
print("FEARS — LOW-PROBLEM CHILDREN (YSR < median)")
print("="*70)
for q in fears_lo[:10]:
    print(f'  • "{q}"')
