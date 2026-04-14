"""
06_figure_and_quotes.py
1. Figure: fears text severity → child YSR across timepoints
2. Curated example quotes organized by finding
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

ROOT     = Path(__file__).parent
DATA_IN  = ROOT / "data" / "merged_clean.csv"
DATA_OUT = ROOT / "data"

MISSING = {"", "nan", "none", "0.0", "1.0", "888", "888.0", "999", "999.0"}
clean = lambda v: "" if str(v).strip().lower() in MISSING else str(v).strip()

df = pd.read_csv(DATA_IN)
for col in df.columns:
    if "YSR" in col or "ASR_Internalizing" in col or "ASR_Total" in col:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace({888: np.nan, 999: np.nan})

# ══════════════════════════════════════════════════════════════════════════
# FIGURE: fears text severity across timepoints
# ══════════════════════════════════════════════════════════════════════════
from sentence_transformers import SentenceTransformer
import warnings; warnings.filterwarnings("ignore")

print("Loading model for figure...")
model = SentenceTransformer("all-MiniLM-L6-v2")
fear_lo = model.encode("mild discomfort, slight preference to avoid, not distressing")
fear_hi = model.encode("severe phobia, intense panic, debilitating, avoidance disrupts daily life")

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

fear_texts = df["asr_29_text.T1"].apply(clean)
fear_sev = np.array([
    cosine(model.encode(t), fear_hi) - cosine(model.encode(t), fear_lo)
    if t else np.nan for t in fear_texts
])

# Timepoints to plot
timepoints = [
    ("T1_YSR_Total",         "T1\n(concurrent)"),
    ("T2_YSR_Total",         "T2\n(~1 yr)"),
    ("T3_YSR_Total",         "T3\n(~2 yr)"),
    ("T4_YSR_Total",         "T4\n(~3–4 yr)"),
    ("TA_ASR_Total",         "TA\n(adulthood)"),
]

rs, ps, ns, labels = [], [], [], []
for col, label in timepoints:
    if col not in df.columns:
        continue
    y = df[col].values
    mask = ~np.isnan(fear_sev) & ~np.isnan(y)
    n = mask.sum()
    if n < 8:
        rs.append(np.nan); ps.append(np.nan); ns.append(n); labels.append(label)
        continue
    r, p = stats.pearsonr(fear_sev[mask], y[mask])
    rs.append(r); ps.append(p); ns.append(n); labels.append(label)

# ── Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.2))

colors = []
for r, p in zip(rs, ps):
    if np.isnan(p):
        colors.append("#cccccc")
    elif p < .01:
        colors.append("#d62728")
    elif p < .05:
        colors.append("#ff7f0e")
    elif p < .10:
        colors.append("#aec7e8")
    else:
        colors.append("#c7c7c7")

x = np.arange(len(labels))
bars = ax.bar(x, [r if not np.isnan(r) else 0 for r in rs],
              color=colors, width=0.55, edgecolor="white", linewidth=0.8, zorder=3)

# Annotate with r and sig
for i, (r, p, n) in enumerate(zip(rs, ps, ns)):
    if np.isnan(r):
        continue
    sig = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else ("†" if p < .10 else "")))
    ax.text(i, r + 0.015, f"r={r:.2f}{sig}\nN={n}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold" if p < .05 else "normal")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Pearson r", fontsize=11)
ax.set_ylim(-0.1, 0.65)
ax.set_title("Parent fears/phobias text severity\n→ child/adolescent total problems across timepoints",
             fontsize=11, fontweight="bold", pad=10)

# Legend
patches = [
    mpatches.Patch(color="#d62728",  label="p < .01"),
    mpatches.Patch(color="#ff7f0e",  label="p < .05"),
    mpatches.Patch(color="#aec7e8",  label="p < .10"),
    mpatches.Patch(color="#c7c7c7",  label="ns"),
]
ax.legend(handles=patches, fontsize=8.5, loc="upper left", framealpha=0.85)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3, zorder=0)

plt.tight_layout()
fig_path = DATA_OUT / "figure_fears_timepoints.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Figure saved: {fig_path}")

# ══════════════════════════════════════════════════════════════════════════
# QUOTES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ILLUSTRATIVE QUOTES")
print("=" * 65)

# ── Fears: high child outcomes vs low ─────────────────────────────────────
fear_df = df[["asr_29_text.T1", "asr_29.T1", "T1_YSR_Total", "T4_YSR_Total"]].copy()
fear_df["text"] = fear_df["asr_29_text.T1"].apply(clean)
fear_df = fear_df[fear_df["text"].str.len() > 0]
fear_df["T1_YSR_Total"] = pd.to_numeric(fear_df["T1_YSR_Total"], errors="coerce")
fear_df["T4_YSR_Total"] = pd.to_numeric(fear_df["T4_YSR_Total"], errors="coerce")
fear_df["sev"] = [fear_sev[i] for i in fear_df.index]

fear_t1 = fear_df.dropna(subset=["T1_YSR_Total"]).sort_values("T1_YSR_Total", ascending=False)
med = fear_t1["T1_YSR_Total"].median()

print("\n--- FEARS: parents of higher-problem children (T1 YSR total above median) ---")
for _, r in fear_t1[fear_t1["T1_YSR_Total"] >= med].head(8).iterrows():
    print(f'  [YSR={r["T1_YSR_Total"]:.0f}, score={r["asr_29.T1"]:.0f}] "{r["text"]}"')

print(f"\n--- FEARS: parents of lower-problem children (T1 YSR total below median) ---")
for _, r in fear_t1[fear_t1["T1_YSR_Total"] < med].tail(8).iterrows():
    print(f'  [YSR={r["T1_YSR_Total"]:.0f}, score={r["asr_29.T1"]:.0f}] "{r["text"]}"')

# ── Social/internal vs external fears ─────────────────────────────────────
import re
social_pat = [r"social", r"anxi", r"crowd", r"public", r"confrontation",
              r"infidelity", r"financ", r"unemploy", r"death", r"harm",
              r"alone", r"worry", r"violence", r"police", r"exploit"]
external_pat = [r"spider", r"snake", r"dog", r"bug", r"rat", r"horse",
                r"balloon", r"height", r"water", r"driv", r"lightning",
                r"fire", r"tight", r"new place", r"getting lost"]

fear_df["is_social"] = fear_df["text"].str.lower().apply(
    lambda t: any(re.search(p, t) for p in social_pat))
fear_df["is_external"] = fear_df["text"].str.lower().apply(
    lambda t: any(re.search(p, t) for p in external_pat))

print("\n--- SOCIAL/INTERNAL FEARS (predict child outcomes at T2+) ---")
for _, r in fear_df[fear_df["is_social"]].iterrows():
    t1 = f"T1={r['T1_YSR_Total']:.0f}" if not np.isnan(r["T1_YSR_Total"]) else "T1=?"
    print(f'  [{t1}, score={r["asr_29.T1"]:.0f}] "{r["text"]}"')

print("\n--- EXTERNAL/OBJECT PHOBIAS (don't predict child outcomes) ---")
for _, r in fear_df[fear_df["is_external"] & ~fear_df["is_social"]].iterrows():
    t1 = f"T1={r['T1_YSR_Total']:.0f}" if not np.isnan(r["T1_YSR_Total"]) else "T1=?"
    print(f'  [{t1}, score={r["asr_29.T1"]:.0f}] "{r["text"]}"')

# ── Sleep: maintenance vs other ────────────────────────────────────────────
sleep_df = df[["asr_100_text.T1", "T1_YSR_Internalizing"]].copy()
sleep_df["text"] = sleep_df["asr_100_text.T1"].apply(clean)
sleep_df = sleep_df[sleep_df["text"].str.len() > 0]
sleep_df["T1_YSR_Internalizing"] = pd.to_numeric(sleep_df["T1_YSR_Internalizing"], errors="coerce")

maint_pat = [r"wake", r"wak", r"middle.*night", r"stay asleep", r"through.*night", r"back asleep"]
sleep_df["maintenance"] = sleep_df["text"].str.lower().apply(
    lambda t: any(re.search(p, t) for p in maint_pat))

sleep_sub = sleep_df.dropna(subset=["T1_YSR_Internalizing"]).sort_values(
    "T1_YSR_Internalizing", ascending=False)

print("\n--- SLEEP MAINTENANCE descriptions (trend toward higher child internalizing) ---")
for _, r in sleep_sub[sleep_sub["maintenance"]].head(8).iterrows():
    print(f'  [YSR-Int={r["T1_YSR_Internalizing"]:.0f}] "{r["text"]}"')

print("\n--- OTHER SLEEP descriptions (onset, physical, vague) ---")
for _, r in sleep_sub[~sleep_sub["maintenance"]].head(8).iterrows():
    print(f'  [YSR-Int={r["T1_YSR_Internalizing"]:.0f}] "{r["text"]}"')

print(f"\nAll outputs saved to {DATA_OUT}/")
print("Done.")
