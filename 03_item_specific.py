"""
03_item_specific.py
Item-specific approach: tailor embeddings to each item's content domain,
group by theoretical syndrome, correlate with matched child outcomes.

Run after 01_data_prep.py.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sentence_transformers import SentenceTransformer

ROOT    = Path(__file__).parent
DATA_IN = ROOT / "data" / "merged_clean.csv"
DATA_OUT = ROOT / "data"

# ── Item-specific severity anchors ────────────────────────────────────────
# Each item gets its own hi/lo anchors matched to its clinical content.
# This should be more discriminating than generic "severe / not at all".
ITEM_ANCHORS = {
    "asr_9_text.T1": {
        "label": "Obsessive thoughts",
        "lo": "passing thoughts, normal everyday worries, easily dismissed",
        "hi": "intrusive uncontrollable obsessive thoughts, can't stop thinking, severe rumination",
    },
    "asr_29_text.T1": {
        "label": "Fears / phobias",
        "lo": "mild discomfort, slight preference to avoid, not distressing",
        "hi": "severe phobia, intense panic, avoidance significantly disrupts daily life",
    },
    "asr_40_text.T1": {
        "label": "Voices / perceptual experiences",
        "lo": "brief unusual perception, easily explained, not distressing",
        "hi": "persistent auditory hallucinations, voices commanding or threatening, very distressing",
    },
    "asr_46_text.T1": {
        "label": "Body tics / twitches",
        "lo": "occasional minor twitch, barely noticeable, not disruptive",
        "hi": "frequent uncontrollable tics, visible to others, significantly disruptive",
    },
    "asr_58_text.T1": {
        "label": "Skin picking",
        "lo": "occasional mild picking, no injury, brief habit",
        "hi": "compulsive skin picking causing wounds, scars, very difficult to stop",
    },
    "asr_84_text.T1": {
        "label": "Strange behaviors",
        "lo": "quirky personal preference, harmless eccentricity",
        "hi": "highly unusual behavior others find alarming, socially impairing, bizarre",
    },
    "asr_85_text.T1": {
        "label": "Strange thoughts",
        "lo": "unconventional opinion, creative or unusual perspective",
        "hi": "paranoid or delusional thinking, thoughts others would find very disturbing or psychotic",
    },
    "asr_92_text.T1": {
        "label": "Legal trouble",
        "lo": "minor infraction, jaywalking, speeding ticket",
        "hi": "serious criminal behavior, assault, theft, significant legal consequences",
    },
    "asr_100_text.T1": {
        "label": "Sleep trouble",
        "lo": "occasional mild difficulty falling asleep, slight fatigue",
        "hi": "severe chronic insomnia, unable to sleep, exhausted, major daily impairment",
    },
}

# ── Theoretical syndrome groupings ────────────────────────────────────────
# Based on ASR subscale structure (Achenbach & Rescorla, 2003)
INTERNALIZING_ITEMS = [
    "asr_9_text.T1",   # Anxious/Depressed → obsessive thoughts
    "asr_29_text.T1",  # Anxious/Depressed → phobias
    "asr_46_text.T1",  # Somatic complaints → body twitches
    "asr_58_text.T1",  # Obsessive-compulsive → skin picking
    "asr_85_text.T1",  # Thought problems → strange thoughts (more internalizing)
    "asr_100_text.T1", # Somatic → sleep
]
EXTERNALIZING_ITEMS = [
    "asr_84_text.T1",  # Rule-breaking → strange behaviors
    "asr_92_text.T1",  # Rule-breaking → law trouble
]
THOUGHT_ITEMS = [
    "asr_40_text.T1",  # Thought problems → voices
    "asr_84_text.T1",  # Thought problems → strange behaviors
    "asr_85_text.T1",  # Thought problems → strange thoughts
]

OUTCOMES = {
    "T1_YSR_Internalizing": "T1 Internalizing (concurrent)",
    "T1_YSR_Externalizing": "T1 Externalizing (concurrent)",
    "T1_YSR_Total":         "T1 Total (concurrent)",
    "T2_YSR_Internalizing": "T2 Internalizing (~1yr prospective)",
    "T2_YSR_Externalizing": "T2 Externalizing (~1yr prospective)",
    "T2_YSR_Total":         "T2 Total (~1yr prospective)",
}

# ── Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_IN)
print(f"Loaded {len(df)} participants")

MISSING = {"", "nan", "none", "888", "888.0", "999", "999.0"}

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def item_sev(text, anchor_lo, anchor_hi):
    s = str(text).strip().lower()
    if s in MISSING:
        return np.nan
    e = model.encode(str(text))
    return cosine(e, anchor_hi) - cosine(e, anchor_lo)

# ══════════════════════════════════════════════════════════════════════════
# Step 1: Per-item severity scores with matched anchors
# ══════════════════════════════════════════════════════════════════════════
print("\nComputing item-specific severity scores...")

emb_cache = {}  # item → (anchor_lo_emb, anchor_hi_emb)
for col, anchors in ITEM_ANCHORS.items():
    emb_cache[col] = (
        model.encode(anchors["lo"]),
        model.encode(anchors["hi"]),
    )

sev_cols = {}
for col, anchors in ITEM_ANCHORS.items():
    a_lo, a_hi = emb_cache[col]
    sev = df[col].apply(lambda t: item_sev(t, a_lo, a_hi)).values
    sev_cols[col] = sev
    n_valid = int(~np.isnan(sev).sum())
    n = int(np.sum(~np.isnan(sev)))
    print(f"  {col.split('_text')[0]:12s}: {n} responses  "
          f"M={np.nanmean(sev):.4f}  SD={np.nanstd(sev):.4f}")

# ══════════════════════════════════════════════════════════════════════════
# Step 2: Syndrome composites
# ══════════════════════════════════════════════════════════════════════════
def nanmean_cols(cols, sev_dict):
    """Row-wise nanmean of selected severity columns."""
    mat = np.column_stack([sev_dict[c] for c in cols])
    result = np.nanmean(mat, axis=1)
    # NaN if ALL items are missing
    all_missing = np.all(np.isnan(mat), axis=1)
    result[all_missing] = np.nan
    return result

df["sev_internalizing"] = nanmean_cols(INTERNALIZING_ITEMS, sev_cols)
df["sev_externalizing"] = nanmean_cols(EXTERNALIZING_ITEMS, sev_cols)
df["sev_thought"]       = nanmean_cols(THOUGHT_ITEMS, sev_cols)

# Also add individual item scores
for col in ITEM_ANCHORS:
    short = col.replace("_text.T1", "").replace("asr_", "sev_item")
    df[short] = sev_cols[col]

for c in ["sev_internalizing", "sev_externalizing", "sev_thought"]:
    n = df[c].notna().sum()
    print(f"\n{c}: {n} non-null")

# ══════════════════════════════════════════════════════════════════════════
# Step 3: Theoretically-matched correlations
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ITEM-SPECIFIC CORRELATIONS")
print("=" * 70)
print("\nA. Per-item severity → matched child outcome\n")

sig = lambda p: "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else " "))

results_item = []
for col, anchors in ITEM_ANCHORS.items():
    sev = sev_cols[col]
    for outcome_col, outcome_label in OUTCOMES.items():
        y = pd.to_numeric(df[outcome_col], errors="coerce").values
        mask = ~np.isnan(sev) & ~np.isnan(y)
        n = int(mask.sum())
        if n < 8:
            continue
        r, p = stats.pearsonr(sev[mask], y[mask])
        results_item.append({
            "item": col.replace("_text.T1", ""),
            "label": anchors["label"],
            "outcome": outcome_col,
            "n": n, "r": round(r, 3), "p": round(p, 3),
            "sig": sig(p),
        })

item_df = pd.DataFrame(results_item)
# Show only the a priori theoretically-relevant pairs
theory_pairs = {
    # internalizing items → child internalizing
    "asr_9":   ["T1_YSR_Internalizing", "T2_YSR_Internalizing"],
    "asr_29":  ["T1_YSR_Internalizing", "T2_YSR_Internalizing"],
    "asr_46":  ["T1_YSR_Internalizing", "T2_YSR_Internalizing"],
    "asr_58":  ["T1_YSR_Internalizing", "T2_YSR_Internalizing"],
    "asr_85":  ["T1_YSR_Total",         "T2_YSR_Total"],
    "asr_100": ["T1_YSR_Internalizing", "T2_YSR_Internalizing"],
    # externalizing items → child externalizing
    "asr_84":  ["T1_YSR_Externalizing", "T2_YSR_Externalizing"],
    "asr_92":  ["T1_YSR_Externalizing", "T2_YSR_Externalizing"],
    # voice item → total
    "asr_40":  ["T1_YSR_Total",         "T2_YSR_Total"],
}

print(f"  {'Item':<12} {'Label':<32} {'Outcome':<26} {'N':>4} {'r':>7} {'p':>7}")
print("  " + "-" * 92)
for item, outcomes in theory_pairs.items():
    for oc in outcomes:
        row = item_df[(item_df["item"] == item) & (item_df["outcome"] == oc)]
        if row.empty:
            continue
        r = row.iloc[0]
        print(f"  {r['item']:<12} {r['label'][:30]:<32} {r['outcome']:<26} "
              f"{r['n']:>4} {r['r']:>+7.3f} {r['p']:>7.3f}{r['sig']}")

# ══════════════════════════════════════════════════════════════════════════
# Step 4: Syndrome-level composites
# ══════════════════════════════════════════════════════════════════════════
print("\n\nB. Syndrome composites → matched child outcomes\n")

composites = {
    "sev_internalizing": {
        "desc": "Parent internalizing text (9,29,46,58,85,100)",
        "outcomes": ["T1_YSR_Internalizing", "T2_YSR_Internalizing", "T1_YSR_Total"],
    },
    "sev_externalizing": {
        "desc": "Parent externalizing text (84,92)",
        "outcomes": ["T1_YSR_Externalizing", "T2_YSR_Externalizing", "T1_YSR_Total"],
    },
    "sev_thought": {
        "desc": "Parent thought-prob text (40,84,85)",
        "outcomes": ["T1_YSR_Total", "T2_YSR_Total"],
    },
}

results_comp = []
print(f"  {'Composite':<30} {'Outcome':<26} {'N':>4} {'r':>7} {'p':>7}")
print("  " + "-" * 78)
for comp_col, meta in composites.items():
    comp = df[comp_col].values
    for oc in meta["outcomes"]:
        y = pd.to_numeric(df[oc], errors="coerce").values
        mask = ~np.isnan(comp) & ~np.isnan(y)
        n = int(mask.sum())
        if n < 8:
            continue
        r, p = stats.pearsonr(comp[mask], y[mask])
        results_comp.append({
            "composite": comp_col, "desc": meta["desc"],
            "outcome": oc, "n": n, "r": round(r, 3), "p": round(p, 3),
        })
        print(f"  {meta['desc'][:28]:<30} {oc:<26} {n:>4} {r:>+7.3f} {p:>7.3f}{sig(p)}")
    print()

# ══════════════════════════════════════════════════════════════════════════
# Step 5: vs generic severity composite (from 02_analysis.py)
# ══════════════════════════════════════════════════════════════════════════
prev = pd.read_csv(ROOT / "data" / "merged_with_severity.csv")
df["text_severity_generic"] = prev["text_severity"].values

print("\nC. Item-specific internalizing composite vs generic composite → T1 YSR Internalizing\n")
y = pd.to_numeric(df["T1_YSR_Internalizing"], errors="coerce").values
for label, col in [("Generic composite", "text_severity_generic"),
                   ("Item-specific internalizing", "sev_internalizing"),
                   ("Item-specific externalizing", "sev_externalizing")]:
    x = df[col].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    n = int(mask.sum())
    if n < 8:
        print(f"  {label}: insufficient data")
        continue
    r, p = stats.pearsonr(x[mask], y[mask])
    print(f"  {label:<32} N={n}  r={r:+.3f}  p={p:.3f}{sig(p)}")

# ── Save ───────────────────────────────────────────────────────────────────
df.to_csv(DATA_OUT / "merged_item_specific.csv", index=False)
pd.DataFrame(results_item).to_csv(DATA_OUT / "aim3_item_correlations.csv", index=False)
pd.DataFrame(results_comp).to_csv(DATA_OUT / "aim3_composite_correlations.csv", index=False)
print("\nSaved: data/merged_item_specific.csv")
print("       data/aim3_item_correlations.csv")
print("       data/aim3_composite_correlations.csv")
print("\nDone.")
