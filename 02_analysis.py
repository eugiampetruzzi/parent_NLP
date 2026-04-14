"""
02_analysis.py
Aim 1 – Validation:  text severity (embedding-based) matches numeric item scores
Aim 2 – Prediction:  text severity score → child YSR
Aim 3 – Comparison:  text severity vs. ASR Total Problems score

Approach:
  For each text item, compute cosine similarity of the response embedding to
  a "high severity" anchor minus a "low severity" anchor. This gives a
  continuous severity signal per item. Average across items with text per
  participant → participant-level text severity score.

Run after 01_data_prep.py.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sentence_transformers import SentenceTransformer

ROOT     = Path(__file__).parent
DATA_IN  = ROOT / "data" / "merged_clean.csv"
DATA_OUT = ROOT / "data"

TEXT_ITEMS = [
    "asr_9_text.T1", "asr_29_text.T1", "asr_40_text.T1", "asr_46_text.T1",
    "asr_58_text.T1", "asr_84_text.T1", "asr_85_text.T1",
    "asr_92_text.T1", "asr_100_text.T1",
]
ITEM_SCORES = [c.replace("_text", "") for c in TEXT_ITEMS]
ITEM_LABELS = {
    "asr_9.T1":   "Can't get mind off thoughts",
    "asr_29.T1":  "Afraid of animals/situations/places",
    "asr_40.T1":  "Hears sounds/voices others don't",
    "asr_46.T1":  "Body twitches / nervous movements",
    "asr_58.T1":  "Picks skin / body parts",
    "asr_84.T1":  "Does things others think strange",
    "asr_85.T1":  "Has thoughts others think strange",
    "asr_92.T1":  "Things that may cause trouble w/ law",
    "asr_100.T1": "Trouble sleeping",
}

OUTCOMES = {
    "T1_YSR_Internalizing": "T1 Internalizing (concurrent)",
    "T1_YSR_Externalizing": "T1 Externalizing (concurrent)",
    "T1_YSR_Total":         "T1 Total (concurrent)",
    "T2_YSR_Internalizing": "T2 Internalizing (prospective ~1yr)",
    "T2_YSR_Externalizing": "T2 Externalizing (prospective ~1yr)",
    "T2_YSR_Total":         "T2 Total (prospective ~1yr)",
}

# ── load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_IN)

print("Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Severity anchors
anchor_lo = model.encode("not at all, never, no problem, none, normal")
anchor_hi = model.encode("very much, often, serious, severe, significant problem")

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

MISSING = {"", "nan", "none", "888", "888.0", "999", "999.0"}

def sev_score(text):
    """Single text → severity signal (hi_sim - lo_sim). Returns NaN for empty/missing."""
    s = str(text).strip().lower()
    if s in MISSING:
        return np.nan
    e = model.encode(str(text))
    return cosine(e, anchor_hi) - cosine(e, anchor_lo)

# ══════════════════════════════════════════════════════════════════════════
# AIM 1 — VALIDATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("AIM 1 — VALIDATION: text severity vs. numeric item score")
print("=" * 65)
print("r(text severity, item 0/1/2 score) for each item.\n")

aim1_rows = []
item_sev = {}  # store per-item severity arrays for later

for text_col, score_col in zip(TEXT_ITEMS, ITEM_SCORES):
    sev = df[text_col].apply(sev_score).values
    scores = pd.to_numeric(df[score_col], errors="coerce").replace({888: np.nan, 999: np.nan}).values
    item_sev[text_col] = sev

    mask = ~np.isnan(sev) & ~np.isnan(scores)
    n = int(mask.sum())
    if n < 8:
        continue

    r, p = stats.pearsonr(sev[mask], scores[mask])
    aim1_rows.append({
        "item":     score_col.replace(".T1", ""),
        "label":    ITEM_LABELS[score_col],
        "n":        n,
        "r":        round(r, 3),
        "p":        round(p, 3),
        "sig":      "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "")),
    })

aim1 = pd.DataFrame(aim1_rows)
print(aim1[["item", "n", "r", "p", "sig", "label"]].to_string(index=False))
aim1.to_csv(DATA_OUT / "aim1_validation.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════
# COMPUTE PARTICIPANT-LEVEL TEXT SEVERITY COMPOSITE
# ══════════════════════════════════════════════════════════════════════════
# Mean severity across items with text (NaN if no text at all)
sev_matrix = np.column_stack([item_sev[c] for c in TEXT_ITEMS])  # (224, 9)
df["text_severity"] = np.nanmean(sev_matrix, axis=1)
df.loc[~df["has_text"], "text_severity"] = np.nan  # only for text-havers

n_with_sev = df["text_severity"].notna().sum()
print(f"\nParticipant-level text severity score: {n_with_sev} / {len(df)} have score")
print(f"  M = {df['text_severity'].mean():.4f}  SD = {df['text_severity'].std():.4f}")

# Save enriched dataset
df.to_csv(DATA_OUT / "merged_with_severity.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════
# AIM 2/3 — PREDICTION + COMPARISON
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("AIM 2/3 — PREDICTION + COMPARISON")
print("=" * 65)
print("Predictor A: Parent ASR Total Problems score   (N ≈ 128)")
print("Predictor B: Text severity composite           (N ≈ 112)")
print("Comparison: Fisher z-test on r_A vs r_B\n")

asr_total    = pd.to_numeric(df["T1_ASR_Total_Problems_Total_Score"], errors="coerce").values
text_sev     = df["text_severity"].values

aim23_rows = []

for outcome_col, outcome_label in OUTCOMES.items():
    y = pd.to_numeric(df[outcome_col], errors="coerce").values

    # Model A: ASR Total → child YSR
    mask_a = ~np.isnan(y) & ~np.isnan(asr_total)
    na = int(mask_a.sum())
    if na > 5:
        ra, pa = stats.pearsonr(asr_total[mask_a], y[mask_a])
    else:
        ra, pa = np.nan, np.nan

    # Model B: text severity → child YSR
    mask_b = ~np.isnan(y) & ~np.isnan(text_sev)
    nb = int(mask_b.sum())
    if nb > 5:
        rb, pb = stats.pearsonr(text_sev[mask_b], y[mask_b])
    else:
        rb, pb = np.nan, np.nan

    # Fisher z-test: is text r significantly different from scores r?
    # Use participants with both predictors for overlap-corrected test
    mask_both = mask_a & mask_b
    n_both = int(mask_both.sum())
    if n_both > 10 and not (np.isnan(ra) or np.isnan(rb)):
        # Meng et al. (1992) test for comparing dependent correlations
        rxy1 = ra  # r(asr_total, y) in overlap sample
        rxy2 = rb  # r(text_sev, y) in overlap sample
        # r between the two predictors
        rx1x2, _ = stats.pearsonr(asr_total[mask_both], text_sev[mask_both])
        # Steiger's test (simplified)
        rbar = (rxy1 + rxy2) / 2
        f  = (1 - rx1x2) / (2 * (1 - rbar**2))
        t_diff = (rxy1 - rxy2) * np.sqrt((n_both - 1) * (1 + rx1x2)) / \
                 np.sqrt(2 * (1 - rxy1**2) * (1 - rxy2**2) + 2 * rx1x2 * (1 - rxy1**2 - rxy2**2 - rx1x2**2))
        p_diff = 2 * stats.t.sf(abs(t_diff), df=n_both - 3)
        r_x1x2 = round(rx1x2, 3)
    else:
        t_diff, p_diff, r_x1x2 = np.nan, np.nan, np.nan

    aim23_rows.append({
        "outcome":   outcome_col,
        "label":     outcome_label,
        "n_scores":  na,
        "r_scores":  round(ra, 3),  "p_scores": round(pa, 3),
        "n_text":    nb,
        "r_text":    round(rb, 3),  "p_text":   round(pb, 3),
        "n_both":    n_both,
        "r_predictors": r_x1x2,
        "t_diff":    round(t_diff, 3) if not np.isnan(t_diff) else np.nan,
        "p_diff":    round(p_diff, 3) if not np.isnan(p_diff) else np.nan,
    })

aim23 = pd.DataFrame(aim23_rows)
aim23.to_csv(DATA_OUT / "aim23_prediction.csv", index=False)

# Pretty print
sig = lambda p: "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "  "))
print(f"{'Outcome':<34} {'Predictor':<12} {'N':>4}  {'r':>6}  {'p':>6}    diff-p")
print("-" * 75)
for _, row in aim23.iterrows():
    lbl = row["label"][:32]
    ra = row["r_scores"]; pa = row["p_scores"]; na = int(row["n_scores"])
    rb = row["r_text"];   pb = row["p_text"];   nb = int(row["n_text"])
    pd_ = row["p_diff"]
    print(f"  {lbl:<32} {'ASR Total':<12} {na:>4}  {ra:>+6.3f}  {pa:>6.3f}{sig(pa)}")
    print(f"  {'':32} {'Text sev.':<12} {nb:>4}  {rb:>+6.3f}  {pb:>6.3f}{sig(pb)}   {pd_:.3f}" if not np.isnan(pd_) else
          f"  {'':32} {'Text sev.':<12} {nb:>4}  {rb:>+6.3f}  {pb:>6.3f}{sig(pb)}")
    print()

print("\nOutputs:")
print("  data/aim1_validation.csv")
print("  data/aim23_prediction.csv")
print("  data/merged_with_severity.csv")
print("\nDone.")
