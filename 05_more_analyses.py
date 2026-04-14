"""
05_more_analyses.py
Additional analyses:
  A. All timepoints for best-performing items (fears, sleep)
  B. Simple text features: word count, keyword categories
  C. Sleep subtype deep-dive (onset vs maintenance vs cause)
  D. Fears content (internal/social vs external/phobia)
  E. Worry/stress language across items → child internalizing
  F. Child/family references in parent text → child outcomes
  G. Number of items endorsed (text count) → child outcomes
  H. Moderation: numeric score × text severity → child outcome
"""

import warnings; warnings.filterwarnings("ignore")
import re, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from sentence_transformers import SentenceTransformer

ROOT     = Path(__file__).parent
DATA_IN  = ROOT / "data" / "merged_clean.csv"
DATA_OUT = ROOT / "data"

MISSING = {"", "nan", "none", "0.0", "1.0", "888", "888.0", "999", "999.0"}

ALL_OUTCOMES = {
    "T1_YSR_Internalizing": "T1 Int",
    "T1_YSR_Externalizing": "T1 Ext",
    "T1_YSR_Total":         "T1 Tot",
    "T2_YSR_Internalizing": "T2 Int",
    "T2_YSR_Externalizing": "T2 Ext",
    "T2_YSR_Total":         "T2 Tot",
    "T3_YSR_Internalizing": "T3 Int",
    "T3_YSR_Externalizing": "T3 Ext",
    "T3_YSR_Total":         "T3 Tot",
    "T4_YSR_Internalizing": "T4 Int",
    "T4_YSR_Total":         "T4 Tot",
    "TA_ASR_Internalizing": "TA Int",
    "TA_ASR_Total":         "TA Tot",
}

df = pd.read_csv(DATA_IN)

# Clean numeric outcomes
for col in ALL_OUTCOMES:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace({888: np.nan, 999: np.nan})

def clean_text(v):
    s = str(v).strip()
    return "" if s.lower() in MISSING else s

def sig(p):
    return "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else ("†" if p < .10 else " ")))

def corr_row(x, y, label_x, label_y):
    mask = ~np.isnan(x.astype(float)) & ~np.isnan(y)
    n = mask.sum()
    if n < 8:
        return None
    r, p = stats.pearsonr(x[mask].astype(float), y[mask])
    return {"predictor": label_x, "outcome": label_y, "n": n,
            "r": round(r, 3), "p": round(p, 3), "sig": sig(p)}

def print_corr(rows, header=""):
    if header:
        print(f"\n{header}")
    print(f"  {'Predictor':<30} {'Outcome':<8} {'N':>4}  {'r':>7}  {'p':>6}")
    print("  " + "-" * 58)
    for r in rows:
        if r:
            print(f"  {r['predictor']:<30} {r['outcome']:<8} {r['n']:>4}  {r['r']:>+7.3f}  {r['p']:>6.3f}{r['sig']}")

results_all = []

# ══════════════════════════════════════════════════════════════════════════
# A. ALL TIMEPOINTS for fears (best item) and sleep (most data)
# ══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("A. FEARS & SLEEP TEXT → ALL TIMEPOINTS")
print("=" * 65)

print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# Fear-specific anchors
fear_lo = model.encode("mild discomfort, slight preference to avoid, not distressing")
fear_hi = model.encode("severe phobia, intense panic, avoidance disrupts daily life, debilitating")

# Sleep-specific anchors
sleep_lo = model.encode("occasional mild difficulty, resting fine, sleep well most nights")
sleep_hi = model.encode("severe chronic insomnia, exhausted, can't sleep, major impairment")

for item_col, a_lo, a_hi, label in [
    ("asr_29_text.T1", fear_lo,  fear_hi,  "Fears text sev"),
    ("asr_100_text.T1", sleep_lo, sleep_hi, "Sleep text sev"),
]:
    texts = df[item_col].apply(clean_text).values
    sev   = np.array([
        cosine(model.encode(t), a_hi) - cosine(model.encode(t), a_lo)
        if t else np.nan for t in texts
    ])
    rows = []
    for oc, oc_short in ALL_OUTCOMES.items():
        if oc not in df.columns:
            continue
        r = corr_row(sev, df[oc].values, label, oc_short)
        if r:
            rows.append(r)
            results_all.append(r)
    print_corr(rows, f"  {label}")

# ══════════════════════════════════════════════════════════════════════════
# B. WORD COUNT as severity proxy
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("B. WORD COUNT → CHILD OUTCOMES")
print("=" * 65)
print("(More words = more elaboration = potentially more distress)")

for item_col, label in [
    ("asr_29_text.T1",  "Fears word count"),
    ("asr_100_text.T1", "Sleep word count"),
    ("asr_9_text.T1",   "Obsessions word count"),
    ("asr_85_text.T1",  "Strange thoughts wc"),
]:
    wc = df[item_col].apply(
        lambda v: len(clean_text(v).split()) if clean_text(v) else np.nan
    ).values
    rows = []
    for oc in ["T1_YSR_Internalizing", "T1_YSR_Total", "T2_YSR_Internalizing", "T2_YSR_Total"]:
        if oc not in df.columns: continue
        r = corr_row(wc, df[oc].values, label, oc.replace("_YSR_", " ").replace(".1",""))
        if r: rows.append(r); results_all.append(r)
    if any(rows):
        print_corr(rows, f"  {label}")

# All-item total word count
total_wc = np.zeros(len(df))
for col in [c for c in df.columns if "_text.T1" in c]:
    wc = df[col].apply(lambda v: len(clean_text(v).split()) if clean_text(v) else 0).values
    total_wc += wc
total_wc = np.where(total_wc == 0, np.nan, total_wc)

rows = []
for oc in ["T1_YSR_Internalizing","T1_YSR_Total","T2_YSR_Internalizing","T2_YSR_Total"]:
    if oc not in df.columns: continue
    r = corr_row(total_wc, df[oc].values, "Total word count (all items)", oc.replace("_YSR_"," "))
    if r: rows.append(r); results_all.append(r)
print_corr(rows, "  Total word count across all items")

# ══════════════════════════════════════════════════════════════════════════
# C. SLEEP SUBTYPES via keywords
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("C. SLEEP SUBTYPES → CHILD OUTCOMES")
print("=" * 65)

sleep_texts = df["asr_100_text.T1"].apply(clean_text).str.lower()

def kw_flag(series, patterns):
    return series.apply(
        lambda t: 1.0 if t and any(re.search(p, t) for p in patterns) else
                  (np.nan if not t else 0.0)
    ).values

subtypes = {
    "Sleep onset (can't fall asleep)": [
        r"can'?t fall", r"fall(ing)? asleep", r"fall to sleep", r"go to sleep",
        r"hard to sleep", r"difficult.*sleep", r"sleep.*difficult"
    ],
    "Sleep maintenance (wakes up)":    [
        r"wake up", r"wak(e|ing)", r"stay asleep", r"stay(ing)? asleep",
        r"through.*night", r"middle.*night", r"back asleep"
    ],
    "Worry/stress-driven sleep":       [
        r"worry", r"worr", r"stress", r"anxi", r"think(ing)?", r"mind", r"thought"
    ],
    "Physical/medical cause":          [
        r"pain", r"apnea", r"menopause", r"hot flash", r"pregnant", r"cpap",
        r"ache", r"physical"
    ],
    "Child/family-related sleep":      [
        r"kid", r"child", r"son", r"daughter", r"baby", r"school"
    ],
}

for subtype, patterns in subtypes.items():
    flags = kw_flag(sleep_texts, patterns)
    n_yes = int(np.nansum(flags))
    rows = []
    for oc in ["T1_YSR_Internalizing", "T1_YSR_Total", "T2_YSR_Internalizing", "T2_YSR_Total"]:
        if oc not in df.columns: continue
        r = corr_row(flags, df[oc].values, subtype[:28], oc.replace("_YSR_"," "))
        if r:
            r["n_yes"] = n_yes
            rows.append(r); results_all.append(r)
    if rows:
        print(f"\n  {subtype} (N endorse={n_yes}):")
        print(f"  {'Outcome':<8} {'N':>4}  {'r':>7}  {'p':>6}")
        for r in rows:
            print(f"  {r['outcome']:<8} {r['n']:>4}  {r['r']:>+7.3f}  {r['p']:>6.3f}{r['sig']}")

# ══════════════════════════════════════════════════════════════════════════
# D. FEARS CONTENT: internal (social/worry) vs external (object phobia)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("D. FEARS CONTENT: SOCIAL/INTERNAL vs EXTERNAL/PHOBIA")
print("=" * 65)

fear_texts = df["asr_29_text.T1"].apply(clean_text).str.lower()

fear_types = {
    "Social/internal fears": [
        r"crowd", r"social", r"people", r"public", r"judg", r"embarrass",
        r"anxi", r"worry", r"financ", r"job", r"work", r"death", r"los(s|ing)",
        r"alone", r"night", r"dark", r"home alone"
    ],
    "External/object phobias": [
        r"spider", r"snake", r"dog", r"animal", r"insect", r"bug",
        r"height", r"water", r"driv", r"fly", r"flying", r"needle",
        r"place", r"lost", r"getting lost"
    ],
}

for ftype, patterns in fear_types.items():
    flags = kw_flag(fear_texts, patterns)
    n_yes = int(np.nansum(flags))
    rows = []
    for oc in ["T1_YSR_Internalizing", "T1_YSR_Total", "T2_YSR_Internalizing", "T2_YSR_Total"]:
        if oc not in df.columns: continue
        r = corr_row(flags, df[oc].values, ftype[:28], oc.replace("_YSR_"," "))
        if r: rows.append(r); results_all.append(r)
    if rows:
        print(f"\n  {ftype} (N={n_yes}):")
        for r in rows:
            print(f"  {r['outcome']:<14} N={r['n']}  r={r['r']:+.3f}  p={r['p']:.3f}{r['sig']}")

# ══════════════════════════════════════════════════════════════════════════
# E. WORRY/STRESS LANGUAGE across all items
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("E. WORRY/STRESS LANGUAGE ACROSS ALL ITEMS")
print("=" * 65)

worry_patterns = [r"worry", r"worr", r"stress", r"anxi", r"fear", r"scar",
                  r"nervou", r"panic", r"overwhelm"]
all_text_combined = df[[c for c in df.columns if "_text.T1" in c and "drugs" not in c]]\
    .apply(lambda row: " ".join(clean_text(v) for v in row if clean_text(v)), axis=1)\
    .str.lower()

worry_flag = all_text_combined.apply(
    lambda t: 1.0 if t.strip() and any(re.search(p, t) for p in worry_patterns)
              else (np.nan if not t.strip() else 0.0)
).values

n_worry = int(np.nansum(worry_flag))
print(f"\n  Parents with worry/stress language anywhere: {n_worry} / {(~np.isnan(worry_flag)).sum()}")
rows = []
for oc in ["T1_YSR_Internalizing","T1_YSR_Total","T2_YSR_Internalizing","T2_YSR_Total",
           "T3_YSR_Internalizing","T3_YSR_Total"]:
    if oc not in df.columns: continue
    r = corr_row(worry_flag, df[oc].values, "Worry/stress language", oc.replace("_YSR_"," "))
    if r: rows.append(r); results_all.append(r)
print_corr(rows, "  Worry/stress language in any item text → child outcomes")

# ══════════════════════════════════════════════════════════════════════════
# F. CHILD/FAMILY REFERENCES in parent text
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("F. CHILD/FAMILY REFERENCES IN PARENT TEXT")
print("=" * 65)

child_patterns = [r"\bkid", r"\bchild", r"\bson\b", r"\bdaughter", r"\bbaby\b",
                  r"\bchildren\b", r"\bschool\b", r"\bparenting\b"]

child_flag = all_text_combined.apply(
    lambda t: 1.0 if t.strip() and any(re.search(p, t) for p in child_patterns)
              else (np.nan if not t.strip() else 0.0)
).values

n_child = int(np.nansum(child_flag))
print(f"\n  Parents mentioning child/family: {n_child} / {(~np.isnan(child_flag)).sum()}")
print("  Texts with child references:")
for idx in np.where(child_flag == 1)[0]:
    print(f"    [{df.iloc[idx]['ELS_ID']}] {all_text_combined.iloc[idx][:100]}")

rows = []
for oc in ["T1_YSR_Internalizing","T1_YSR_Total","T2_YSR_Internalizing","T2_YSR_Total"]:
    if oc not in df.columns: continue
    r = corr_row(child_flag, df[oc].values, "Mentions child/family", oc.replace("_YSR_"," "))
    if r: rows.append(r); results_all.append(r)
print_corr(rows, "  Child/family mentions → child outcomes")

# ══════════════════════════════════════════════════════════════════════════
# G. NUMBER OF ITEMS WITH TEXT (breadth of endorsement)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("G. NUMBER OF ITEMS WITH TEXT (BREADTH OF ENDORSEMENT)")
print("=" * 65)

text_cols = [c for c in df.columns if "_text.T1" in c and "drugs" not in c]
n_items = df[text_cols].apply(
    lambda row: sum(1 for v in row if clean_text(str(v))), axis=1
).values.astype(float)
n_items[n_items == 0] = np.nan

print(f"\n  Distribution: {pd.Series(n_items).value_counts().sort_index().to_dict()}")

rows = []
for oc in ["T1_YSR_Internalizing","T1_YSR_Total","T2_YSR_Internalizing","T2_YSR_Total",
           "T3_YSR_Internalizing","T3_YSR_Total"]:
    if oc not in df.columns: continue
    r = corr_row(n_items, df[oc].values, "N items with text", oc.replace("_YSR_"," "))
    if r: rows.append(r); results_all.append(r)
print_corr(rows, "  N items with text → child outcomes")

# ══════════════════════════════════════════════════════════════════════════
# H. MODERATION: numeric score × text severity → child outcome
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("H. MODERATION: DOES TEXT SEVERITY ADD OVER NUMERIC SCORE?")
print("=" * 65)
print("  Model 1: item score only → child outcome")
print("  Model 2: item score + text severity → child outcome (among text-havers)")
print("  Δr²: unique variance from adding text\n")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Reload item-specific severity from 03
df3 = pd.read_csv(DATA_OUT / "merged_item_specific.csv")

for item, text_sev_col, score_col, outcome_col in [
    ("Fears",  "sev_item29",  "asr_29.T1",  "T1_YSR_Internalizing"),
    ("Fears",  "sev_item29",  "asr_29.T1",  "T1_YSR_Total"),
    ("Sleep",  "sev_item100", "asr_100.T1", "T1_YSR_Internalizing"),
    ("Sleep",  "sev_item100", "asr_100.T1", "T1_YSR_Total"),
]:
    if text_sev_col not in df3.columns or score_col not in df3.columns:
        continue
    score = pd.to_numeric(df3[score_col], errors="coerce").replace({888:np.nan,999:np.nan}).values
    tsev  = df3[text_sev_col].values
    y     = pd.to_numeric(df3[outcome_col], errors="coerce").values

    # Score-only model (all non-missing)
    mask1 = ~np.isnan(score) & ~np.isnan(y)
    if mask1.sum() < 10: continue
    r1, p1 = stats.pearsonr(score[mask1], y[mask1])

    # Score + text model (text-havers only)
    mask2 = mask1 & ~np.isnan(tsev)
    if mask2.sum() < 10: continue
    # partial r of text over score
    sc = StandardScaler()
    X2 = np.column_stack([score[mask2], tsev[mask2]])
    X2s = sc.fit_transform(X2)
    ym = y[mask2]
    # r² for score alone vs score+text (in the same subsample)
    r1s, _ = stats.pearsonr(score[mask2], ym)
    r2_model = np.corrcoef(X2s @ np.linalg.lstsq(X2s, ym, rcond=None)[0], ym)[0,1]
    dr2 = r2_model**2 - r1s**2
    # partial correlation of text controlling for score
    score_res = ym - score[mask2] * (np.cov(score[mask2], ym)[0,1] / np.var(score[mask2]))
    tsev_res  = tsev[mask2] - score[mask2] * (np.cov(score[mask2], tsev[mask2])[0,1] / np.var(score[mask2]))
    if np.std(tsev_res) < 1e-10:
        continue
    pr, pp = stats.pearsonr(tsev_res, score_res)

    print(f"  {item} → {outcome_col.replace('_YSR_',' ')}:")
    print(f"    Score only (N={mask1.sum()}):          r={r1:+.3f} p={p1:.3f}{sig(p1)}")
    print(f"    Score only [text-havers] (N={mask2.sum()}): r={r1s:+.3f}")
    print(f"    + text severity:                 ΔR²={dr2:+.4f}  partial-r(text|score)={pr:+.3f} p={pp:.3f}{sig(pp)}")
    print()

# ══════════════════════════════════════════════════════════════════════════
# SAVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════
res_df = pd.DataFrame([r for r in results_all if r])
res_df.to_csv(DATA_OUT / "more_analyses_results.csv", index=False)

# Print top findings
print("\n" + "=" * 65)
print("TOP FINDINGS (p < .10, sorted by |r|)")
print("=" * 65)
top = res_df[res_df["p"] < .10].sort_values("r", key=abs, ascending=False)
if len(top):
    print(top[["predictor","outcome","n","r","p","sig"]].to_string(index=False))
else:
    print("  None reached p < .10")

print("\nSaved: data/more_analyses_results.csv")
print("Done.")
