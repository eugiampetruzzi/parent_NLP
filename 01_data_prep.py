"""
01_data_prep.py
Merge parent ASR (raw + summary) with child YSR outcomes.
Outputs: data/merged_clean.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
DATA_OUT = ROOT / "data"
DATA_OUT.mkdir(exist_ok=True)

# ── paths ──────────────────────────────────────────────────────────────────
ASR_RAW  = ROOT / "asr.T1.csv"
ASR_SUM  = Path("/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/1 - Data/ELS/parent psychopathology/parental_ASR_T1.csv")
YSR_FILE = Path("/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/1 - Data/ELS/ysr/YSR_ASR_T1-TA.xlsx")

# ── item metadata ──────────────────────────────────────────────────────────
TEXT_ITEMS = {
    "asr_9_text.T1":   "Can't get mind off certain thoughts",
    "asr_29_text.T1":  "I am afraid of certain animals, situations, or places",
    "asr_40_text.T1":  "I hear sounds or voices others think aren't there",
    "asr_46_text.T1":  "Parts of my body twitch or make nervous movements",
    "asr_58_text.T1":  "I pick my skin or other parts of my body",
    "asr_84_text.T1":  "I do things others think are strange",
    "asr_85_text.T1":  "I have thoughts others would think are strange",
    "asr_92_text.T1":  "I do things that may cause trouble with the law",
    "asr_100_text.T1": "I have trouble sleeping",
}

MISSING_CODES = {"888", "888.0", "999", "999.0", "nan", ""}

# ── load ───────────────────────────────────────────────────────────────────
print("Loading data...")
asr_raw = pd.read_csv(ASR_RAW)
asr_sum = pd.read_csv(ASR_SUM)
ysr     = pd.read_excel(YSR_FILE)

# ── clean text ─────────────────────────────────────────────────────────────
def clean_cell(v):
    s = str(v).strip()
    return "" if s in MISSING_CODES else s

for col in TEXT_ITEMS:
    asr_raw[col] = asr_raw[col].apply(clean_cell)

# Concatenate all text per participant (space-separated, skip blanks)
asr_raw["all_text"] = asr_raw[list(TEXT_ITEMS)].apply(
    lambda row: " | ".join(v for v in row if v), axis=1
)
asr_raw["has_text"] = asr_raw["all_text"].str.len() > 0
asr_raw["n_items_with_text"] = asr_raw[list(TEXT_ITEMS)].apply(
    lambda row: sum(1 for v in row if v), axis=1
)

# ── merge ──────────────────────────────────────────────────────────────────
df = (asr_raw
      .merge(ysr,     on="ELS_ID", how="inner")
      .merge(asr_sum, on="ELS_ID", how="left"))

# ── select useful columns ──────────────────────────────────────────────────
KEEP_NUMERIC = [
    # item-level scores (0/1/2), replacing missing codes with NaN
    "asr_9.T1", "asr_29.T1", "asr_40.T1", "asr_46.T1", "asr_58.T1",
    "asr_84.T1", "asr_85.T1", "asr_92.T1", "asr_100.T1",
    # parent summary
    "T1_ASR_Internalizing_Problems_Total_Score",
    "T1_ASR_Externalizing_Problems_Total_Score",
    "T1_ASR_Total_Problems_Total_Score",
    "T1_ASR_Thought_Problems_Total_Score",
    "T1_ASR_AnxiousDepressed_Total_Score",
    # child YSR
    "T1_YSR_Internalizing", "T1_YSR_Externalizing", "T1_YSR_Total",
    "T2_YSR_Internalizing", "T2_YSR_Externalizing", "T2_YSR_Total",
    "T3_YSR_Internalizing", "T3_YSR_Externalizing", "T3_YSR_Total",
    "T4_YSR_Internalizing", "T4_YSR_Externalizing", "T4_YSR_Total",
    "TA_ASR_Internalizing", "TA_ASR_Externalizing", "TA_ASR_Total",
]

# recode 888/999 in numeric cols to NaN
for col in KEEP_NUMERIC:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace({888: np.nan, 999: np.nan})

ID_COLS    = ["ELS_ID"]
TEXT_COLS  = list(TEXT_ITEMS.keys()) + ["all_text", "has_text", "n_items_with_text"]
final_cols = ID_COLS + TEXT_COLS + [c for c in KEEP_NUMERIC if c in df.columns]

out = df[final_cols].copy()
out.to_csv(DATA_OUT / "merged_clean.csv", index=False)

# ── summary ────────────────────────────────────────────────────────────────
print(f"\nOutput: data/merged_clean.csv  ({out.shape[0]} rows × {out.shape[1]} cols)")
print(f"\nText coverage:")
print(f"  Any text:          {out['has_text'].sum()} / {len(out)}")
for col, label in TEXT_ITEMS.items():
    n = (out[col].str.len() > 0).sum()
    print(f"  {col.split('_text')[0]:12s}: {n:3d}  — {label[:50]}")

print(f"\nChild outcome coverage (N non-null):")
for col in ["T1_YSR_Total", "T2_YSR_Total", "T3_YSR_Total", "T4_YSR_Total", "TA_ASR_Total"]:
    if col in out.columns:
        print(f"  {col}: {out[col].notna().sum()}")

print("\nDone.")
