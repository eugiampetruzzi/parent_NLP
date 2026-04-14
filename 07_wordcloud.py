"""
07_wordcloud.py
Word clouds from parent free-text ASR responses.

Panels:
  A/B  – Fears (asr_29): high-YSR parents vs low-YSR parents
  C/D  – Fears: social/internal language vs external/object phobias
  E    – Sleep (asr_100): all descriptions
"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from wordcloud import WordCloud, STOPWORDS

ROOT    = Path(__file__).parent
DATA_IN = ROOT / "data" / "merged_clean.csv"
DATA_OUT = ROOT / "data"

MISSING = {"", "nan", "none", "0.0", "1.0", "888", "888.0", "999", "999.0"}
clean = lambda v: "" if str(v).strip().lower() in MISSING else str(v).strip()

df = pd.read_csv(DATA_IN)
for col in ["T1_YSR_Total", "T2_YSR_Total", "T1_YSR_Internalizing"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace({888: np.nan, 999: np.nan})

# ── Custom stopwords ───────────────────────────────────────────────────────
STOPS = set(STOPWORDS) | {
    "i", "m", "ve", "don", "t", "s", "re", "ll", "d",
    "of", "to", "and", "a", "the", "in", "is", "that",
    "it", "for", "on", "with", "as", "at", "by", "be",
    "am", "are", "was", "were", "have", "has", "had",
    "this", "but", "or", "an", "my", "me", "them",
    "not", "also", "just", "like", "can", "get", "when",
    "they", "do", "so", "from", "if", "which", "will",
    "about", "when", "one", "no",
}

def make_wc(text: str, colormap: str = "Blues", bg: str = "white",
            max_words: int = 80) -> WordCloud:
    wc = WordCloud(
        width=700, height=420,
        background_color=bg,
        colormap=colormap,
        stopwords=STOPS,
        max_words=max_words,
        min_font_size=10,
        prefer_horizontal=0.85,
        collocations=True,         # show bigrams
        collocation_threshold=5,
    )
    wc.generate(text)
    return wc

# ══════════════════════════════════════════════════════════════════════════
# FEARS corpus – split by child YSR level
# ══════════════════════════════════════════════════════════════════════════
fears = df[["asr_29_text.T1", "T1_YSR_Total"]].copy()
fears["text"] = fears["asr_29_text.T1"].apply(clean)
fears = fears[fears["text"].str.len() > 0].dropna(subset=["T1_YSR_Total"])
med = fears["T1_YSR_Total"].median()

corpus_hi = " ".join(fears.loc[fears["T1_YSR_Total"] >= med, "text"])
corpus_lo = " ".join(fears.loc[fears["T1_YSR_Total"] <  med, "text"])

# ── Fears: social/internal vs external/object ──────────────────────────────
social_pat  = [r"social", r"anxi", r"crowd", r"public", r"confrontation",
               r"infidelity", r"financ", r"unemploy", r"death", r"harm",
               r"alone", r"worry", r"violence", r"police", r"exploit"]
external_pat = [r"spider", r"snake", r"dog", r"bug", r"rat", r"horse",
                r"balloon", r"height", r"water", r"driv", r"lightning",
                r"fire", r"tight", r"new place", r"getting lost"]

fears_all = df[["asr_29_text.T1"]].copy()
fears_all["text"] = fears_all["asr_29_text.T1"].apply(clean)
fears_all = fears_all[fears_all["text"].str.len() > 0]
fears_all["is_social"]   = fears_all["text"].str.lower().apply(
    lambda t: any(re.search(p, t) for p in social_pat))
fears_all["is_external"] = fears_all["text"].str.lower().apply(
    lambda t: any(re.search(p, t) for p in external_pat))

corpus_social   = " ".join(fears_all.loc[fears_all["is_social"],   "text"])
corpus_external = " ".join(fears_all.loc[fears_all["is_external"] & ~fears_all["is_social"], "text"])

# ── Sleep corpus ──────────────────────────────────────────────────────────
sleep = df[["asr_100_text.T1"]].copy()
sleep["text"] = sleep["asr_100_text.T1"].apply(clean)
sleep = sleep[sleep["text"].str.len() > 0]
corpus_sleep = " ".join(sleep["text"])

# ══════════════════════════════════════════════════════════════════════════
# BUILD FIGURE  (3 rows × 2 cols; sleep spans full bottom row)
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(13, 11))
gs  = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.08)

ax_hi  = fig.add_subplot(gs[0, 0])
ax_lo  = fig.add_subplot(gs[0, 1])
ax_soc = fig.add_subplot(gs[1, 0])
ax_ext = fig.add_subplot(gs[1, 1])
ax_sl  = fig.add_subplot(gs[2, :])

def show(ax, wc, title, subtitle=""):
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
    if subtitle:
        ax.text(0.5, -0.03, subtitle, ha="center", va="top",
                transform=ax.transAxes, fontsize=9, color="#555555",
                style="italic")

n_hi  = (fears["T1_YSR_Total"] >= med).sum()
n_lo  = (fears["T1_YSR_Total"] <  med).sum()
n_soc = fears_all["is_social"].sum()
n_ext = (fears_all["is_external"] & ~fears_all["is_social"]).sum()
n_sl  = len(sleep)

show(ax_hi,  make_wc(corpus_hi,  "Reds"),
     f"Fears — parents of higher-problem children",
     f"T1 YSR Total ≥ median  (n = {n_hi})")

show(ax_lo,  make_wc(corpus_lo,  "Blues"),
     f"Fears — parents of lower-problem children",
     f"T1 YSR Total < median  (n = {n_lo})")

show(ax_soc, make_wc(corpus_social,  "Purples"),
     "Social / internal fears",
     f"Predict child outcomes at T2+ (n = {n_soc})")

show(ax_ext, make_wc(corpus_external, "Greens"),
     "External / object phobias",
     f"Do not predict child outcomes (n = {n_ext})")

show(ax_sl,  make_wc(corpus_sleep, "copper_r", max_words=100),
     f"Parent sleep trouble descriptions  (n = {n_sl})",
     "asr_100: 'I have trouble sleeping (describe)'")

fig.suptitle("Parent ASR Free-Text Responses: Word Clouds",
             fontsize=14, fontweight="bold", y=1.01)

out = DATA_OUT / "wordclouds.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════
# Also save a standalone fears-only figure (cleaner for slides/email)
# ══════════════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 2, figsize=(12, 4.5))
show(axes[0], make_wc(corpus_hi,  "Reds"),
     "Higher-problem children (YSR ≥ median)",
     f"n = {n_hi} parents")
show(axes[1], make_wc(corpus_lo,  "Blues"),
     "Lower-problem children (YSR < median)",
     f"n = {n_lo} parents")
fig2.suptitle("How parents describe their fears/phobias\n(ASR item 29 free-text)",
              fontsize=13, fontweight="bold")
plt.tight_layout()
out2 = DATA_OUT / "wordcloud_fears_hilo.png"
plt.savefig(out2, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

fig3, axes = plt.subplots(1, 2, figsize=(12, 4.5))
show(axes[0], make_wc(corpus_social,   "Purples"),
     "Social / internal fears",
     f"worry, harm, financial, social anxiety … (n = {n_soc})")
show(axes[1], make_wc(corpus_external, "Greens"),
     "External / object phobias",
     f"spiders, heights, dogs, driving … (n = {n_ext})")
fig3.suptitle("Two types of fear language in parent free-text\n"
              "Social fears predict child outcomes; external phobias do not",
              fontsize=13, fontweight="bold")
plt.tight_layout()
out3 = DATA_OUT / "wordcloud_fears_type.png"
plt.savefig(out3, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")
