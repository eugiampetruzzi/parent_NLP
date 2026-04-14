"""
04_llm_rater.py
Use Claude (Haiku 4.5 via Batch API for 50% cost reduction) to extract
clinically meaningful features from each text response:

  severity     : 0/1/2 — matches the item's own numeric scale
  specificity  : 1/2/3 — how detailed/descriptive the response is
  distress     : 0/1 — does the text indicate personal distress?
  child_ref    : 0/1 — does the text mention children, parenting, or family?

Outputs:
  data/llm_ratings.csv   — one row per (ELS_ID, item)
  data/llm_summary.csv   — wide format, one row per participant

Usage:
  Set ANTHROPIC_API_KEY in your environment, then:
    python3 04_llm_rater.py           # run ratings + correlations
    python3 04_llm_rater.py --analyze # skip API calls, run analysis on existing ratings

Cost estimate: ~400 texts × ~150 input tokens + ~20 output tokens ≈ ~$0.03 with Haiku 4.5
"""

import os, sys, json, time, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT     = Path(__file__).parent
DATA_IN  = ROOT / "data" / "merged_clean.csv"
DATA_OUT = ROOT / "data"

RATINGS_FILE = DATA_OUT / "llm_ratings.csv"

TEXT_ITEMS = {
    "asr_9_text.T1":   "I can't get my mind off certain thoughts (describe the thoughts):",
    "asr_29_text.T1":  "I am afraid of certain animals, situations, or places (describe):",
    "asr_40_text.T1":  "I hear sounds or voices that other people think aren't there (describe):",
    "asr_46_text.T1":  "Parts of my body twitch or make nervous movements (describe):",
    "asr_58_text.T1":  "I pick my skin or other parts of my body (describe):",
    "asr_84_text.T1":  "I do things that other people think are strange (describe):",
    "asr_85_text.T1":  "I have thoughts that other people would think are strange (describe):",
    "asr_92_text.T1":  "I do things that may cause me trouble with the law (describe):",
    "asr_100_text.T1": "I have trouble sleeping (describe):",
}

MISSING = {"", "nan", "none", "888", "888.0", "999", "999.0"}

SYSTEM_PROMPT = """You are a clinical psychologist rating open-ended descriptions written by adults on a self-report questionnaire (Adult Self-Report / ASR). For each item + response, rate exactly four dimensions as integers.

Return ONLY valid JSON with these four keys:
  severity    : 0 = not a problem, 1 = somewhat of a problem, 2 = significant problem
  specificity : 1 = vague or minimal description, 2 = moderately specific, 3 = detailed/specific
  distress    : 0 = no personal distress indicated, 1 = distress, worry, or suffering mentioned
  child_ref   : 0 = no mention of children/parenting/family, 1 = mentions children, family, or parenting

No explanation, no markdown — only the JSON object."""


def parse_llm_json(text: str) -> dict | None:
    """Extract JSON from LLM response, handling minor formatting issues."""
    text = text.strip()
    # strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
        # validate keys and types
        for k in ("severity", "specificity", "distress", "child_ref"):
            if k not in obj or not isinstance(obj[k], (int, float)):
                return None
        return {k: int(obj[k]) for k in ("severity", "specificity", "distress", "child_ref")}
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════
# RATING VIA BATCH API
# ══════════════════════════════════════════════════════════════════════════

def collect_texts(df: pd.DataFrame) -> list[dict]:
    """Collect all non-empty text responses as (els_id, item_col, item_question, text)."""
    rows = []
    for col, question in TEXT_ITEMS.items():
        for _, row in df.iterrows():
            val = str(row[col]).strip().lower()
            if val in MISSING:
                continue
            rows.append({
                "els_id":   row["ELS_ID"],
                "item_col": col,
                "question": question,
                "text":     str(row[col]).strip(),
            })
    print(f"Total texts to rate: {len(rows)}")
    return rows


def run_batch(rows: list[dict], client) -> dict:
    """
    Submit via Batch API, poll until done, return {custom_id: rating_dict}.
    Uses prompt caching on the shared system prompt.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    requests = []
    for i, r in enumerate(rows):
        user_msg = f'Item: "{r["question"]}"\nResponse: "{r["text"]}"'
        requests.append(Request(
            custom_id=f"row-{i}",
            params=MessageCreateParamsNonStreaming(
                model="claude-haiku-4-5",
                max_tokens=64,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},  # cache shared system prompt
                }],
                messages=[{"role": "user", "content": user_msg}],
            )
        ))

    print(f"Submitting batch of {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch ID: {batch.id}  Status: {batch.processing_status}")

    # Poll
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        counts = batch.request_counts
        print(f"  Processing... {counts.processing} remaining / "
              f"{counts.succeeded} done / {counts.errored} errors")
        time.sleep(15)

    print(f"Batch complete. succeeded={batch.request_counts.succeeded}  "
          f"errored={batch.request_counts.errored}")

    # Collect results
    results = {}
    for result in client.messages.batches.results(batch.id):
        if result.result.type == "succeeded":
            text = next(
                (b.text for b in result.result.message.content if b.type == "text"), ""
            )
            parsed = parse_llm_json(text)
            if parsed:
                results[result.custom_id] = parsed
            else:
                print(f"  Parse failed for {result.custom_id}: {repr(text[:80])}")
        else:
            print(f"  Failed: {result.custom_id}")

    return results


def build_ratings_df(rows: list[dict], results: dict) -> pd.DataFrame:
    records = []
    for i, r in enumerate(rows):
        rating = results.get(f"row-{i}")
        if rating is None:
            continue
        records.append({
            "ELS_ID":   r["els_id"],
            "item":     r["item_col"].replace("_text.T1", ""),
            "item_col": r["item_col"],
            "text":     r["text"],
            **rating,
        })
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def run_analysis(ratings_df: pd.DataFrame, df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("LLM RATINGS — ANALYSIS")
    print("=" * 65)

    sig = lambda p: "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else " "))

    # ── Aim 1 validation: LLM severity vs numeric item score ──────────────
    print("\nAim 1 — LLM severity vs numeric item score:")
    print(f"  {'Item':<12} {'N':>4}  {'r(LLM,score)':>14}  {'p':>6}")
    print("  " + "-" * 45)
    for item_col in TEXT_ITEMS:
        item = item_col.replace("_text.T1", "")
        score_col = item + ".T1"
        sub = ratings_df[ratings_df["item_col"] == item_col].copy()
        if len(sub) < 8 or score_col not in df.columns:
            continue
        sub = sub.merge(
            df[["ELS_ID", score_col]].dropna(),
            on="ELS_ID", how="inner"
        )
        sub[score_col] = pd.to_numeric(sub[score_col], errors="coerce")
        sub = sub.dropna(subset=[score_col])
        sub = sub[~sub[score_col].isin([888, 999])]
        if len(sub) < 5:
            continue
        r, p = stats.pearsonr(sub["severity"], sub[score_col])
        print(f"  {item:<12} {len(sub):>4}  r={r:+.3f}  p={p:.3f}{sig(p)}")

    # ── Build participant-level features ──────────────────────────────────
    # Mean severity, mean specificity, any distress, any child_ref
    wide = ratings_df.groupby("ELS_ID").agg(
        llm_severity_mean   = ("severity",    "mean"),
        llm_specificity_mean= ("specificity", "mean"),
        llm_distress_any    = ("distress",    "max"),
        llm_child_ref_any   = ("child_ref",   "max"),
        n_items_rated       = ("item",        "nunique"),
    ).reset_index()

    df2 = df.merge(wide, on="ELS_ID", how="left")

    # ── Aim 2/3: LLM features → child YSR ────────────────────────────────
    OUTCOMES = {
        "T1_YSR_Internalizing": "T1 Internalizing (concurrent)",
        "T1_YSR_Externalizing": "T1 Externalizing (concurrent)",
        "T1_YSR_Total":         "T1 Total (concurrent)",
        "T2_YSR_Internalizing": "T2 Internalizing (prospective)",
        "T2_YSR_Total":         "T2 Total (prospective)",
    }

    print("\nAim 2/3 — LLM severity composite → child YSR:")
    print(f"  {'Feature':<24} {'Outcome':<26} {'N':>4}  {'r':>7}  {'p':>6}")
    print("  " + "-" * 72)

    features = ["llm_severity_mean", "llm_specificity_mean", "llm_distress_any", "llm_child_ref_any"]

    results = []
    for feat in features:
        for oc, oc_label in OUTCOMES.items():
            y = pd.to_numeric(df2[oc], errors="coerce").values
            x = df2[feat].values
            mask = ~np.isnan(x.astype(float)) & ~np.isnan(y)
            n = int(mask.sum())
            if n < 8:
                continue
            r, p = stats.pearsonr(x[mask].astype(float), y[mask])
            results.append({"feature": feat, "outcome": oc, "n": n,
                             "r": round(r, 3), "p": round(p, 3)})
            print(f"  {feat:<24} {oc_label[:24]:<26} {n:>4}  {r:>+7.3f}  {p:>6.3f}{sig(p)}")
        print()

    # Compare LLM severity vs ASR Total
    print("Comparison: LLM severity mean vs ASR Total → T1 YSR Internalizing")
    y = pd.to_numeric(df2["T1_YSR_Internalizing"], errors="coerce").values
    for label, col in [("ASR Total score", "T1_ASR_Total_Problems_Total_Score"),
                       ("LLM severity mean", "llm_severity_mean")]:
        x = pd.to_numeric(df2[col], errors="coerce").values
        mask = ~np.isnan(x) & ~np.isnan(y)
        n = int(mask.sum())
        if n < 5:
            print(f"  {label}: insufficient data")
            continue
        r, p = stats.pearsonr(x[mask], y[mask])
        print(f"  {label:<24}  N={n}  r={r:+.3f}  p={p:.3f}{sig(p)}")

    # Save wide ratings
    wide.to_csv(DATA_OUT / "llm_summary.csv", index=False)
    pd.DataFrame(results).to_csv(DATA_OUT / "llm_aim23.csv", index=False)
    print("\nSaved: data/llm_summary.csv, data/llm_aim23.csv")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true",
                        help="Skip API calls; run analysis on existing ratings")
    args = parser.parse_args()

    df = pd.read_csv(DATA_IN)

    if args.analyze or RATINGS_FILE.exists():
        if RATINGS_FILE.exists():
            print(f"Loading existing ratings from {RATINGS_FILE}")
            ratings_df = pd.read_csv(RATINGS_FILE)
            print(f"  {len(ratings_df)} rated responses")
        else:
            print("No ratings file found. Run without --analyze first.")
            sys.exit(1)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("\nANTHROPIC_API_KEY not set.")
            print("Set it with:  export ANTHROPIC_API_KEY=sk-ant-...")
            print("Or in Claude Code:  ! export ANTHROPIC_API_KEY=sk-ant-...")
            sys.exit(1)

        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        rows = collect_texts(df)
        results = run_batch(rows, client)

        ratings_df = build_ratings_df(rows, results)
        ratings_df.to_csv(RATINGS_FILE, index=False)
        print(f"\nSaved {len(ratings_df)} ratings → {RATINGS_FILE}")

    run_analysis(ratings_df, df)
    print("\nDone.")


if __name__ == "__main__":
    main()
