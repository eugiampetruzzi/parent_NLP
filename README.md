# parent_NLP

NLP analysis of parent free-text responses on the Adult Self-Report (ASR), linked to child outcomes (YSR) across timepoints.

**Core question:** Do the written descriptions parents add to ASR items predict child psychopathology — beyond what the numeric checkbox captures?

---

## Read in this order

### 1. `01_data_prep.py`
Start here. Merges the raw ASR text data, YSR child outcomes, and parent summary scores into a single clean file (`data/merged_clean.csv`). Handles missing codes (888, 999), cleans text columns, creates `has_text` and `n_items_with_text` flags.

### 2. `02_analysis.py`
The core analysis.
- **Aim 1 (Validation):** Does embedding-based text severity correlate with the numeric item score? Computes cosine similarity of each response to hi/lo severity anchors.
- **Aim 2 (Prediction):** Does parent text severity predict child YSR (concurrent + prospective)?
- **Aim 3 (Comparison):** Fisher z-test comparing text severity vs. ASR Total score as predictors.
- Output: `data/aim1_validation.csv`, `data/aim23_prediction.csv`, `data/merged_with_severity.csv`

### 3. `03_item_specific.py`
Refines the severity scoring with **item-specific anchors** (e.g., fears get fear-specific hi/lo phrases rather than generic "severe"/"none"). Groups items into internalizing / externalizing / thought-problem syndromes and tests theoretically matched correlations with child outcomes.
- Output: `data/aim3_item_correlations.csv`, `data/aim3_composite_correlations.csv`

### 4. `05_more_analyses.py`
Extended analyses across all timepoints (T1–TA). Key findings:
- Fears text severity → child YSR Total grows from r=.35 (T1) to r=.48 (T4)
- Social/internal fears (worry, confrontation, unemployment) drive the effect; external phobias (spiders, heights) do not
- Sleep maintenance descriptions (waking up at night) trend toward higher child internalizing
- Word count of fears text predicts T2 outcomes (r≈.35)
- Output: `data/more_analyses_results.csv`

### 5. `06_figure_and_quotes.py`
Generates the main figure (fears text severity → child YSR Total across timepoints T1–TA) and prints curated example quotes organized by group.
- Output: `data/figure_fears_timepoints.png`

### 6. `07_wordcloud.py` + `08_example_sentences.py`
Visualization scripts. Word clouds and quote-card figures contrasting:
- Fears text: parents of higher- vs. lower-problem children
- Fears text: social/internal fears vs. external phobias
- Sleep text: maintenance (waking up) vs. other descriptions
- Output: `data/wordcloud_*.png`, `data/quotes_*.png`

---

## Optional / not yet run

### `04_llm_rater.py`
Uses Claude (Haiku via Batch API) to re-rate each text on the original 0/1/2 scale, plus specificity, distress, and child-reference flags. Requires `ANTHROPIC_API_KEY`. Run with:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 04_llm_rater.py
```
Re-run analysis only (once ratings exist):
```bash
python3 04_llm_rater.py --analyze
```

### `09_setup_annotation.py`
Generates the human annotation materials in `human_annotation/`. Run once to regenerate if needed.

---

## Human annotation

`human_annotation/` contains materials for validating LLM ratings against trained human raters:
- **`annotation_guide.xlsx`** — codebook with rubric, calibration examples, and instructions
- **`texts_to_rate.xlsx`** — 49 sampled texts (6–8 per item, stratified by numeric score) with input cells for severity, specificity, distress, and confidence ratings; true numeric score hidden

---

## Data

| File | Description |
|------|-------------|
| `data/asr.T1.csv` | Raw ASR T1 data (text + numeric items) |
| `data/merged_clean.csv` | Main analysis file (output of `01_data_prep.py`) |
| `data/merged_with_severity.csv` | + generic text severity composite (from `02`) |
| `data/merged_item_specific.csv` | + item-specific severity scores (from `03`) |
| `data/aim1_validation.csv` | r(text severity, numeric score) per item |
| `data/aim23_prediction.csv` | Text severity vs. ASR Total → child YSR |
| `data/aim3_item_correlations.csv` | Item-specific severity → child outcomes |
| `data/more_analyses_results.csv` | Extended analyses across all timepoints |

---

## Key finding (one sentence)

Parents who described **social and internal fears** (being alone, confrontation, financial stress, social anxiety) had children with significantly higher total problems **3–4 years later** (r = .48, p < .01), whereas parents describing **external phobias** (spiders, heights, dogs) showed no such association.
