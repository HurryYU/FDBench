# Data fields and analysis units

All tabular files use UTF-8 CSV with a header. JSONL files contain one JSON record
per line. Empty CSV cells denote no value; `null` in JSON denotes no value. The
original rating sheets contain no unresolved `U` ratings and no free-text notes.
Ratios are dimensionless; fields ending in `_pct` are percentages.

| Dataset or fields | Meaning and allowed values |
|---|---|
| task_id | Frozen task identifier T0000 through T0105 |
| source_page | Original Design2Code page identifier; used to identify shared-source clusters |
| category | layout, text, boxmodel or stacking |
| complaint | Original task description; retained without linguistic editing |
| op_id | Controlled CSS mutation operator |
| gt_property, gt_line | Mutated CSS property and one-based line in the task HTML |
| gt_before, gt_after | Original and mutated declaration strings; empty gt_after denotes deletion |
| target_boxes, student_boxes | Visible-element DOM paths, tags and bounding boxes for correct and defective pages |
| gt_regions | Bounding boxes of geometrically changed elements; not human root-cause regions |
| x, y, w, h | Position and width/height in CSS pixels at viewport 1280 by 960, device scale factor 1 |
| diff_stats.n_mismatch | Number of geometrically mismatching elements |
| diff_stats.ratio | Mismatch count divided by the reference visible-element count |
| diff_stats.max_offset_px | Maximum element displacement in pixels |
| target_png, student_png | Paths relative to data/tasks |
| case_id | Blinded construction case A-001 through A-106 |
| candidate_id | Blinded repaired candidate B-001 through B-160 |
| R1, R2, R3 | Anonymous identifiers for the three independent raters |
| defect_visible, single_primary_defect, page_intact, css_like_defect | Binary construction ratings: 1 yes, 0 no; protocol also permits U unresolved |
| construction_valid | Majority yes on each of defect_visible, single_primary_defect and page_intact |
| defect_resolution | 0 unresolved, 1 partly resolved, 2 fully resolved, U cannot judge |
| new_regression | 0 no new visible regression, 1 new regression, U cannot judge |
| target_similarity_1_5 | Ordinal similarity: 1 clearly dissimilar through 5 essentially the same |
| human_fix | A rater succeeds if defect_resolution is 2 and new_regression is 0; final endpoint is three-rater majority |
| automatic_fix | Original frozen structural repair verdict; 0/false failure, 1/true success |
| generator, condition, replicate | Original model identifier, correct or shuffled evidence, and replicate number |
| prompt_sha256, raw_response_sha256, fixed_html_sha256, render_sha256 | Original frozen candidate provenance hashes |
| raw_response, parsed, edits, fixed_html_final | Original model output, parsed output, proposed edits and repaired HTML |
| TP, TN, FP, FN | Automatic verdict relative to the majority human verdict |
| scope | all: 160 candidates; valid: 128 from 32 valid tasks; invalid: 32 from 8 invalid tasks |
| arm / configuration | Historical system configuration A through F; descriptive usage examples |
| fix | Historical automatic repair outcome, 0 or 1 |
| iters, latency_s, in_tokens, out_tokens | Attempts, runtime in seconds, input/output token counts |
| doc_id, chunk_id | Frozen knowledge-page and text-chunk identifiers |
| subset | KB-1 reference pages; KB-2 guides; KB-3 generated quick-reference pages |
| prop, gold.json | CSS property and rule-based relevant-reference-page mapping |
| in_text_index, in_visual_index | 0 absent or 1 present; KB-3 is absent from the text index |
| source_sha256, package_sha256 | Original source and packaged file hashes in the lineage manifest |

The observation units are 106 tasks for construction validation and 160 repair
candidates clustered within 40 sampled tasks for criterion validation. Task
identifiers and source_page retain both levels of dependence. Percentages and
Wilson intervals must not be interpreted as cluster-adjusted inference.
