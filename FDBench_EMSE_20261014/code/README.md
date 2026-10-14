# Reproducing the reported results

From the extracted package root, run:

    python3 code/verify_package.py --root .
    python3 code/recompute_results.py --root . --out /tmp/fdbench_recomputed

These commands require only the Python standard library. The second command
recomputes task labels, rater agreement, all three confusion matrices, criterion
metrics, ordinal alpha, ICC, Wilson precision references, and the historical
A–F repair rates. It makes no model calls and creates no new observations.

`src/` preserves the original construction, rendering, indexing and scoring
implementations. `historical_sources/` preserves the generation and blind-package
implementations used by the human study. These historical scripts contain their
original filesystem assumptions; adapt paths before executing them. Install the
recorded dependencies and the matching browser only if performing new rendering.
New model calls may differ from the archived outputs even with the same model ID.
The archived model outputs, HTML and screenshots are the evidence for this study.
