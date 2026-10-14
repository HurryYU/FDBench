# FDBench replication data for Empirical Software Engineering

Article: FDBench: Automated Construction and Measurement Validation of Visual Front-end Debugging Benchmark
Authors: Ping Liu and Zihao Yu
Correspondence: linapink1031@gmail.com
Affiliation: Big Data and Intelligent Computing Technology Innovation Center, Chongqing College of Humanities, Science & Technology, Chongqing 401524, China
Version: FDBench v1. Package prepared 14 October 2026.

Online Resource 1 contains the 106 frozen benchmark tasks from 63 source pages,
three independent construction rating sheets, three repair rating sheets covering
160 candidates from 40 sampled tasks, the completed case-to-task/model mapping,
the original model responses, repaired HTML, rating screenshots, historical A–F
outputs and metrics, analysis code, and numeric source data for Tables 1–6 and
Figure 2. All 106 tasks are retained. The 88-task valid subset and the 18 excluded
task identifiers are listed separately.

Online Resource 2 contains 642 knowledge-library page images (558 property pages,
68 guide pages and 16 generated quick-reference pages), 626 MDN source documents,
9200 text chunks, property-to-document ground truth, original text and visual
index payloads and identifier lists, source attribution and document hashes.
Both archives extract into the same `FDBench_EMSE_20261014` directory. Extract
both if reproducing retrieval or inspecting the complete frozen knowledge library.
The core statistical analysis can be reproduced with Online Resource 1 alone.

Start with `source_data/manuscript_map.csv` and `DATA_DICTIONARY.md`. Run:

    python3 code/verify_package.py --root .
    python3 code/recompute_results.py --root . --out /tmp/fdbench_recomputed

The first command checks every archived checksum and candidate source hash.
The second regenerates numeric results without new model calls. Source files,
ratings, model content, HTML and screenshots have not been edited. Selection,
renaming and derived CSV exports are recorded in the provenance lineage CSV files.
The model names, generation conditions and automatically assembled materials
remain identified so that the evidence and provenance can be inspected.

The sampling seed is 20260907; each sampled task contributes four related
candidates. Wilson intervals are descriptive references under an independence
assumption and are not adjusted for task or page clustering. The archived raw
outputs are the study evidence; new model calls or renders may produce different
results. Historical generation scripts retain their original path assumptions.

Model weights, unrelated experiments, annotation module C, credentials, caches,
terminal transcripts and working drafts are excluded. Reviewer identities are
represented by R1, R2 and R3, consistent with the original data.

See `RIGHTS_AND_ATTRIBUTION.md` for third-party terms and outstanding public-deposit
licensing decisions. Repository identifiers have not been assigned.
