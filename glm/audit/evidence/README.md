# GLM audit evidence

This directory contains selected, later audit evidence for the unchanged GLM
app. It is separate from `../session.jsonl`, which is the creation-session
record.

## Included

- `AUDIT.md`: the earlier independent audit report, retained with its original
  verdict and sanitized local references.
- `PI_SESSION_ANALYSIS.md`: the later session-history analysis, including the
  measured usage, ordering, elapsed time, training violation, and chronology.
- `browser/results.json` and `browser/independent_compare.json`: compact browser
  scenario and independent trace-comparison results.
- `numerical/findings.md`, `numerical/tolerances.json`,
  `numerical/weight_capture.json`, and `numerical/instrumentation_manifest.json`:
  compact numerical/cache findings and declared comparison metadata.
- `baseline/`: sanitized baseline and no-git-history evidence; the baseline
  inventory anchors original file integrity without publishing the source root.
- `report/` and `review/`: compact audit integration, validation, and method
  review summaries.

## Intentionally omitted

- Browser screenshots and all other image files: visual artifacts are not
  published. The reports retain textual visual limitations and findings.
- Browser raw snapshots and scenario state dumps: these are large, redundant
  captures containing generated DOM/state and local execution details.
- The aggregate audit result dump and large numerical result/reference/capture
  tensors: these are retained only through compact summaries and omission
  records in `../../manifests/omissions.json`.
- Audit runners, scratch copies, caches, `__pycache__`, and dependency/browser
  installations: not needed to inspect the selected evidence and not part of
  the public app artifact.

All included text/JSON was processed with the shared `sanitize_public.py`
helper. Redaction counts and source/public hashes are in `../../manifests/`.
Redactions remove private paths, local/file URLs, internal addresses, sensitive
keys, opaque payloads, and the full home-directory listing while preserving
record ordering and substantive pass/fail/training findings.
