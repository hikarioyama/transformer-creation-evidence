# Numerical and cache audit findings

## Verdict

`results.json` is a numerical **PASS** for the implemented computation and true incremental path: 213/213 executed cases PASS, with no blocked or unexecuted cases. The 100 full-sequence cases cover length 1, short sequences, repeats, alternating patterns, near-context length 15, max context length 16, and seeded random inputs. All captured valid intermediates (including Q/K/V, scores, attention, residual/LN/MLP states, final LN, logits, and probabilities at every position) match the independent reference under the predeclared binary64 rule `abs(a-r) <= 1e-9 + 1e-7*abs(r)`. Largest observed absolute error across result domains is `3.019806626980426e-14`.

`tolerances.json` is the standalone declaration consumed before the reference, cache probes, and comparison. It fixes float64 `atol=1e-9`, `rtol=1e-7`, and mutation read-dependence threshold `1e-12`; no post-hoc relaxation is used. The independent reference is NumPy equations in `reference.py`; it is not a translation of the JavaScript forward function. Fixed JSON weights were extracted without UI rounding (`weights.json`, source hash in `weight_capture.json`). `model-instrument.diff` shows the only changes made to the isolated model copy: Q/K/V projection counters, invocation-input capture, and matrix tags. `verify_integrity.py` reports no changes to original files.

## Cache evidence

All 100 cache cases pass at every prefix, including growth through length 16. Each supplied prefix is checked against the exact case token prefix, and each cache run has exactly one step per supplied token. Each append has exactly one newly invoked Q/K/V projection per layer, and captured projection inputs equal the reference LN1 row for the new position. Prefix K/V row object identities remain unchanged; assembled stored K/V and attention rows match the reference; cache lengths, fresh position, start offset, causal row lengths, logits, and probabilities all match. Recursive finite checks cover fresh rows, stored K/V, assembled attention, logits, and probabilities. A separate isolated mutation probe changes an old stored K row or V row by `+1.0` before a fresh append and observes a logit/attention delta above the predeclared `1e-12` threshold (length-1 cases are explicitly N/A), demonstrating actual old-KV read dependence rather than trusting counters. This is computational reuse evidence, not reliance on the UI's displayed counters.

## Causality and validity

Twelve same-length pairs mutate only a suffix (split positions 1, 4, 8, 15 and additional lengths). Every earlier internal tensor and earlier logit/probability remains invariant. Dict-recursive finite checks cover every captured tensor; all probabilities are finite, strictly positive, and normalized; all represented attention rows are finite, strictly positive, and normalized. Future masked entries are not represented in the ragged score/attention rows, so there are no invalid masked values to accept.

## Scope and handoff gaps

The numerical PASS establishes the core implementation and cache behavior, not faithful presentation by the UI. AuditBrowser independently observed two UI trace issues to carry into the top-level audit: `app.js:262-282` does not copy `tokEmb`/`posEmb`/`embed` when merging a cached append (the T=9 embedding view remains length 8), and the first-layer attention arc group is empty while the second-layer group owns the arcs (reported as a `viz3d.js` arc-group routing issue). These are browser/UI evidence, not failures of the core numerical reference above.

Original-source anchors: full forward and trace tensors `js/model.js:145-229`; cache storage/fresh computation/assembled views `js/model.js:231-357`; UI path selection `js/app.js:284-318`; trace-to-visualization mapping `js/app.js:323-357`.
