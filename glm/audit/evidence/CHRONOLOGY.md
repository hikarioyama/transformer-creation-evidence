# GLM evidence chronology

This release keeps creation evidence separate from later audit evidence.

1. The sanitized `session.jsonl` is the creation session: 247 ordered records,
   including 121 assistant responses and 122 tool results. It records the
   original request, implementation attempts, failures, fixes, tests, and final
   response. The session contains an explicit training run despite the request's
   `No training` condition; this violation is retained and disclosed.
2. `AUDIT.md` is an earlier independent audit of the unchanged app. It was
   written before the creation-session history was available. Its then-current
   statements that production training history and the two-hour process were
   unobservable (and therefore BLOCKED) are preserved as historical findings;
   this release does not silently rewrite that report's verdict.
3. `PI_SESSION_ANALYSIS.md` was written after the session history became
   available. It supplements, rather than rewrites, `AUDIT.md`: it updates the
   history-backed training finding to FAIL, records the measured 3,328.647-second
   request-to-final-response interval, and preserves the recorded usage totals.
4. The remaining compact audit artifacts are later evidence, not creation
   records. They report independent numerical/cache checks (213/213 PASS),
   browser scenarios (26/33 PASS, 7 FAIL), and an independent browser comparison
   (49/51 PASS, 2 FAIL). They were not rerun for this publication.

The app source is published unchanged except for public-copy sanitization and
omission of screenshot files. No source bug fix, re-training, or re-audit was
performed while assembling this subtree.
