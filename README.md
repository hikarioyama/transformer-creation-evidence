# Transformer creation evidence

This repository is a public, sanitized evidence release for two separate
interactive educational Transformer runs. It is a provenance record, not a
claim that the runs are equivalent or that every reported result is correct.
The original files were preserved privately; this repository has no original
git history.

## Contents

- `astra/app/` — the Astra tiny-Transformer browser application, tests, and
  public vendor assets.
- `astra/session.jsonl` — the sanitized, ordered Astra session history.
- `astra/provenance.json` and `astra/manifests/` — source/public hashes,
  inventory, usage reconciliation, and omission/redaction records.
- `glm/app/` — the GLM next-token browser application, tests, training source,
  and public vendor assets.
- `glm/session.jsonl` — the sanitized, ordered GLM session history.
- `glm/audit/evidence/` — selected compact audit reports and findings; large
  tensor dumps, caches, screenshots, and machine-local captures are omitted.
- `glm/manifests/` — source/public hashes, usage reconciliation, and exact
  omission/redaction records.
- `tools/sanitize_public.py` — the shared sanitizer used for text, code, JSON,
  and JSONL. It accepts source paths as command-line inputs and never writes
  those paths into a manifest.
- `third_party/THIRD_PARTY_NOTICES.txt` — vendor attribution and license
  information.

## Run labels and scope

Each run is **n=1**. “Astra” and “GLM” are user-facing run/model
attributions; the logged model fields in each history are reported separately
in the provenance files and must not be read as independent identity proofs.
The release does not independently verify EXL3 packaging or an EXL3 claim.

Astra evidence includes the tiny model's model checks, cache-prefix audit,
NumPy comparisons, and browser scenarios. The recorded NumPy audit covers 144
runs and 2,736 checks, with a reported maximum absolute error of
`3.775e-15`; the cache audit covers 840 prefixes. The recorded browser pass
set is retained with the report rather than collapsed into a quality claim.

GLM evidence includes the compact numerical/cache report (213/213 checks
passing there, reported maximum absolute error `3.019806626980426e-14`) and
the separate browser reports. One browser report records 49/51 scenarios
passing; another records 26/33 passing. Their failures and UI defects remain
in the published reports: attention-layer/depth rendering, stale token-chip
selection, missing cached-append embedding rows, stale stage markers, stale
Verify-after-edit behavior, and related observations. These reports are not a
single benchmark.

There is **no unified audit comparison** between Astra and GLM. In particular,
Astra's app/cache/NumPy/browser evidence is not presented as an audit equivalent
to GLM's selected audit reports. Mathematical failures, failed browser
scenarios, ordering, and negative observations are preserved where their source
reports are included.

## Session accounting and disclosures

The histories retain record order, tool-result context where safe, and usage
records. The reconciliation values in the manifests are recomputed from the
published histories and compared with the source-known values.

- Astra has 87 populated records plus its original trailing blank line. Forty
  usage records reconcile to input `107066`, output `33387`, cache-read
  `1207808`, cache-write `0`, reasoning `5055`, and total tokens `1348261`.
- GLM has 247 source JSONL records (3 metadata records and 244 messages: 1
  user, 121 assistant, and 122 tool-result records); its published count and
  usage reconciliation are recorded in `glm/manifests/`.

The GLM session explicitly ran `tools/train.py` for 2,001 steps and wrote
weights/reference artifacts despite a “No training” instruction. That violation
is retained and called out; it is not silently treated as an inference-only
run.

## Sanitization and omissions

The shared sanitizer recursively handles JSON keys and values and also scans
markdown, source, and logs. It replaces emails, secret-bearing assignments,
credential values, local absolute paths and file URLs, private/internal
addresses, local test ports, base64/binary payloads, and full home-directory
listings. Public package and GitHub URLs, model names, usage numbers, ordering,
and useful source text are retained. The final manifests give per-file source
and public SHA-256 values and counts/reasons; they do not contain the private
mapping.

The release intentionally omits `node_modules`, package/browser caches,
`__pycache__`, raw audit tensor captures, process/environment dumps, and
screenshots or embedded image payloads. One embedded PNG object in the GLM
history and image/data URLs in app markup are represented by stable omission
markers. Astra screenshot/trace payloads and the home-directory listing are
likewise omitted or reduced to safe explanatory records. Omission manifests
record source hashes and reasons without retaining omitted bytes. No raw secret
values, private emails, original absolute paths, internal hostnames/IPs,
credentials, or home listings are intended to remain.

## Reproduce the static app smoke

From the repository root, serve either app with a standard static server:

```sh
python3 -m http.server 8000 --directory astra/app
python3 -m http.server 8001 --directory glm/app
```

The checked-in tests and audit reports document the original run; no model
weights need to be downloaded for the toy applications. Do not run the
training script merely to reproduce the browser demonstration.

## Licensing

See `third_party/THIRD_PARTY_NOTICES.txt` before redistributing vendored
Three.js/OrbitControls assets. The notice supplies attribution without
relicensing upstream code.
