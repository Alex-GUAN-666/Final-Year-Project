# Public source copies and provenance

The public package preserves the runnable reconstruction, mathematical data and historical research evidence while removing personal workstation paths and notebook UI payloads.

- Nine notebook files are declared public derivatives. Personal local directories are replaced with generic paths. Widget state and rich HTML/widget display payloads are removed. Code and Markdown sources are unchanged except for those path substitutions; execution counts, plain-text training logs and plain-text evaluation records remain.
- Two historical arithmetic-generation scripts use generic paths instead of a local home directory. They remain historical references; use the `fyp` entrypoints to run the reconstruction.
- The other seven bundled source files, including all six original mathematical data/result files, retain their original bytes. The mathematical examples and generated solutions were not rewritten for publication.
- The two unredacted papers and the unsanitized notebook originals are not published here. Their original hashes remain available for provenance.

`source_manifest.json` retains each supplied input's original `bytes` and `sha256`. A changed public copy additionally has `published_bytes` and `published_sha256`. The artifact auditor checks publication hashes where present and original hashes otherwise; it never treats a public derivative as byte-identical to the original. [The transformation report](../reports/publication_sanitization.json) lists all eleven derivatives.

Original-source audits and superseded `v1` reports describe the files as received before this publication pass. Their original hashes are historical identifiers. Current `artifact_audit_v2.json` and paired-evaluation reconciliation were regenerated from the declared public derivatives; the recovered 12/20 and 15/20 scores and five-question overlap remain the same. No model accuracy was created or changed by this pass.

Archived notebooks can contain historical unrestricted execution and environment assumptions. They are evidence references, not the supported runnable interface. The documented `fyp` interface separates model inference from Docker-isolated code scoring.

## Repository maintenance and attribution

The original research project is credited to Yuzhen Guan and Weizhou Lu. The September 2026 restoration of the repository, including the maintained Python implementation and its documentation, used AI-assisted tooling. This maintenance work is separate from the original experiment records.

The archive uses descriptive filenames for navigation. The original submitted names, previous repository paths and content hashes remain in `source_manifest.json`; the [archive index](../archive/README.md) provides the filename mapping. Renaming does not alter the archived file contents. Some recovered mathematical questions still lack complete upstream attribution. No blanket license is assigned to third-party code, model weights or datasets.
