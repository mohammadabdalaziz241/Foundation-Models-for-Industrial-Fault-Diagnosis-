# Native12 migration: source audit and outstanding protocol work

**SOURCE AUDIT COMPLETE; PROTOCOL AUDIT INCOMPLETE; NOT FROZEN; NO TRAINING.**

The official 12k DE and FE fault catalogues were downloaded and their 105 linked MAT files were retrieved directly from CWRU. SHA-256 hashes and MAT variable headers were recorded. No model was fitted, no signal-value diagnostic or TEST performance was computed, and no 48k MAT source was downloaded. These are verified source candidates; final active-retention and per-fold window eligibility await reconciliation with the current Otter v2 registry.

| Fault-bearing location | Ball files | Inner-race files | Outer-race files | Total |
| --- | ---: | ---: | ---: | ---: |
| DE | 16 | 16 | 28 | 60 |
| FE | 12 | 12 | 21 | 45 |
| Total | 28 | 28 | 49 | 105 |

The conservative end/class/diameter grouping yields 7 Ball, 7 InnerRace and 6 OuterRace identity candidates (20 total). These counts are not a substitute for reconciling the current physical-specimen registry. Loads and outer-race orientations were not counted as additional specimens.

All 105 files have a DE signal in their MAT headers. A further FE sensor signal is present in 97; the eight DE 28-mil Ball/InnerRace acquisitions expose DE only. This is a real sensor-availability imbalance to handle before freezing the sensor policy. Fault-bearing location and accelerometer location must not be conflated. No byte-identical duplicate file was found among the 105 official downloads.

## Exact verified native12 acquisition list

Each filename below is the official canonical filename. Source labels, per-file SHA-256, channel variables/shapes and source URLs are in OFFICIAL_NATIVE12_SOURCE_INVENTORY.json. Identity names in this table remain reconciliation candidates.

| Identity candidate | Count | Official filenames |
| --- | ---: | --- |
| DE_B007 | 4 | 118.mat, 119.mat, 120.mat, 121.mat |
| DE_B014 | 4 | 185.mat, 186.mat, 187.mat, 188.mat |
| DE_B021 | 4 | 222.mat, 223.mat, 224.mat, 225.mat |
| DE_B028 | 4 | 3005.mat, 3006.mat, 3007.mat, 3008.mat |
| DE_IR007 | 4 | 105.mat, 106.mat, 107.mat, 108.mat |
| DE_IR014 | 4 | 169.mat, 170.mat, 171.mat, 172.mat |
| DE_IR021 | 4 | 209.mat, 210.mat, 211.mat, 212.mat |
| DE_IR028 | 4 | 3001.mat, 3002.mat, 3003.mat, 3004.mat |
| DE_OR007 | 12 | 130.mat, 131.mat, 132.mat, 133.mat, 144.mat, 145.mat, 146.mat, 147.mat, 156.mat, 158.mat, 159.mat, 160.mat |
| DE_OR014 | 4 | 197.mat, 198.mat, 199.mat, 200.mat |
| DE_OR021 | 12 | 234.mat, 235.mat, 236.mat, 237.mat, 246.mat, 247.mat, 248.mat, 249.mat, 258.mat, 259.mat, 260.mat, 261.mat |
| FE_B007 | 4 | 282.mat, 283.mat, 284.mat, 285.mat |
| FE_B014 | 4 | 286.mat, 287.mat, 288.mat, 289.mat |
| FE_B021 | 4 | 290.mat, 291.mat, 292.mat, 293.mat |
| FE_IR007 | 4 | 278.mat, 279.mat, 280.mat, 281.mat |
| FE_IR014 | 4 | 274.mat, 275.mat, 276.mat, 277.mat |
| FE_IR021 | 4 | 270.mat, 271.mat, 272.mat, 273.mat |
| FE_OR007 | 12 | 294.mat, 295.mat, 296.mat, 297.mat, 298.mat, 299.mat, 300.mat, 301.mat, 302.mat, 305.mat, 306.mat, 307.mat |
| FE_OR014 | 5 | 309.mat, 310.mat, 311.mat, 312.mat, 313.mat |
| FE_OR021 | 4 | 315.mat, 316.mat, 317.mat, 318.mat |

## Excluded sources

Exclude all 52 acquisitions linked by the official 48k DE fault catalogue, covering Ball, InnerRace and OuterRace at the listed 7/14/21-mil severities and loads/orientations:

109.mat, 110.mat, 111.mat, 112.mat, 122.mat, 123.mat, 124.mat, 125.mat, 135.mat, 136.mat, 137.mat, 138.mat, 148.mat, 149.mat, 150.mat, 151.mat, 161.mat, 162.mat, 163.mat, 164.mat, 174.mat, 175.mat, 176.mat, 177.mat, 189.mat, 190.mat, 191.mat, 192.mat, 201.mat, 202.mat, 203.mat, 204.mat, 213.mat, 214.mat, 215.mat, 217.mat, 226.mat, 227.mat, 228.mat, 229.mat, 238.mat, 239.mat, 240.mat, 241.mat, 250.mat, 251.mat, 252.mat, 253.mat, 262.mat, 263.mat, 264.mat, 265.mat.

Also exclude 97.mat, 98.mat, 99.mat and 100.mat (normal baselines), all Normal_* aliases, duplicates and every derivative of any excluded acquisition. The older public registry explicitly documents the normal-baseline 48k exception. Matching every local Otter alias and cache descendant to this denylist remains pending.

## The requested ten-item audit

| Item | Evidence/status |
| --- | --- |
| 1. Exact retained list/count | Exact official native12 source candidates: 105; verified file hashes listed. Final active retained/window-eligible inventory pending current Otter reconciliation. |
| 2. 48k exclusions | Exact 52-file official 48k fault denylist plus four 48k normal baselines; local aliases/derivative traversal pending. |
| 3. Physical specimens/class | Candidate counts 7/7/6; physical identity reconciliation pending. |
| 4. Every fold TRAIN/VAL/TEST allocation | Not rebuilt: current protected specimen-role registry is unavailable here. No replacement allocation is invented. |
| 5. DE/FE by class/split | Source-level fault-location counts above; header sensor availability recorded. Split-level tables pending actual allocation/policy. |
| 6. Recordings/windows by class/split | Source recording totals above. Split and window counts pending current window settings/roles. |
| 7. No 48k in active CWRU manifests | Source candidate inventory has zero 48k or normal-baseline IDs. Current Otter active manifests/loaders/caches have not yet been migrated or certified. |
| 8. Protocol name/version/hashes | Proposed pcste_v2_cwru_native12_specimen_v1; source inventory/hash available. No protocol freeze digest or new normaliser hashes exist yet. |
| 9. Old screen eligibility | Old mixed-rate C0/C1/C2 are history only; all six new screen cells require fresh native12 training. |
| 10. New matrix/cost | Six exact planned IDs in NATIVE12_RUN_PLAN.json; 40.9–42.6 GPU-hours at historical throughput, pending native12 cost refinement. |

## Why the freeze cannot yet be completed here

The connected public repository is at c07107ab4a4c72907509e342ba3eb65dd4424709 (31 August 2026). Its GitHub API has no commit 04cfd92e and its tree lacks the current pcste_v2 X1 directories. Therefore the current Otter physical/role registries, native12 representation/window settings, v2 launch integration and unchanged JNU/HIT/MaFaulDa baseline hashes must be supplied or processed on Otter. The 105 raw native12 sources are available in this workspace, but they cannot establish these missing project-specific facts.

The migration specification includes an execution handoff to complete this work on the current Otter branch and return the audit before launching anything. It does not instruct checking out the older public branch over the current worktree.

## Verification and hashes

- All 105 official native12 downloads have readable MAT headers and SHA-256 hashes.
- Catalogue acquisition IDs are disjoint from all 52 official 48k fault IDs and normal IDs 97–100.
- Three negative source-filter checks reject a 48k rate/category, a 48k category relabelled 12k, and a 12k category relabelled 48k before downloading.
- The six-run matrix has unique IDs, prohibits historical control reuse, and sets training_allowed=false.
- Source-helper syntax and matrix consistency checks passed. Current-executor integration is untested and remains a freeze blocker.

Source inventory SHA-256: `bd97cebced8be3a1425ba5edbff116386edf7b63756282376d1c715a0a22f611`. This is an inventory-file hash, **not** a frozen experimental-protocol hash.

Sources: [official 12k DE](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data), [official 12k FE](https://engineering.case.edu/bearingdatacenter/12k-fan-end-bearing-fault-data), [official 48k DE](https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data), [repository rate provenance](https://github.com/mohammadabdalaziz241/Foundation-Models-for-Industrial-Fault-Diagnosis-/blob/c07107ab4a4c72907509e342ba3eb65dd4424709/src/methodology_v2/registry.py).
