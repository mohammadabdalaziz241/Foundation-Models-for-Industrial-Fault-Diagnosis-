# PC-STE v2: strictly native 12 kHz CWRU

Status: **AUDIT_INCOMPLETE — NOT_FROZEN — TRAINING_BLOCKED**.
Proposed protocol ID: `pcste_v2_cwru_native12_specimen_v1`.

This revision implements the user's new experimental-plan decision. It specifies the repository migration; it does not certify that the current Otter manifests, folds or normalisers have already been rebuilt. JNU, HIT and MaFaulDa retain their current Otter v2 configuration.

## Source eligibility and historical lineage

Allow only acquisitions from the official **12k Drive End Bearing Fault Data** and **12k Fan End Bearing Fault Data** categories, matched by canonical file ID, source URL and verified bytes. A directory name or mutable sampling-rate field is insufficient proof of native rate. Preserve 12,000 Hz processing; no 48→12 resampling is permitted.

Exclude the entire 48k DE source category and all its renamed copies, windows, spectrograms, embeddings and resampled derivatives. An apparently 12k channel inside an excluded 48k acquisition does not make that file eligible. Exclude normal baselines `97.mat`–`100.mat`, their `Normal_*` aliases and duplicate copies: the connected repository documents those files as 48k even in its 12k directory. Keep the present three fault classes; no Healthy class is added.

Use an explicit native12 allowlist in the active source root and loaders. Historical 48k evidence stays in a separate read-only history namespace and is never reached by new loader fallbacks. This exclusion does not require destroying previous experiment evidence.

All mixed-rate Stage B/C, seed-replication and X1 results are diagnostic history, not the final CWRU benchmark. X1 remains NOT_ADVANCED; Stage C remains PARTIAL. The old C0/C1/C2 runs are ineligible as matched controls or initialisations for native12. Rebuild any future CWRU SSL pool, normaliser, teacher or pretrained encoder provenance too. Removing 48k only from validation/TEST is insufficient. New valid SSL checkpoints may be shared across downstream arms only when their pretraining inputs and objective are identical.

Sources: [official categories](https://engineering.case.edu/bearingdatacenter/download-data-file), [12k DE](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data), [12k FE](https://engineering.case.edu/bearingdatacenter/12k-fan-end-bearing-fault-data), [repository native-rate findings](https://github.com/mohammadabdalaziz241/Foundation-Models-for-Industrial-Fault-Diagnosis-/blob/c07107ab4a4c72907509e342ba3eb65dd4424709/src/methodology_v2/registry.py).

## Physical identity and folds

Reconcile native12 acquisitions with the **current Otter physical specimen registry**. Loads, channels, recording repetitions, renamed copies and former 12/48 variants do not create new specimens. Preserve conservative grouping of outer-race orientations unless documented physical identity evidence establishes otherwise. Report confirmed identities separately from conservative assumptions.

First preserve each fold's current physical TRAIN/VAL/TEST roles while replacing its acquisition universe with eligible native12 recordings of those identities. A native12 acquisition of an existing TEST specimen remains TEST in that fold. Do not move difficult validation specimens into TRAIN or promote historical TEST specimens into development. Reconcile prior data-use history as well as labels.

If native coverage makes an allocation infeasible, give a metadata-only feasibility report and deterministic proposed revision, identifying each role change and reason. Preserve protected TEST identities. Do not choose allocations from model performance; if a necessary change would expose a protected TEST identity, keep the protocol blocked.

Require every class in every split and **at least two usable physical TRAIN specimens per class after window eligibility**; target three or more where feasible. Report attainable counts rather than assuming three. All recordings/channels/windows of a specimen share one split within each fold. Seeds change training randomness, not specimen assignments. Across-fold role reuse follows the documented fold protocol and does not create independent physical populations.

TEST access during this audit is limited to registration metadata, byte hashes, structural sample counts and deterministic window indices required to document the seal. No TEST waveform statistics, embeddings, predictions, losses or performance inform source eligibility, representation design, normalisation or selection. Specify structural exclusion rules without consulting outcomes; normaliser fitting sees TRAIN only.

## Sensor location, balance and windows

Record fault-bearing end and accelerometer location separately, alongside bearing family, actual MATLAB channel, native rate, specimen, recording, diameter, outer-race orientation, load and RPM provenance. The source category identifies the faulty bearing; channel suffixes identify the sensor. The bearing families have different geometries. [CWRU bearing information](https://engineering.case.edu/bearingdatacenter/bearing-information)

Preserve the current class-independent CWRU sensor-selection policy if reproducible on eligible files. If a fixed sensor is used for every class, verify that it is actually fixed. If multiple sensor locations are used, require each used location across all TRAIN classes and audit equal location exposure within class; use TRAIN-only exposure balancing where needed. Never choose the nearest sensor from ground-truth fault identity or allocate one sensor to one class. Missing channels and unequal lengths must be explicit. If a replacement sensor policy is necessary, specify a class-independent choice before freeze; this migration does not silently introduce a two-channel architecture.

Produce DE/FE contingency tables for BOTH meanings at specimen, recording and window levels, by class/split/fold. Audit bearing-family/severity coverage and class/specimen exposure. Metadata checks control observed confounding; they cannot prove that a model has learned no nuisance information. Keep natural VAL/TEST supports and every eligible held-out specimen.

Read the current executor's **12k** window/STFT parameters. Preserve its physical window duration and representation choices where compatible; do not relabel 48k sample counts/frequency coordinates as 12k. Record window length/stride, FFT/hop, padding/tail rules and valid bands. Show structural short-file exclusions and zero-window specimens before checking TRAIN support.

## Regeneration and enforcement

Create a new versioned namespace using the actual executor's conventions. Regenerate every CWRU-dependent source/identity registry, recording/window manifest, fold summary, label subset, sampler plan, representation/cache manifest, N2b normaliser, run config and fingerprint. Fit CWRU N2b on new TRAIN windows only, with the existing log1p standard-deviation floor of 0.05. All arms use the same normaliser for a fold. Store fit-ID hashes and normaliser byte hashes.

JNU/HIT/MaFaulDa source IDs, labels, split/window rows, representation settings, normaliser bytes and sampling policies remain exactly as in the **current Otter v2 baseline**, including radial-only stratified MaFaulDa. New composite registry hashes may change, but prove unchanged per-dataset rows and normaliser hashes. Do not reconstruct these datasets from the older public dissertation branch.

Freeze one optimiser-step count per fold shared by C0/C1/C2. Preserve the current non-CWRU exposure/stream rather than letting the new CWRU population silently change epoch length. Generate the original mixed stream first, then replace only CWRU slots in C1/C2. Match C1/C2's complete balanced stream. Derive the contrastive positive-count bound from the actual new multiplicities; 1.258436 is not a universal floor.

Integrate a hard loader/launcher check: every CWRU source and ancestor must be native12-allowlisted, both original/current rates must be 12000, resampling lineage must be absent, hashes must match, and caches/normalisers must belong to the new protocol. Reject missing freeze digests, stale paths, 48k files relabelled as 12k and 48→12 derivatives. Test these negative cases. A document alone does not enforce this gate; current-executor integration is required before FROZEN status.

Hash canonical manifests and normalisers, then a deterministic bundle index of relative paths/hashes plus protocol, code, representation and sampler settings. Exclude timestamps, machine-specific absolute paths and the digest file itself from its own digest input. A source-inventory hash or Git commit does not substitute for the complete protocol freeze hash.

## Fresh run matrix and prospective criteria

The proposed first comparison after audit delivery and freeze is six fresh four-domain S0 runs, 50 epochs each:

| Run ID | Arm | Fold | Seed |
| --- | --- | --- | --- |
| pcstev2_n12v1_c0_f1_s42 | C0 N2b | 1 | 42 |
| pcstev2_n12v1_c1_f1_s42 | C1 specimen-balanced sampling | 1 | 42 |
| pcstev2_n12v1_c2_f1_s42 | C2 same sampling + contrastive loss | 1 | 42 |
| pcstev2_n12v1_c0_f3_s42 | C0 N2b | 3 | 42 |
| pcstev2_n12v1_c1_f3_s42 | C1 specimen-balanced sampling | 3 | 42 |
| pcstev2_n12v1_c2_f3_s42 | C2 same sampling + contrastive loss | 3 | 42 |

Preserve the 192-D model, heads, optimiser/schedule and four-domain validation MacroDomainF1 selector (strict improvement, earlier on ties). Match initial weights, head seeds, new native12 inputs, normalisers and non-CWRU streams within fold–seed cells. C2 retains tau=0.10 and coefficient=0.10 within the existing CWRU domain weight. No coefficient or temperature tuning accompanies migration.

Carry X1's numeric intervention gates forward in a NEW frozen criteria file, comparing against NEW C0: both folds require CWRU Macro-F1 ≥0.70, gain ≥0.10, every specimen recall ≥0.50, Macro-3 change ≥−0.02, each other-domain F1 change ≥−0.05, MaFaulDa HM/U-cage recall ≥0.60, and every MaFaulDa class recall change ≥−0.05. If both candidates pass, prefer C1 unless C2 adds mean CWRU F1 ≥0.02 and loses no more than 0.01 against C1 on either fold. If neither passes, stop the intervention; report C0's absolute adequacy separately. Historical gates/outcomes are not rewritten.

At the completed screen's historical 491–511 s/epoch, six runs cost **40.9–42.6 GPU-hours**, about **13.6–14.2 h in two waves on four comparable free GPUs**. This is a reference estimate, not new-protocol measured throughput. Allow 42–54 GPU-hours as a provisional contingency envelope; refine from the audited step counts and shapes. No training timing probe precedes audit/freeze.

For a later selected candidate vs C0, confirmation is 2 arms ×3 folds ×3 seeds (42/1337/2026) =18 cells. Four native12 cells exist after screening, leaving **14 additional runs**, or **20 new runs total**, about 136.4–141.9 GPU-hours at historical throughput. Confirmation retains the prior per-fold mean F1/gain targets, no paired cell below −0.05, every candidate specimen recall ≥0.50 and all shared guards per cell.

If a separately justified final study retains all three arms, the matrix is 27 cells, with 21 additional after screening, about 184.1–191.6 GPU-hours total. This is an optional alternative, not an automatic additional launch. SSL/S1, low-label and compression matrices require later specification and native12-compatible initialisation. Do not carry forward the obsolete 800/1,250 GPU-hour roadmap estimates.

## Ten-item audit and Otter execution handoff

Start from the current Otter worktree containing reported commit `04cfd92e` and the v2 executor, or a documented descendant. The public branch inspected here is `c07107ab4a4c72907509e342ba3eb65dd4424709`, dated 31 August, and lacks that commit. Import these new migration files onto the current worktree; do not replace it with the older public branch. Read applicable AGENTS.md instructions.

Implement the new version, reconcile/download native12 sources, regenerate CWRU artifacts, integrate the launch guard, and verify other-dataset preservation. `audit_official_sources.py` supplies source catalogue/hash evidence only; it neither generates folds nor fits normalisers nor certifies a freeze.

Before any training, return:

1. Exact retained official IDs/names/URLs, local aliases and SHA-256 hashes.
2. Every excluded 48k source/category/alias/derivative, including misplaced normal baselines.
3. Reconciled physical specimens per class, with identity provenance/confidence.
4. Every fold's complete TRAIN/VAL/TEST specimen allocation and documented changes.
5. DE/FE distributions by class/split for both fault-bearing and sensor location.
6. Recording/window counts by class/split, structural exclusions and balanced TRAIN exposure.
7. Automated evidence of zero excluded-source ancestry in every active CWRU registry/cache/loader.
8. New protocol identifier, commit, complete per-file hash inventory and freeze digest.
9. Explicit old-run ineligibility and fresh C0/C1/C2 requirements.
10. Exact run matrix, steps/epoch and updated estimated GPU cost.

Commit the audit and present it to the user. Mark FROZEN only after every mandatory item has concrete passing evidence. Otherwise keep `training_allowed=false` and report blockers while completing all feasible work. This migration task ends with the audit/freeze report and launches **no training**. TEST stays sealed. The prior mixed-rate frozen-probe diagnostic is superseded as the next action.
