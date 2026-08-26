# CWRU reconstruction supplemental extraction

Read-only CWRU-only extraction from the authoritative frozen-checkpoint reconstruction replay. Generated 2026-08-22T20:20:45.370496+00:00. No training, optimizer, backward pass, checkpoint selection, or writes to original experiment artifacts occurred.

The reported quantities are **masked normalized log-STFT reconstruction metrics**, not waveform metrics. The exact validation membership and deterministic masks used by the frozen SSL experiment are retained. All 9/9 fold×seed cells passed saved-MSE reproduction at tolerance 1e-06.

The four-dataset comparison preserves the existing authoritative JNU/HIT/MaFaulDa results. Macro-3 remains JNU + HIT + MaFaulDa; CWRU has not been added to or substituted into Macro-3.

Deterministic figure selection: lexicographic middle CWRU validation window from SSL fold 1, seed 42 (`f1:CWRU:cwru_X163:DE:validation:192000-240000`), matching the existing analysis procedure.
