# Channel census — methodology_v2 Part 3A.1

Sources: authoritative dataset documentation audited in Part 1 (CWRU
Bearing Data Center pages; JNU repo readme; Hou et al. 2023 Tables IV/V +
Fig. 13/15; MaFaulDa UFRJ site §1.3) and the local file structure. No
test signal content was inspected. Machine-readable: `channel_census.csv`.

## CWRU (48 kHz DE family — retained benchmark subset)

| Channel | Sensor type | Orientation | Location | Candidate for main encoder? |
|---|---|---|---|---|
| DE_time | accelerometer, 12 o'clock housing | radial/vertical | drive-end bearing housing (the seeded-fault bearing) | **PRIMARY** |
| FE_time | accelerometer, 12 o'clock housing | radial/vertical | fan-end bearing housing (remote from fault) | alternative |
| RPM | torque transducer/encoder (scalar) | — | shaft | no (metadata) |

DE and FE are synchronous. No BA channel exists in the 48 kHz family.

## JNU

| Channel | Sensor type | Orientation | Location | Candidate? |
|---|---|---|---|---|
| acc_vertical | PCB MA352A60 accelerometer | vertical (documented) | test-bearing housing | **PRIMARY (only channel)** |

Single-channel dataset; no alternative and no non-vibration channels.

## HIT (full Drive release; 6 synchronous channels @ 25 kHz)

| Channel | Sensor type | Orientation | Location | Candidate? |
|---|---|---|---|---|
| ch1 | displacement (Table IV: KISTLER 8776A50M1) | horizontal | LP rotor | no (displacement, different quantity; note: official GitHub windows were cut from ch1) |
| ch2 | displacement | vertical | LP rotor | no |
| ch3 | acceleration K9000XL | normal-to-casing (radial) | casing, point 3 | **PRIMARY** |
| ch4 | acceleration K9000XL | normal-to-casing (radial) | casing, point 4 | alternative |
| ch5 | acceleration K9000XL | normal-to-casing (radial) | casing, point 5 | alternative |
| ch6 | acceleration K9000XL | normal-to-casing (radial) | casing, point 6 | alternative |

The inter-shaft bearing is internal (combustion cylinder); no sensor sits
on it directly — casing acceleration is the only vibration modality. The
exact axial station of points 3–6 relative to the bearing is not
documented; ch3 is proposed as the deterministic first acceleration
channel, with ch4–ch6 as a future ablation.

## MaFaulDa (8 synchronous columns @ 50 kHz, 2× NI 9234)

| Channel | Sensor type | Orientation | Location | Candidate? |
|---|---|---|---|---|
| col1 | Monarch MT-190 tachometer | — | shaft | no (speed signal) |
| col2 | IMI 601A01 accelerometer | axial | underhang bearing | no |
| col3 | IMI 601A01 accelerometer | **radial** | underhang bearing | **PRIMARY** |
| col4 | IMI 601A01 accelerometer | tangential | underhang bearing | no |
| col5 | IMI 604B31 triaxial | axial | overhang bearing | no |
| col6 | IMI 604B31 triaxial | **radial** | overhang bearing | alternative |
| col7 | IMI 604B31 triaxial | tangential | overhang bearing | no |
| col8 | Shure SM81 microphone | — | ambient | no (acoustic) |

## Cross-dataset synthesis

The closest common physical denominator is a **single radial/vertical
accelerometer on (or nearest to) the monitored bearing/casing**:
CWRU DE_time · JNU acc_vertical · HIT ch3 · MaFaulDa col3.

Caveats to disclose: sensor-to-fault proximity differs (CWRU sensor sits
on the faulted bearing's housing; HIT sensors are on the engine casing,
far from the internal inter-shaft bearing; MaFaulDa col3 is on the
underhang bearing whether or not the fault is there); sensor models,
mounting and sensitivities differ by rig. A fault-position-adaptive
channel policy for MaFaulDa was **rejected** because the channel choice
would encode the label. Channel selection is a recommendation only —
final approval belongs to the human decision in
`part3a_recommendations.yaml`.
