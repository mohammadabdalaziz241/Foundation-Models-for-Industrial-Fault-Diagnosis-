# Dataset census — methodology_v2 Part 1

One section per candidate dataset. Values marked 'not documented' were not stated by the original source and are deliberately left unresolved rather than inferred.

## CWRU

| Field | Value |
|---|---|
| original_source | https://engineering.case.edu/bearingdatacenter |
| local_path | data/raw (12k DE) + data/raw_cwru_48k (48k DE) |
| n_raw_files | 116 |
| n_recordings | 112 |
| machinery_type | 2 hp motor test stand, seeded-fault drive-end bearing (SKF 6205-2RS JEM; NTN equivalent for 28 mil) |
| sensor_type | accelerometer (DE / FE / BA where present) |
| sensor_channels | DE; DE+FE; DE+FE+BA |
| sampling_rates_hz | 12000; 48000 |
| recording_duration_s | 1.32892-10.2431 |
| samples_per_recording | 63788-491446 |
| rpm_speeds | 1718-1797 |
| load_conditions | 0hp; 1hp; 2hp; 3hp |
| fault_classes | B007; B014; B021; B028; IR007; IR014; IR021; IR028; Normal; OR007@12; OR007@3; OR007@6; OR014@6; OR021@12; OR021@3; OR021@6 |
| fault_severities | 14 mil; 21 mil; 28 mil; 7 mil |
| fault_locations | ball; healthy; inner_race; outer_race |
| bearing_positions | drive_end |
| physical_bearing_identity | fault specimen inferable from fault spec (type+size); per-specimen serials not published; OR clock positions may share a specimen (not documented) |
| experiment_run_identity | one .mat per acquisition; consecutive canonical file numbers across the 4 loads of a specimen |
| multi_file_same_experiment | yes: same specimen recorded at 4 loads and at both 12 kHz and 48 kHz |
| official_train_test_split | none |
| file_format | MATLAB .mat (v5) |
| licence_provenance | publicly distributed by CWRU Bearing Data Center; no explicit licence text on site |

## JNU

| Field | Value |
|---|---|
| original_source | https://github.com/ClarkGableWang/JNU-Bearing-Dataset |
| local_path | data/raw_jnu/JNU-Bearing-Dataset (git 75b33611b516) |
| n_raw_files | 12 |
| n_recordings | 12 |
| machinery_type | rotating machinery test rig, rolling bearing |
| sensor_type | PCB MA352A60 accelerometer, vertical direction, 1 channel |
| sensor_channels | acc_vertical |
| sampling_rates_hz | 50000 |
| recording_duration_s | 10.01-30.03 |
| samples_per_recording | 500500-1.5015e+06 |
| rpm_speeds | 600-1000 |
| load_conditions | not applicable / not documented |
| fault_classes | ib; n; ob; tb |
| fault_severities | 0.3mm x 0.05mm wire-cut dent |
| fault_locations | healthy; inner_race; outer_race; rolling_element |
| bearing_positions | not documented |
| physical_bearing_identity | one seeded specimen per condition (inferred), reused across speeds |
| experiment_run_identity | one CSV per (condition, speed) |
| multi_file_same_experiment | same specimen across 3 speeds |
| official_train_test_split | none |
| file_format | single-column CSV |
| licence_provenance | no licence file in repository |

## HIT

| Field | Value |
|---|---|
| original_source | https://github.com/HouLeiHIT/HIT-dataset |
| local_path | data/raw_hit/HIT-dataset (github, ef1765597751) + data/raw_hit/gdrive_full/HIT-dataset |
| n_raw_files | 134 |
| n_recordings | 134 |
| machinery_type | modified real aero-engine (dual rotor), inter-shaft bearing |
| sensor_type | 2x displacement + 4x acceleration, 6 channels |
| sensor_channels | disp1+disp2+acc1+acc2+acc3+acc4 |
| sampling_rates_hz | 25000 |
| recording_duration_s | 14.7456 |
| samples_per_recording | 368640 |
| rpm_speeds | LP 1000-5000, HP 1200-6000 r/min, 28 planned speed groups (paper Table V) |
| load_conditions | not applicable / not documented |
| fault_classes | 0; 1; 2 |
| fault_severities | depth 0.5 mm, length 0.5 mm; depth 0.5 mm, length 1.0 mm |
| fault_locations | healthy; inner_race; outer_race |
| bearing_positions | inter_shaft |
| physical_bearing_identity | session -> physical bearing documented (data1/2 healthy, data3/4 inner x2 specimens, data5 outer x1) |
| experiment_run_identity | session (assembly) x speed-group; speed value recorded per series in column 7 |
| multi_file_same_experiment | one .npy per session holds all speed groups |
| official_train_test_split | YES (github xtrain/xtest) — but window-level random; see integrity report |
| file_format | .npy (full) / .mat shards (github) |
| licence_provenance | paper CC BY 4.0; dataset licence not stated |

## MAFAULDA

| Field | Value |
|---|---|
| original_source | https://www02.smt.ufrj.br/~offshore/mfs/page_01.html |
| local_path | data/raw_mafaulda/full (extracted from full.zip) |
| n_raw_files | 1951 |
| n_recordings | 1951 |
| machinery_type | SpectraQuest Machinery Fault Simulator ABVT (single rig) |
| sensor_type | 2x triaxial-equivalent accelerometer sets + tachometer + mic |
| sensor_channels | 8ch(tacho;uh_ax,rad,tan;oh_ax,rad,tan;mic) |
| sampling_rates_hz | 50000 |
| recording_duration_s | 5 |
| samples_per_recording | 250000 |
| rpm_speeds | 724.992-3735.55 |
| load_conditions | 10g; 15g; 20g; 25g; 30g; 35g; 6g; added_mass_0g; added_mass_20g; added_mass_35g; added_mass_6g |
| fault_classes | horizontal-misalignment; imbalance; normal; overhang/ball_fault; overhang/cage_fault; overhang/outer_race; underhang/ball_fault; underhang/cage_fault; underhang/outer_race; vertical-misalignment |
| fault_severities | 0.51mm; 0.5mm; 0.63mm; 1.0mm; 1.27mm; 1.40mm; 1.5mm; 1.78mm; 1.90mm; 10g; 15g; 2.0mm |
| fault_locations | bearing_ball_fault; bearing_cage_fault; bearing_outer_race; healthy; horizontal_misalignment; imbalance; vertical_misalignment |
| bearing_positions | overhang; underhang |
| physical_bearing_identity | 3 defective bearings (ball/cage/outer), each reused at underhang AND overhang positions; single rig |
| experiment_run_identity | one 5 s CSV per (configuration, speed) |
| multi_file_same_experiment | same fault configuration recorded at ~49 speeds |
| official_train_test_split | none |
| file_format | 8-column CSV |
| licence_provenance | publicly distributed by UFRJ/SMT; no explicit licence text on site |
