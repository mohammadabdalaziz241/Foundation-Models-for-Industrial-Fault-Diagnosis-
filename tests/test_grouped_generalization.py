import json
from pathlib import Path

from scripts.build_foundation_manifests import GROUPED_PROTOCOL, parse_args
from scripts.run_foundation_experiment import validate_combo
from src.paderborn_preprocessing import BY_BEARING_SPLIT
from src.xjtu_preprocessing import split_xjtu_ssl_val_bearings


def test_frozen_paderborn_groups_and_xjtu_groups_are_disjoint():
    sets=[set(BY_BEARING_SPLIT[s]) for s in ("train","val","test")]
    assert not any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3))
    train,val=map(set,split_xjtu_ssl_val_bearings(seed=42))
    assert len(train)==12 and len(val)==3 and not train&val


def test_grouped_cli_and_matrix():
    args=parse_args(["--protocol",GROUPED_PROTOCOL])
    assert args.protocol==GROUPED_PROTOCOL
    for condition,adaptation in (("S0","scratch"),("S1","ssl"),("S1","linear"),("S1","full"),("S2","ssl"),("S2","linear"),("S2","full")):
        validate_combo(condition,adaptation)
