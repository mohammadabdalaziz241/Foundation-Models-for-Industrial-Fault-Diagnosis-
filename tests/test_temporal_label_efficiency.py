import csv,hashlib,json
from pathlib import Path
from scripts.build_temporal_label_subsets import build
def rows(p):return list(csv.DictReader(p.open()))
def test_temporal_subsets_are_nested_deterministic_and_isolated(tmp_path):
 src=Path("metadata/foundation_manifests/temporal_regions_v1_seed42");target=tmp_path/"m";target.mkdir()
 for name in ("train_supervised.csv","validation_supervised.csv","test_supervised.csv"):(target/name).write_bytes((src/name).read_bytes())
 first=build(target);snap={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (target/"label_efficiency").iterdir()};second=build(target);assert first==second;assert snap=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (target/"label_efficiency").iterdir()}
 val={r["sample_id"] for r in rows(target/"validation_supervised.csv")};test={r["sample_id"] for r in rows(target/"test_supervised.csv")};prev=set()
 expected={("cwru",c) for c in ("normal","inner_race","ball","outer_race")}|{("paderborn",c) for c in ("normal","inner_race","outer_race")}
 for pct in (1,5,10,25,50,100):
  rr=rows(target/"label_efficiency"/f"train_supervised_{pct}pct.csv");ids={r["sample_id"] for r in rr};assert len(ids)==len(rr);assert prev<=ids;prev=ids;assert not ids&(val|test);assert {(r["dataset"],r["dataset_class_label"]) for r in rr}==expected
