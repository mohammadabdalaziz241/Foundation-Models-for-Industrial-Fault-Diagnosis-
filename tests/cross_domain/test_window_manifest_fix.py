"""Regression tests for the Paderborn window-manifest bugfix (Option B:
resampled-domain containment) and the narrow completed-checkpoint
carry-forward. No optimizer step, no CUDA, no sealed-test signal access."""
import csv, hashlib, importlib.util, json, math
from pathlib import Path
import pytest
from src.cross_domain import executor as ex

ROOT=Path(__file__).resolve().parents[2]
META=ROOT/'metadata/cross_domain_v1/amendment_003'
def _load(name,rel):
 spec=importlib.util.spec_from_file_location(name,ROOT/rel); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
matrix=_load('cdmatrix_windowfix','scripts/run_cross_domain_matrix.py')
def rows(p):
 with Path(p).open(newline='') as f: return list(csv.DictReader(f))
def key(r): return (r['dataset'],r['recording'],r['original_start'],r['original_end'])

PHANTOM={('N15_M07_F10_KA05_16','6'),('N15_M07_F10_KA08_11','6'),('N15_M07_F10_KA15_16','6')}
BOUNDARY_RECS=('N15_M07_F10_KA09_12','N15_M07_F10_KA16_16','N15_M07_F10_KA30_13','N15_M07_F10_KI07_20','N15_M07_F10_KI08_11','N15_M07_F10_KI08_17','N15_M07_F10_KI17_10','N15_M07_F10_KI21_8')

def test_paderborn_lengths_read_from_actual_files_and_rule_is_resampled_domain():
 # windows in the manifests must match the Option B predicate computed from
 # the ACTUAL per-file lengths for all eleven sub-nominal recordings
 import sys; sys.path.insert(0,str(ROOT))
 from src.paderborn_loader import read_pu_channels
 from src.vibrationclip.preprocessing import expected_resampled_length
 pu=[r for r in rows(META/'fold_HUST/ssl_source.csv') if r['dataset']=='paderborn']
 by_rec={}
 for r in pu: by_rec.setdefault(r['recording'],[]).append(r)
 for rec in tuple(r for r,_ in PHANTOM)+BOUNDARY_RECS:
  bearing=rec.split('_')[3]
  n=len(read_pu_channels(ROOT/f'data/raw_paderborn/{bearing}/{rec}.mat')['vibration_1'])
  rl=expected_resampled_length(n,'paderborn')
  expected={str(i) for i,s in enumerate(s for s in range(0,n,32000) if s*3//8+24000<=rl)}
  got={r['window_index'] for r in by_rec[rec]}
  assert got==expected,(rec,n,rl,sorted(got),sorted(expected))
def test_phantom_rows_absent_everywhere_and_count_4057():
 pu=[r for r in rows(META/'fold_HUST/ssl_source.csv') if r['dataset']=='paderborn']
 assert len(pu)==4057
 for p in META.rglob('*.csv'):
  if 'hust_recording_audit' in p.name or 'target_test_sealed' in p.name: continue
  assert not any(r['dataset']=='paderborn' and (r['recording'],r['window_index']) in PHANTOM for r in rows(p)),p
def test_boundary_windows_retained():
 pu={(r['recording'],r['window_index']) for r in rows(META/'fold_HUST/ssl_source.csv') if r['dataset']=='paderborn'}
 for rec in BOUNDARY_RECS: assert (rec,'6') in pu,rec
def test_other_datasets_unchanged_counts():
 union={}
 for fold in ('PADERBORN','OTTAWA','HUST'):
  for r in rows(META/f'fold_{fold}/ssl_source.csv'): union[key(r)]=r['dataset']
 from collections import Counter
 c=Counter(union.values())
 assert c=={'cwru':697,'ottawa':570,'hust':570,'paderborn':4057},c
def test_sealed_manifests_byte_identical_to_amendment_002():
 old=ROOT/'metadata/cross_domain_v1'
 for fold in ('PADERBORN','OTTAWA','HUST'):
  a={key(r) for r in rows(old/f'fold_{fold}/target_test_sealed.csv')}
  b={key(r) for r in rows(META/f'fold_{fold}/target_test_sealed.csv')}
  assert a==b,fold
def test_all_permitted_rows_satisfy_runtime_guard_predicate():
 audit=_load('cdwbaudit','scripts/audit_window_bounds.py').audit(store=ex.RealBatchStore())
 assert audit['total_invalid']==0,audit
 assert audit['n_rows']==5894 and audit['n_recordings']==683
 assert audit['per_dataset']['paderborn']['rows']==4057
def test_boundary_windows_load_via_store_one():
 store=ex.RealBatchStore()
 pu=[r for r in rows(META/'fold_HUST/ssl_source.csv') if r['dataset']=='paderborn']
 import numpy as np
 for rec in BOUNDARY_RECS[:3]+BOUNDARY_RECS[-1:]:   # incl. sealed-bearing KI21_8 via its SOURCE-manifest row
  r=[x for x in pu if x['recording']==rec and x['window_index']=='6'][0]
  spec=store.one(r); assert spec.shape==(128,128) and spec.dtype==np.float32

# ---------------------------------------------- carry-forward verification
CF_JOBS=('ssl__paderborn__CD-S1__seed42','ssl__paderborn__CD-S1-CM__seed42','ssl__paderborn__CD-CLIP__seed42')
def test_carry_forward_verifies_three_preserved_checkpoints():
 reg={j['job_id']:j for j in matrix.jobs()}
 for jid in CF_JOBS:
  rec,how=matrix.verify_completed_for_skip(reg[jid])
  assert how=='carry-forward-approved'
  assert (rec['target'],rec['arm'],rec['seed'])==(reg[jid]['target'],reg[jid]['arm'],reg[jid]['seed'])
def _tmp_job(tmp,mh):
 out=tmp/'out'; out.mkdir(exist_ok=True)
 (out/'model.pt').write_bytes(b'w')
 (out/'COMPLETE.json').write_text(json.dumps({'target':'ottawa','arm':'CD-S1','seed':43,'selected_step':7,'manifest_hashes':mh,'artifacts':{'model.pt':hashlib.sha256(b'w').hexdigest()}}))
 return {'job_id':'ssl__ottawa__CD-S1__seed43','job_type':'ssl_pretrain','target':'ottawa','arm':'CD-S1','seed':43,'manifest_hashes':{'source_train':'A','source_val':'B','target_train':'C','target_val':'D','target_test':'E'},'checkpoint_dependency':None,'config_hash':'x','output_path':'out'}
def test_carry_forward_cannot_apply_to_unlisted_jobs(tmp_path):
 j=_tmp_job(tmp_path,{'source_train':'A','source_val':'B','target_train':'OLD','target_val':'D','target_test':'E'})
 with pytest.raises(SystemExit,match='not carry-forward approved'):
  matrix.verify_completed_for_skip(j,root=tmp_path)
def test_carry_forward_rejects_source_manifest_mismatch(tmp_path,monkeypatch):
 j=_tmp_job(tmp_path,{'source_train':'WRONG','source_val':'B','target_train':'OLD','target_val':'D','target_test':'E'})
 cf={'jobs':{j['job_id']:{'seed':43,'selected_step':7,'manifest_hashes':{'source_train':'A','source_val':'B'},'artifacts':{'model.pt':hashlib.sha256(b'w').hexdigest()}}}}
 (tmp_path/'cross_domain').mkdir(); (tmp_path/'cross_domain/CARRY_FORWARD_APPROVED.json').write_text(json.dumps(cf))
 with pytest.raises(SystemExit,match='source_train hash mismatch'):
  matrix.verify_completed_for_skip(j,root=tmp_path)
def test_completed_with_matching_hashes_skips_without_carry_forward(tmp_path):
 mh={'source_train':'A','source_val':'B','target_train':'C','target_val':'D','target_test':'E'}
 j=_tmp_job(tmp_path,mh)
 rec,how=matrix.verify_completed_for_skip(j,root=tmp_path)
 assert how=='hash-verified'
