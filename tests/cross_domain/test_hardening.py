"""Post-re-audit hardening tests: launch gate, parameter accounting, frozen
analysis, diagnostics spec, recovery/lock safety, raw-hash memoization.
No test performs an optimizer step, CUDA work, or sealed-test sample access."""
import hashlib, importlib.util, json, os, subprocess, sys
from pathlib import Path
import pytest, torch
from src.cross_domain import executor as ex

ROOT=Path(__file__).resolve().parents[2]
def _load(name,rel):
 spec=importlib.util.spec_from_file_location(name,ROOT/rel); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
matrix=_load('cdmatrix_hardening','scripts/run_cross_domain_matrix.py')
analysis=_load('cdanalysis','scripts/analyze_cross_domain.py')

# ------------------------------------------------------------- launch gate
GOOD_COMMIT='ab12cd34ef56ab12cd34ef56ab12cd34ef56ab12'
def make_marker_root(tmp,status='READY_FOR_TRAINING',commit=GOOD_COMMIT[:12],jobs='207',breakfield=None,omit=None,malformed=False):
 root=tmp/'root'; root.mkdir(exist_ok=True)
 fields={}
 for key,rel in matrix.MARKER_HASHED_FILES.items():
  p=root/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(f'content of {rel}\n')
  fields[key]=hashlib.sha256(p.read_bytes()).hexdigest()
 if breakfield: fields[breakfield]='0'*64
 if omit: fields.pop(omit)
 lines=[f'status: {status}',f'git_commit: {commit}',f'planned_training_jobs: {jobs}']
 lines+= [f'{k}: {v}' for k,v in fields.items()]
 if malformed: lines.insert(1,'this line has no separator')
 (root/'cross_domain').mkdir(exist_ok=True); (root/'cross_domain/READY_TO_TRAIN').write_text('\n'.join(lines)+'\n')
 return root
ENV={'CROSS_DOMAIN_ALLOW_TRAINING':'1'}

def test_gate_positive_logic(tmp_path):
 root=make_marker_root(tmp_path)
 m=matrix.verify_launch_authorization(root=root,environ=ENV,commit=GOOD_COMMIT)
 assert m['status']=='READY_FOR_TRAINING' and m['planned_training_jobs']=='207'
def test_gate_refuses_missing_marker(tmp_path):
 root=make_marker_root(tmp_path); (root/'cross_domain/READY_TO_TRAIN').unlink()
 with pytest.raises(SystemExit,match='absent'): matrix.verify_launch_authorization(root=root,environ=ENV,commit=GOOD_COMMIT)
def test_gate_refuses_reaudit_status(tmp_path):
 root=make_marker_root(tmp_path,status='READY_FOR_INDEPENDENT_RE_AUDIT')
 with pytest.raises(SystemExit,match='does not authorize'): matrix.verify_launch_authorization(root=root,environ=ENV,commit=GOOD_COMMIT)
def test_gate_refuses_wrong_commit(tmp_path):
 root=make_marker_root(tmp_path)
 with pytest.raises(SystemExit,match='git_commit'): matrix.verify_launch_authorization(root=root,environ=ENV,commit='ffffffffffff'+GOOD_COMMIT[12:])
def test_gate_refuses_wrong_amendment_hash(tmp_path):
 root=make_marker_root(tmp_path,breakfield='amendment_003_sha256')
 with pytest.raises(SystemExit,match='amendment_003_sha256'): matrix.verify_launch_authorization(root=root,environ=ENV,commit=GOOD_COMMIT)
def test_gate_refuses_wrong_protocol_hash(tmp_path):
 root=make_marker_root(tmp_path,breakfield='base_protocol_sha256')
 with pytest.raises(SystemExit,match='base_protocol_sha256'): matrix.verify_launch_authorization(root=root,environ=ENV,commit=GOOD_COMMIT)
def test_gate_refuses_wrong_registry_hash(tmp_path):
 root=make_marker_root(tmp_path,breakfield='job_registry_sha256')
 with pytest.raises(SystemExit,match='job_registry_sha256'): matrix.verify_launch_authorization(root=root,environ=ENV,commit=GOOD_COMMIT)
def test_gate_refuses_missing_env(tmp_path):
 root=make_marker_root(tmp_path)
 with pytest.raises(SystemExit,match='CROSS_DOMAIN_ALLOW_TRAINING'): matrix.verify_launch_authorization(root=root,environ={},commit=GOOD_COMMIT)
def test_gate_refuses_malformed_marker(tmp_path):
 root=make_marker_root(tmp_path,malformed=True)
 with pytest.raises(SystemExit,match='malformed'): matrix.verify_launch_authorization(root=root,environ=ENV,commit=GOOD_COMMIT)
def test_gate_refuses_missing_hash_field(tmp_path):
 root=make_marker_root(tmp_path,omit='executor_sha256')
 with pytest.raises(SystemExit,match='executor_sha256'): matrix.verify_launch_authorization(root=root,environ=ENV,commit=GOOD_COMMIT)
def test_gate_refuses_wrong_job_count(tmp_path):
 root=make_marker_root(tmp_path,jobs='180')
 with pytest.raises(SystemExit,match='207'): matrix.verify_launch_authorization(root=root,environ=ENV,commit=GOOD_COMMIT)
def test_cli_requires_explicit_flag():
 r=subprocess.run([sys.executable,str(ROOT/'scripts/run_cross_domain_matrix.py')],capture_output=True,text=True)
 assert r.returncode!=0

# --------------------------------------------------- parameter accounting
def test_optimizer_parameter_accounting_matches_real_continuation_optimizers():
 enc=ex.E2Encoder(); sim=ex.SimMIM2(enc)
 cm_opt=torch.optim.AdamW(sim.parameters(),lr=1e-3)
 assert ex.optimizer_trainable_parameters(cm_opt)==12975424
 proj=ex.ProjectionHead2(); tp=ex.TextProjection(); temp=ex.LearnableTemperature()
 clip_opt=torch.optim.AdamW(list(sim.parameters())+list(proj.parameters())+list(tp.parameters())+list(temp.parameters()),lr=1e-3)
 assert ex.optimizer_trainable_parameters(clip_opt)==13353025
 assert 13353025-12975424==377601
def test_optimizer_parameter_accounting_no_double_count():
 # torch itself rejects duplicate params across groups, so exercise the
 # helper's dedup logic against an optimizer-shaped object directly
 import types
 lin=torch.nn.Linear(4,3); w,b=lin.weight,lin.bias
 fake=types.SimpleNamespace(param_groups=[{'params':[w,b]},{'params':[w]}])
 assert ex.optimizer_trainable_parameters(fake)==w.numel()+b.numel()
 b.requires_grad_(False)
 assert ex.optimizer_trainable_parameters(fake)==w.numel()

# --------------------------------------------------------------- analysis
def _rec(target,arm,mode,frac,seed,f1):
 pc={c:{'precision':f1,'recall':f1,'f1':f1,'support':10} for c in ('normal','inner_race','outer_race')}
 return {'target':target,'arm':arm,'mode':mode,'label_fraction':frac,'seed':seed,'macro_f1':f1,'per_class':pc,'confusion_matrix':[[10,0,0],[0,10,0],[0,0,10]],'n_test_windows':30,'path':'synthetic'}
def test_analysis_aggregation_descriptive_and_equal_weight():
 recs=[]
 vals={'paderborn':0.9,'ottawa':0.6,'hust':0.3}
 for t,v in vals.items():
  for s in (42,43,44):
   recs.append(_rec(t,'CD-S1-CM','linear',100,s,v))
   recs.append(_rec(t,'CD-CLIP','linear',100,s,v+0.05))
   recs.append(_rec(t,'CD-S0','linear',100,s,v-0.1))
   recs.append(_rec(t,'CD-S1','linear',100,s,v-0.02))
 agg=analysis.aggregate(recs)
 e=agg['per_target']['ottawa']['CD-CLIP']['linear_label100']
 assert e['n_seeds']==3 and abs(e['mean_macro_f1']-0.65)<1e-12 and e['sd_macro_f1']==0.0
 rq2=agg['primary_comparisons']['RQ2']
 assert rq2['arms']==['CD-S1-CM','CD-CLIP']
 assert abs(rq2['per_target']['hust']['linear_label100']['mean_delta']-0.05)<1e-12
 ew=rq2['equal_weight_domain_mean_delta']['linear_label100']
 assert ew['complete_three_domains'] and abs(ew['value']-0.05)<1e-12
 mean_of_means=sum(vals.values())/3+0.05
 got=agg['equal_weight_domain_average']['CD-CLIP']['linear_label100']['equal_weight_mean_macro_f1']
 assert abs(got-mean_of_means)<1e-12
 assert 'FORBIDDEN' in agg['pooled_window_metric']
 rq1=agg['primary_comparisons']['RQ1']
 assert abs(rq1['per_target']['paderborn']['linear_label100']['mean_delta']-0.08)<1e-12
def test_analysis_paired_deltas_use_shared_seeds_only():
 recs=[_rec('hust','CD-S1-CM','linear',100,42,0.5),_rec('hust','CD-S1-CM','linear',100,43,0.6),_rec('hust','CD-CLIP','linear',100,42,0.7)]
 agg=analysis.aggregate(recs)
 e=agg['primary_comparisons']['RQ2']['per_target']['hust']['linear_label100']
 assert e['paired_seeds']==[42] and abs(e['mean_delta']-0.2)<1e-12 and e['sd_delta'] is None
def test_analysis_no_results_exits_cleanly(tmp_path):
 assert analysis.load_records(tmp_path)==[]

# ------------------------------------------------------- diagnostics spec
def test_diagnostics_spec_count_and_isolation():
 t=(ROOT/'configs/cross_domain/diagnostics_amendment_003.yaml').read_text()
 n_diag=t.count('- name:')
 arms=t.split('pretrained_arms: [')[1].split(']')[0].split(',')
 targets=t.split('targets: [')[1].split(']')[0].split(',')
 seeds=t.split('seeds: [')[1].split(']')[0].split(',')
 derived=n_diag*len(arms)*len(targets)*len(seeds)
 assert n_diag==5 and len(arms)==3 and derived==135
 assert 'planned_diagnostics: 135' in t
 assert t.count('sealed_test_access: false')==5 and 'sealed_test_access: true' not in t
 assert 'cd_s0_included: false' in t and 'CD-S0' not in t.split('pretrained_arms: [')[1].split(']')[0]
 assert 'post_hoc_descriptive_only' in t and 'checkpoint_selection' in t.split('feedback_forbidden')[1].split(']')[0]

# ------------------------------------------------- recovery / lock safety
def _job(tmp,jid='ssl__paderborn__CD-S1__seed42'):
 return {'job_id':jid,'job_type':'ssl_pretrain','target':'paderborn','arm':'CD-S1','seed':42,'mode':None,'label_fraction':None,'manifest_hashes':{},'checkpoint_dependency':None,'config_hash':'x','output_path':f'models/tmp_out/{jid}'}
def _stub_success(root,j):
 out=root/j['output_path']; out.mkdir(parents=True,exist_ok=True)
 p=out/'model.pt'; p.write_bytes(b'weights')
 (out/'COMPLETE.json').write_text(json.dumps({'target':j['target'],'arm':j['arm'],'seed':j['seed'],'selected_step':1,'manifest_hashes':j['manifest_hashes'],'artifacts':{'model.pt':hashlib.sha256(b'weights').hexdigest()}}))
def test_run_job_success_writes_durable_records(tmp_path,monkeypatch):
 j=_job(tmp_path)
 monkeypatch.setattr(matrix,'train_ssl',lambda job: _stub_success(tmp_path,job))
 assert matrix.run_job(j,root=tmp_path)=='COMPLETE'
 state=tmp_path/'logs/cross_domain_v1/state'
 done=json.loads((state/(j['job_id']+'.DONE.json')).read_text())
 assert done['state']=='COMPLETE' and done['hostname'] and done['start_utc'] and done['end_utc'] and done['output_artifacts']
 assert (tmp_path/'logs/cross_domain_v1'/(j['job_id']+'.log')).exists()
 assert not (state/(j['job_id']+'.RUNNING.json')).exists()
 assert matrix.run_job(j,root=tmp_path)=='SKIPPED'
def test_run_job_failure_writes_marker_and_blocks_retry(tmp_path,monkeypatch):
 j=_job(tmp_path,'ssl__ottawa__CD-S1__seed43')
 def boom(job): raise RuntimeError('synthetic failure')
 monkeypatch.setattr(matrix,'train_ssl',boom)
 with pytest.raises(RuntimeError): matrix.run_job(j,root=tmp_path)
 state=tmp_path/'logs/cross_domain_v1/state'
 f=json.loads((state/(j['job_id']+'.FAILED.json')).read_text())
 assert f['state']=='FAILED' and 'synthetic failure' in f['error']
 assert not (state/(j['job_id']+'.RUNNING.json')).exists()
 with pytest.raises(SystemExit,match='failure marker'): matrix.run_job(j,root=tmp_path)
def test_run_job_refuses_duplicate_running(tmp_path,monkeypatch):
 j=_job(tmp_path,'ssl__hust__CD-S1__seed44')
 state=tmp_path/'logs/cross_domain_v1/state'; state.mkdir(parents=True)
 (state/(j['job_id']+'.RUNNING.json')).write_text('{}')
 monkeypatch.setattr(matrix,'train_ssl',lambda job: _stub_success(tmp_path,job))
 with pytest.raises(SystemExit,match='RUNNING marker'): matrix.run_job(j,root=tmp_path)
def test_completed_skip_requires_hash_verification(tmp_path):
 j=_job(tmp_path,'ssl__paderborn__CD-S1__seed43')
 out=tmp_path/j['output_path']; out.mkdir(parents=True)
 (out/'model.pt').write_bytes(b'weights')
 (out/'COMPLETE.json').write_text(json.dumps({'artifacts':{'model.pt':'0'*64}}))
 with pytest.raises(SystemExit,match='hash verification'): matrix.run_job(j,root=tmp_path)

# ------------------------------------------------- DC centering (bugfix)
def _worst_known_offender_row():
 # H_2_0 window 8 (bearing UO-02): legacy float32-mean residual 1.775e-05,
 # the worst case found by the failure audit of 2026-08-10.
 for r in __import__('csv').DictReader((ROOT/'metadata/cross_domain_v1/amendment_003/fold_PADERBORN/ssl_source.csv').open()):
  if r['dataset']=='ottawa' and r['recording']=='H_2_0' and r['window_index']=='8': return r
 raise AssertionError('known offender row missing from manifest')
def test_dc_centering_regression_high_offset_ottawa():
 import numpy as np
 from src.vibrationclip.preprocessing import DC_TOLERANCE, dc_remove
 r=_worst_known_offender_row(); store=ex.RealBatchStore()
 sig=store.signal(r); start=round(int(r['original_start'])*24000/int(r['original_sample_rate']))
 sl=np.asarray(sig[start:start+24000],dtype=np.float64)
 # legacy float32-mean path violates the frozen tolerance on this window —
 # guards against regression to float32 mean accumulation
 legacy=abs(float(np.asarray(dc_remove(sl),dtype=np.float64).mean()))
 assert legacy>DC_TOLERANCE
 # fixed float64 centering meets the tolerance with a substantial margin
 fixed=abs(float((sl-sl.mean()).astype(np.float32).astype(np.float64).mean()))
 assert fixed<=DC_TOLERANCE and fixed<1e-6
 # the actual executor path (float64 centering before float32 cast) passes
 # the unchanged assert inside log_spectrogram end-to-end
 spec=store.one(r)
 assert spec.shape==(128,128) and spec.dtype==np.float32
def test_dc_audit_all_permitted_windows_zero_violations():
 audit_mod=_load('cddcaudit','scripts/audit_dc_residuals.py')
 out=audit_mod.audit(store=ex.RealBatchStore())
 assert out['tolerance']==1e-5
 assert out['n_windows']==5894 and out['n_recordings']==683   # 5897-3 phantom Paderborn windows removed by the Option B manifest fix
 assert out['violations']==0, out
 assert out['max_abs_mean']<1e-6, out
 assert set(out['per_dataset'])=={'cwru','hust','ottawa','paderborn'}

# ------------------------------------- dependency path resolution (bugfix)
def test_all_registry_dependencies_resolve_to_frozen_output_paths():
 js=[json.loads(x) for x in (ROOT/'metadata/cross_domain_v1/job_registry_amendment_003.jsonl').read_text().splitlines()]
 checked=0
 for j in js:
  if j['arm']=='CD-S0':
   assert j['checkpoint_dependency'] is None; continue
  if j['job_type']=='downstream': want=(j['target'],j['arm'],j['seed'])
  elif j['arm'] in ('CD-S1-CM','CD-CLIP'): want=(j['target'],'CD-S1',j['seed'])
  else: assert j['checkpoint_dependency'] is None; continue   # CD-S1 pretrain
  resolved=ex.registry_prerequisite(*want)
  assert resolved==j['checkpoint_dependency'], (j['job_id'],resolved)
  assert 'amendment_003/' in resolved   # versioned path preserved
  checked+=1
 assert checked==153   # 18 continuations + 135 non-S0 downstream
def test_dependency_resolution_uses_versioned_path_and_real_checkpoint():
 # the completed real CD-S1 paderborn seed42 artifact must resolve and verify
 resolved=ex.registry_prerequisite('paderborn','CD-S1',42)
 assert resolved=='models/cross_domain_v1/amendment_003/ssl/paderborn/CD-S1/seed_42'
 p,rec=ex.exact_dependency('paderborn','CD-S1',42)
 assert 'amendment_003' in str(p) and (p/'COMPLETE.json').exists()
 assert (rec['target'],rec['arm'],rec['seed'])==('paderborn','CD-S1',42)
 # the old unversioned reconstruction must play no role
 assert not (ROOT/'models/cross_domain_v1/ssl').exists()
def _fake_registry(tmp_path,out_rel,target='paderborn',arm='CD-S1',seed=42):
 reg=tmp_path/'registry.jsonl'
 reg.write_text(json.dumps({'job_type':'ssl_pretrain','target':target,'arm':arm,'seed':seed,'output_path':out_rel})+'\n')
 return reg
def test_dependency_rejections(tmp_path,monkeypatch):
 out=tmp_path/'ck'; out.mkdir()
 monkeypatch.setattr(ex,'ROOT',Path('/'))
 monkeypatch.setattr(ex,'REGISTRY_PATH',_fake_registry(tmp_path,str(out).lstrip('/')))
 # missing COMPLETE.json
 with pytest.raises(FileNotFoundError): ex.exact_dependency('paderborn','CD-S1',42)
 # identity mismatches: wrong target / arm / seed recorded in COMPLETE.json
 (out/'encoder.pt').write_bytes(b'weights')
 good_sha=hashlib.sha256(b'weights').hexdigest()
 for bad in ({'target':'ottawa','arm':'CD-S1','seed':42},{'target':'paderborn','arm':'CD-CLIP','seed':42},{'target':'paderborn','arm':'CD-S1','seed':43}):
  (out/'COMPLETE.json').write_text(json.dumps({**bad,'artifacts':{'encoder.pt':good_sha}}))
  with pytest.raises(ValueError,match='identity mismatch'): ex.exact_dependency('paderborn','CD-S1',42)
 # artifact hash mismatch
 (out/'COMPLETE.json').write_text(json.dumps({'target':'paderborn','arm':'CD-S1','seed':42,'artifacts':{'encoder.pt':'0'*64}}))
 with pytest.raises(ValueError,match='hash mismatch'): ex.exact_dependency('paderborn','CD-S1',42)
 # registry without exactly one prerequisite
 monkeypatch.setattr(ex,'REGISTRY_PATH',tmp_path/'empty.jsonl'); (tmp_path/'empty.jsonl').write_text('')
 with pytest.raises(ValueError,match='exactly one'): ex.exact_dependency('paderborn','CD-S1',42)

# --------------------------------------------------------- raw sha memo
def test_raw_sha_memoized_and_staleness_detected(tmp_path):
 p=tmp_path/'raw.bin'; p.write_bytes(b'aaa')
 h1=ex.raw_sha(p)
 assert h1==ex.sha(p)==hashlib.sha256(b'aaa').hexdigest()
 assert ex.raw_sha(p)==h1
 p.write_bytes(b'bbbb')
 assert ex.raw_sha(p)==hashlib.sha256(b'bbbb').hexdigest()
