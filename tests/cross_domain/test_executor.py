import csv, importlib.util, json
from pathlib import Path
import pytest, torch
from src.cross_domain import executor as ex
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('cdmatrix',ROOT/'scripts/run_cross_domain_matrix.py'); matrix=importlib.util.module_from_spec(spec); spec.loader.exec_module(matrix)
def read(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def test_exactly_207_unique_jobs_and_outputs():
 js=matrix.jobs(); assert len(js)==len({j['job_id'] for j in js})==len({j['output_path'] for j in js})==207
def test_matrix_breakdown():
 js=matrix.jobs(); assert sum(j['job_type']!='downstream' for j in js)==27
 assert sum(j.get('mode')=='linear' for j in js)==36 and sum(j.get('mode')=='low_label' for j in js)==108 and sum(j.get('mode')=='finetune' for j in js)==36
@pytest.mark.parametrize('j',[j for j in matrix.jobs() if j['job_type']!='downstream'])
def test_target_absent_both_source_roles(j):
 assert all(r['dataset']!=j['target'] for r in read(ROOT/j['source_train'])+read(ROOT/j['source_val']))
def test_grouped_source_roles_disjoint_and_matched_arms():
 js=matrix.jobs()
 for target in ex.TARGETS:
  for seed in ex.SEEDS:
   pair=[j for j in js if j['job_type']!='downstream' and j['target']==target and j['seed']==seed]
   assert len(pair)==3 and len({j['source_train'] for j in pair})==len({j['source_val'] for j in pair})==1
   a=read(ROOT/pair[0]['source_train']); b=read(ROOT/pair[0]['source_val'])
   assert not {(r['dataset'],r['physical_bearing']) for r in a}&{(r['dataset'],r['physical_bearing']) for r in b}
def test_clip_is_label_informed_not_canonical_s6p():
 t=(ROOT/'configs/cross_domain/arms.yaml').read_text(); assert 'objective_equivalent_to_canonical_S6p: false' in t and 'source_class_labels_select_text: true' in t
def test_s0_cannot_load_ssl():
 with pytest.raises(ValueError): ex.configure_probe('CD-S0','linear',ex.E2Encoder().state_dict())
def test_frozen_and_finetune_policies():
 m=ex.configure_probe('CD-S1','linear'); assert not any(p.requires_grad for p in m.encoder.parameters()) and all(p.requires_grad for p in m.head.parameters())
 m=ex.configure_probe('CD-S1','finetune'); assert all(p.requires_grad for p in m.encoder.parameters()) and all(p.requires_grad for p in m.head.parameters())
def test_architecture_equivalence():
 a=ex.E2Encoder(); b=ex.SimMIM2().encoder
 assert list(a.state_dict())==list(b.state_dict()) and {k:tuple(v.shape) for k,v in a.state_dict().items()}=={k:tuple(v.shape) for k,v in b.state_dict().items()}
 assert sum(p.numel() for p in a.parameters())==12449856
def test_prompt_complete_identity_free_and_exclusions_refuse():
 assert set(ex.PROTOTYPES)==set(ex.CLASS_INDEX)
 assert not any(x in ' '.join(ex.PROTOTYPES.values()).lower() for x in ('cwru','paderborn','ottawa','hust'))
 ts=ex.TextStore()
 with pytest.raises(ValueError): ts.batch([{'harmonised_label':'EXCLUDE'}],torch.device('cpu'))
def test_fraction_manifest_exact_and_shared():
 js=matrix.jobs()
 for target in ex.TARGETS:
  for seed in ex.SEEDS:
   for frac in (5,10,25):
    q=[j for j in js if j.get('mode')=='low_label' and j['target']==target and j['seed']==seed and j['label_fraction']==frac]
    assert len(q)==4 and len({j['train_manifest'] for j in q})==1
def test_dependency_exact_no_substitution():
 with pytest.raises(FileNotFoundError): ex.exact_dependency('ottawa','CD-CLIP',42)
 with pytest.raises(ValueError): ex.registry_prerequisite('ottawa','CD-S0',42)
def test_config_hash_reproducible():
 assert [j['config_hash'] for j in matrix.jobs()]==[j['config_hash'] for j in matrix.jobs()]
def test_sealed_test_training_guard():
 with pytest.raises(ex.SealedTestAccessError): ex.read_manifest(ROOT/'metadata/cross_domain_v1/amendment_003/fold_HUST/target_test_sealed.csv',training=True)
def test_training_jobs_never_use_test_as_train_or_val():
 for j in matrix.jobs():
  assert 'target_test_sealed' not in str(j.get('train_manifest')) and 'target_test_sealed' not in str(j.get('val_manifest'))
def test_source_balancing_deterministic_and_domains_equal():
 j=next(x for x in matrix.jobs() if x['job_type']!='downstream'); r=read(ROOT/j['source_train']); a=ex.balanced_indices(r,64,42); b=ex.balanced_indices(r,64,42); assert a==b
 counts={d:sum(r[i]['dataset']==d for i in a) for d in {x['dataset'] for x in r}}; assert max(counts.values())-min(counts.values())<=1
 big=ex.balanced_indices(r,120000,43); exposure={}
 for i in big: exposure[(r[i]['dataset'],r[i]['physical_bearing'])]=exposure.get((r[i]['dataset'],r[i]['physical_bearing']),0)+1
 assert set(exposure)=={(x['dataset'],x['physical_bearing']) for x in r}
 for ds in {x['dataset'] for x in r}:
  vals=[v for (d,_),v in exposure.items() if d==ds]; assert max(vals)/min(vals)<1.25
def test_compute_match_contract():
 js=matrix.jobs()
 for target in ex.TARGETS:
  for seed in ex.SEEDS:
   a=next(j for j in js if j['target']==target and j['seed']==seed and j['arm']=='CD-S1-CM' and j['job_type']!='downstream'); b=next(j for j in js if j['target']==target and j['seed']==seed and j['arm']=='CD-CLIP' and j['job_type']!='downstream')
   assert (a['checkpoint_dependency'],a['source_train'],a['source_val'])==(b['checkpoint_dependency'],b['source_train'],b['source_val'])
   assert ex.SSL['clip_steps']==6000 and ex.SSL['batch']==64
def test_no_batchnorm_or_dropout():
 m=ex.E2Encoder(); assert not any(isinstance(x,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout)) for x in m.modules())
def test_cache_equivalence(tmp_path):
 j=next(x for x in matrix.jobs() if x['job_type']!='downstream'); row=read(ROOT/j['source_train'])[0]
 uncached=ex.RealBatchStore(use_cache=False).one(row); store=ex.RealBatchStore(use_cache=True,cache_root=tmp_path); first=store.one(row); second=store.one(row)
 import numpy as np
 np.testing.assert_allclose(uncached,first,rtol=0,atol=1e-7); np.testing.assert_array_equal(first,second); assert list(tmp_path.glob('*.json')) and not list(tmp_path.glob('*.lock'))
