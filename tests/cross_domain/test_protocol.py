from src.cross_domain.protocol import balanced_roster, assert_integrity, stable_rows_hash

def row(ds,b,r,s,e,split='ssl'):
 return dict(dataset=ds,physical_bearing=b,recording=r,operating_condition='',native_label='N',harmonised_label='normal',window_index='0',original_start=str(s),original_end=str(e),split=split,seed='42',fold='A')
def test_balanced_reproducible_and_domain_balanced():
 rows=[row('a','a1','r1',0,10),row('a','a2','r2',0,10),row('b','b1','r3',0,10)]
 x=balanced_roster(rows,12,42); y=balanced_roster(rows,12,42)
 assert stable_rows_hash(x)==stable_rows_hash(y)
 assert sum(r['dataset']=='a' for r in x)==6
def test_target_and_partition_leakage_detected():
 ssl=[row('target','s','rs',0,10)]
 parts={'target':'target','train':[row('target','x','r',0,10,'train')], 'val':[row('target','x','r',10,20,'val')], 'test':[]}
 e=assert_integrity(ssl,parts); assert 'dataset leakage' in e and 'bearing leakage' in e and 'recording leakage' in e
def test_arm_and_label_identity():
 a=[row('a','b','r',0,10)]; b=[row('a','b','r',10,20)]
 e=assert_integrity([],{'target':'z','train':[],'val':[],'test':[]},{'CD-S1':a,'CD-CLIP':b},{'x':a,'y':b})
 assert 'arm mismatch' in e and 'label mismatch' in e
