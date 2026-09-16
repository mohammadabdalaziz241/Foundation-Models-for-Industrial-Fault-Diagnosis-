import numpy as np
from src.multisource_batches import BalancedSourceBatcher

def test_three_sources_are_equal_in_every_batch_and_small_pool_cycles():
    pools=[np.zeros((5,1,8),np.float32),np.ones((13,1,8),np.float32),np.ones((29,1,8),np.float32)*2]
    batches=list(BalancedSourceBatcher(pools,batch_size=12,seed=3))
    assert len(batches)==8
    for X,d in batches:
        assert X.shape==(12,1,8)
        assert np.array_equal(np.bincount(d.numpy(),minlength=3),[4,4,4])

def test_batcher_is_deterministic():
    pools=[np.arange(40,dtype=np.float32).reshape(5,1,8),np.arange(56,dtype=np.float32).reshape(7,1,8)]
    a=list(BalancedSourceBatcher(pools,4,9)); b=list(BalancedSourceBatcher(pools,4,9))
    assert all(np.array_equal(x[0],y[0]) and np.array_equal(x[1],y[1]) for x,y in zip(a,b))
