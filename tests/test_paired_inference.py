import numpy as np
from src.paired_inference import holm_adjust,paired_summary

def test_paired_summary_counts_and_difference():
    x=np.array([2.,3.,4.,5.]); y=np.array([1.,3.,5.,2.]); s=paired_summary(x,y,n_boot=1000)
    assert (s["wins"],s["ties"],s["losses"])==(2,1,1)
    assert np.isclose(s["mean_difference"],.75)

def test_holm_is_monotone_in_sorted_order_and_bounded():
    raw=[.01,.04,.03,.4]; adj=holm_adjust(raw)
    order=np.argsort(raw); vals=np.asarray(adj)[order]
    assert np.all(np.diff(vals)>=-1e-12) and all(0<=x<=1 for x in adj)
