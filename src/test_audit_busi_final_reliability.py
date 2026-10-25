"""Isolated mathematical/grouping/policy fixtures, never dataset predictions."""
import unittest
import numpy as np
import pandas as pd
from audit_busi_final_reliability import (
    ROOT,OUT,DataGuard,ssim,UnionFind,make_groups,metric_record,wilson,
    CONFIRMED,STRONG,UNCERTAIN,
    run_immediate_coroutine,
)


class BUSIAuditTests(unittest.TestCase):
    def test_ssim_identity_symmetry_and_degradation(self):
        a=np.random.default_rng(42).random((128,128))
        self.assertAlmostEqual(ssim(a,a),1.,places=12)
        self.assertAlmostEqual(ssim(a,1-a),ssim(1-a,a),places=12)
        self.assertLess(ssim(a,1-a),0)
        self.assertLess(ssim(a,np.ones_like(a)*.5),.1)

    def test_ssim_luminance_known_constant_formula(self):
        a=np.ones((128,128))*.25;b=np.ones_like(a)*.5
        expected=(2*.25*.5+.01**2)/(.25**2+.5**2+.01**2)
        self.assertAlmostEqual(ssim(a,b),expected,places=10)

    def test_ssim_rejects_bad_geometry(self):
        with self.assertRaises(ValueError):ssim(np.zeros((8,8)),np.zeros((8,8)))
        with self.assertRaises(ValueError):ssim(np.zeros((32,32)),np.zeros((32,33)))

    def test_union_find_transitive_and_order_independent(self):
        a=UnionFind(5);b=UnionFind(5)
        for x,y in [(2,3),(0,1),(1,2)]:a.union(x,y)
        for x,y in [(2,1),(1,0),(3,2)]:b.union(x,y)
        self.assertEqual([a.find(i) for i in range(5)],[b.find(i) for i in range(5)])

    def fixture(self):
        items=pd.DataFrame(dict(image_index=range(4),split=["train","train","val","val"],
            label=[0,0,1,2],label_name=["Normal","Normal","Benign","Malignant"],path=list("abcd")))
        edges=pd.DataFrame(dict(a_index=[0,1,2],b_index=[1,2,3],similarity_classification=[CONFIRMED,STRONG,UNCERTAIN]))
        return items,edges

    def test_conservative_groups_exclude_uncertain_edges(self):
        items,edges=self.fixture();g=make_groups(items,edges,{CONFIRMED,STRONG})
        self.assertEqual(set(g.image_index),{0,1,2});self.assertEqual(g.group_id.nunique(),1)
        self.assertTrue(g.cross_split_group.all());self.assertTrue(g.conflicting_label_group.all())
        self.assertTrue(g.identity_scope.str.contains("NOT verified").all())
        lower=make_groups(items,edges,{CONFIRMED});self.assertEqual(set(lower.image_index),{0,1})
        self.assertFalse(lower.cross_split_group.any())

    def test_disconnected_groups_and_empty_edges(self):
        items,edges=self.fixture();g=make_groups(items,edges.iloc[:0],{CONFIRMED})
        self.assertTrue(g.empty);self.assertIn("group_id",g.columns)

    def test_metrics_use_fixed_label_order_and_actual_predictions(self):
        frame=pd.DataFrame(dict(label=[0,0,1,2,2],predicted_index=[0,1,1,2,1]))
        r=metric_record("fixture",frame)
        self.assertEqual(r["correct"],3);self.assertEqual(r["n"],5)
        self.assertAlmostEqual(r["accuracy"],.6);self.assertAlmostEqual(r["malignant_recall"],.5)
        self.assertEqual(r["malignant_support"],2)

    def test_missing_class_recall_unestimated_not_false_zero(self):
        frame=pd.DataFrame(dict(label=[1,1],predicted_index=[1,1]));r=metric_record("fixture",frame)
        self.assertIsNone(r["malignant_recall"]);self.assertEqual(r["balanced_accuracy"],1)

    def test_wilson_extreme_and_empty_support(self):
        self.assertEqual(wilson(0,0),(None,None))
        lo,hi=wilson(10,10);self.assertLess(lo,1);self.assertAlmostEqual(hi,1)
        lo,hi=wilson(0,10);self.assertAlmostEqual(lo,0);self.assertGreater(hi,0)

    def test_guard_blocks_unlisted_data_and_source_writes(self):
        guard=DataGuard(OUT)
        with self.assertRaises(PermissionError):guard("open",(str(ROOT/"data/unlisted_fixture.png"),"r",0))
        with self.assertRaises(PermissionError):guard("open",(str(ROOT/"data/processed/val.csv"),"w",0))
        guard("open",(str(ROOT/"data/processed/val.csv"),"r",0))

    def test_in_memory_endpoint_coroutine_needs_no_network_event_loop(self):
        async def upload():return b"fixture"
        async def endpoint():return await upload()
        self.assertEqual(run_immediate_coroutine(endpoint()),b"fixture")


if __name__=="__main__":unittest.main(verbosity=2)
