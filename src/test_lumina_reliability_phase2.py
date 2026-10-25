"""Isolated Phase 2 safety and numerical tests; fixtures are not study outcomes."""
import unittest
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
from lumina_reliability_phase2 import (
    matched_controls,control_candidates,match_details,apply_fill,select_new_cases,
    transform_image,PROFILES,MEAN,STD,
)


class Phase2Tests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42)
        self.rgb=rng.integers(0,256,(60,80,3),dtype=np.uint8)
        self.lesion=np.zeros((60,80),bool);self.lesion[20:31,31:44]=True
        self.text=np.zeros_like(self.lesion);self.text[48:54,12:30]=True
        self.roi=np.zeros_like(self.lesion);self.roi[7:13,10:18]=True

    def controls(self):
        return matched_controls(self.rgb,self.lesion,self.roi,self.text,"fixture","lesion")[0]

    def test_control_area_shape_and_exclusions(self):
        ys,xs=np.nonzero(self.roi);template=self.roi[ys.min():ys.max()+1,xs.min():xs.max()+1]
        for c in self.controls():
            mask=c['mask'];self.assertEqual(mask.sum(),self.roi.sum())
            self.assertFalse((mask&(self.lesion|self.text|self.roi)).any())
            self.assertTrue(c['shape_exact'])
            y,x=np.nonzero(mask);np.testing.assert_array_equal(mask[y.min():y.max()+1,x.min():x.max()+1],template)

    def test_multiple_seeds_reproducible_distinct_when_feasible(self):
        a,b=self.controls(),self.controls()
        self.assertEqual([c['seed'] for c in a],[42,2026,3407])
        self.assertTrue(all(np.array_equal(x['mask'],y['mask']) for x,y in zip(a,b)))
        self.assertEqual(len({c['mask'].tobytes() for c in a}),3)

    def test_no_control_invented_when_disjoint_area_insufficient(self):
        huge=np.ones((60,80),bool);huge[:2]=False
        controls,error=matched_controls(self.rgb,self.lesion,huge,self.text,'fixture','large')
        self.assertEqual(controls,[]);self.assertIn('Insufficient',error)

    def test_shape_fallback_is_explicit_and_area_exact(self):
        # An isolated pair of far-spaced native pixels cannot translate to the
        # deliberately tiny allowed donor set; fallback must disclose shape loss.
        rgb=np.zeros((20,20,3),np.uint8);lesion=np.zeros((20,20),bool)
        roi=np.zeros((20,20),bool);roi[0,0]=True;roi[19,19]=True
        text=np.ones((20,20),bool);text[0,0]=False;text[19,19]=False;text[8:11,8:11]=False
        controls,error=matched_controls(rgb,lesion,roi,text,'fixture','shape')
        self.assertIsNone(error);self.assertEqual(len(controls),3)
        self.assertTrue(all(not c['shape_exact'] for c in controls))
        self.assertTrue(all(c['mask'].sum()==2 for c in controls))

    def test_fill_range_dtype_geometry_and_locality(self):
        for fill,strengths in [('blur',[.0125,.025]),('local_median',[.25,.5,1.])]:
            for strength in strengths:
                changed,_=apply_fill(self.rgb,self.roi,self.lesion,self.text,fill,strength)
                self.assertEqual(changed.dtype,np.uint8);self.assertEqual(changed.shape,self.rgb.shape)
                np.testing.assert_array_equal(changed[~self.roi],self.rgb[~self.roi])
                self.assertGreaterEqual(changed.min(),0);self.assertLessEqual(changed.max(),255)

    def test_median_blend_has_no_uint8_arithmetic_overflow(self):
        rgb=np.full((30,30,3),255,np.uint8);mask=np.zeros((30,30),bool);mask[10:20,10:20]=True
        rgb[mask]=0
        changed,_=apply_fill(rgb,mask,np.zeros_like(mask),np.zeros_like(mask),'local_median',.5)
        self.assertTrue(np.all(changed[mask]==128))

    def test_pixel_channel_order_and_normalization_independently(self):
        rgb=np.empty((60,80,3),np.uint8);rgb[...,0]=255;rgb[...,1]=128;rgb[...,2]=0
        for profile in PROFILES.values():
            actual=transform_image(rgb,profile).numpy()
            image=Image.fromarray(rgb);size=(profile.input_size,)*2
            image=(ImageOps.pad(image,size,method=Image.Resampling.BILINEAR,color=(0,0,0),centering=(.5,.5))
                if profile.preprocessing=='pad' else image.resize(size,Image.Resampling.BILINEAR))
            expected=(np.asarray(image,dtype=np.float32).transpose(2,0,1)/255-np.array(MEAN)[:,None,None])/np.array(STD)[:,None,None]
            np.testing.assert_allclose(actual,expected,atol=5e-7,rtol=1e-6)
            self.assertEqual(actual.shape,(3,profile.input_size,profile.input_size))

    def test_matching_statistics_exact_zero_for_same_descriptor(self):
        d=dict(cy=10.,cx=12.,lesion_distance=.2,gray_mean=.4,gray_sd=.1)
        m=match_details(d,d,(60,80));self.assertEqual(m['matching_cost'],0);self.assertTrue(m['all_soft_matched'])

    def test_sampling_exclusion_scarce_category_and_reproducibility(self):
        rows=[]
        for label,n in [(2,30),(3,20),(4,40),(5,3)]:
            for i in range(n):
                for side in ['l','r']:rows.append(dict(case_id=f'{label}_{i}',birads=label,image_id=f'{label}_{i}_{side}'))
        frame=pd.DataFrame(rows);pilot=pd.DataFrame(dict(case_id=['2_0','3_0','4_0','5_0']))
        a,_,q=select_new_cases(frame,pilot,60,42);b,_,_=select_new_cases(frame,pilot,60,42)
        pd.testing.assert_frame_equal(a,b);self.assertEqual(len(a),60);self.assertEqual(a.case_id.nunique(),60)
        self.assertFalse(set(a.case_id)&set(pilot.case_id));self.assertEqual(q[5],2)

    def test_corrected_busi_mapping_kept(self):
        self.assertEqual(PROFILES['busi'].classes,('Normal','Benign','Malignant'))

    def test_holm_family_adjustment_known_values(self):
        from report_lumina_reliability_phase2 import holm
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
        np.testing.assert_allclose(holm([1.,1.,1.]),[1.,1.,1.])

    def test_bootstrap_respects_small_support_and_constant_effect(self):
        from report_lumina_reliability_phase2 import case_interval
        self.assertEqual(case_interval([.1]*4),(None,None))
        lo,hi=case_interval([.02]*10)
        self.assertAlmostEqual(lo,.02);self.assertAlmostEqual(hi,.02)

    def test_sign_randomization_reproducible_and_zero_effect_not_positive(self):
        from report_lumina_reliability_phase2 import sign_randomization
        self.assertEqual(sign_randomization([0.]*20),1.)
        a=sign_randomization([.02]*20);b=sign_randomization([.02]*20)
        self.assertEqual(a,b);self.assertLess(a,.01)


if __name__=='__main__':unittest.main(verbosity=2)
