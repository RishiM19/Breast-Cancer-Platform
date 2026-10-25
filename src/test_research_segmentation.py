"""Geometry/policy/failure fixtures, not medical performance measurements."""
import base64
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from . import research_segmentation as s
from .research_inference import ResearchEngine, validate_research_response, decode_image, InvalidImage
from .test_research_integration import StubEngine, png


class SegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)
    def fixture(self, size=(79,61), logits=None):
        if logits is None:
            logits=torch.zeros((1,2,256,256));logits[:,1,:80,:110]=4;logits[:,1,220:,210:]=2
        return s.SegmentationResult(requested_domain='UCLM',domain_relation='native_research',status='ok',
            image_width=size[0],image_height=size[1],checkpoint_sha256=s.SHA256,checkpoint_verification='verified',
            reason='Explicit synthetic geometry fixture',**s.encoded_outputs(logits,size))
    def test_native_sizes_and_binary_nearest(self):
        for size in [(19,37),(856,606),(1400,220),(256,256),(1,1),(1,513),(513,1)]:
            r=self.fixture(size);m=s.decode_png(r.predicted_mask,'L',size)
            self.assertEqual(m.shape,(size[1],size[0]));self.assertTrue(np.isin(m,[0,255]).all())
    def test_asymmetric_landmarks_no_rotation_mirror_or_padding(self):
        r=self.fixture((512,768));m=s.decode_png(r.predicted_mask,'L',(512,768))
        self.assertEqual(m[0,0],255);self.assertEqual(m[0,-1],0);self.assertEqual(m[-1,0],0);self.assertEqual(m[-1,-1],255)
        # Integer-scale exact pixel center restoration, independently using repeat.
        source=np.zeros((256,256),np.uint8);source[:80,:110]=255;source[220:,210:]=255
        np.testing.assert_array_equal(m,np.repeat(np.repeat(source,3,axis=0),2,axis=1))
    def test_noninteger_resize_pixel_center_convention(self):
        size=(79,61);m=s.decode_png(self.fixture(size).predicted_mask,'L',size)
        x=np.floor((np.arange(size[0])+.5)*256/size[0]).astype(int)
        y=np.floor((np.arange(size[1])+.5)*256/size[1]).astype(int)
        source=np.zeros((256,256),np.uint8);source[:80,:110]=255;source[220:,210:]=255
        np.testing.assert_array_equal(m,source[y[:,None],x[None,:]])
    def test_overlay_alpha_and_color_matches_foreground(self):
        r=self.fixture();m=s.decode_png(r.predicted_mask,'L',(79,61));a=s.decode_png(r.overlay,'RGBA',(79,61))
        np.testing.assert_array_equal(a[m==0],np.zeros((int((m==0).sum()),4),dtype=np.uint8))
        self.assertTrue((a[m>0]==[0,255,128,112]).all())
    def test_empty_genuine_output_and_transparency(self):
        r=self.fixture(logits=torch.zeros((1,2,256,256)))
        self.assertTrue(r.empty_mask);self.assertFalse(s.decode_png(r.overlay,'RGBA',(79,61)).any())
    def test_threshold_strict_ties_background(self):
        logits=torch.zeros((1,2,256,256));logits[:,1,0,0]=1e-7;logits[:,1,0,1]=-1e-7
        r=self.fixture((256,256),logits);m=s.decode_png(r.predicted_mask,'L',(256,256))
        self.assertEqual(m[0,0],255);self.assertEqual(m[0,1],0);self.assertEqual(m[1,1],0)
    def test_probability_map_float32_lossless(self):
        r=self.fixture();p=np.load(io.BytesIO(base64.b64decode(r.foreground_probability_npy_base64)),allow_pickle=False)
        self.assertEqual(p.shape,(256,256));self.assertEqual(p.dtype,np.float32);self.assertEqual(p[100,100],.5)
    def test_grayscale_and_rgb_tensor_parity(self):
        im=Image.new('L',(43,77),90)
        self.assertTrue(torch.equal(s.preprocess_image(im),s.preprocess_image(im.convert('RGB'))))
    def test_rgba_converts_without_geometric_transform(self):
        im=Image.new('RGBA',(53,31),(23,87,190,30));rgb=im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
        expected=((np.asarray(rgb,dtype=np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1).copy()
        np.testing.assert_array_equal(s.preprocess_image(im)[0].numpy(),expected)
    def test_nonfinite_and_wrong_shape_rejected(self):
        for bad in [torch.ones(1,3,256,256),torch.full((1,2,256,256),float('nan')),torch.full((1,2,256,256),float('inf'))]:
            with self.assertRaises(ValueError):s.encoded_outputs(bad,(79,61))
    def test_invalid_binary_input_rejected(self):
        with self.assertRaises(ValueError):s.restore_mask(np.full((256,256),2),(79,61))
    def test_ineligible_domains_never_read_checkpoint(self):
        e=s.SegmentationEngine(torch.device('cpu'),Path('DOES_NOT_EXIST'))
        with patch.object(Path,'is_file',side_effect=AssertionError('Must not inspect weights')):
            for domain in ['unknown','BUSI','BUS-BRA']:
                r=e.analyze(Image.new('RGB',(23,17)),domain);self.assertEqual(r.status,'not_run');self.assertIsNone(r.predicted_mask)
        self.assertEqual(e.load_count,0)
    def test_missing_checkpoint_no_fallback(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'data/processed/lumina_integration_stage3') as temp:
            r=s.SegmentationEngine(torch.device('cpu'),Path(temp)/'absent.pth').analyze(Image.new('RGB',(23,17)),'UCLM')
        self.assertEqual(r.status,'missing');self.assertIsNone(r.overlay);self.assertIsNone(r.foreground_probability_npy_base64)
    def test_nondefault_exif_orientation_declined_before_weights(self):
        e=s.SegmentationEngine(torch.device('cpu'),Path('MUST_NOT_BE_READ'))
        for orientation in range(2,9):
            original=Image.new('RGB',(79,61));exif=original.getexif();exif[274]=orientation
            buf=io.BytesIO();original.save(buf,format='JPEG',exif=exif)
            decoded=decode_image(buf.getvalue())
            self.assertEqual(decoded.getexif().get(274),orientation)
            with patch.object(Path,'is_file',side_effect=AssertionError('Must not inspect weights')):
                r=e.analyze(decoded,'UCLM')
            self.assertEqual(r.status,'not_run');self.assertEqual(r.error_code,'unsupported_image_orientation');self.assertIsNone(r.overlay)
    def test_wrong_checkpoint_hash_no_load(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'data/processed/lumina_integration_stage3') as temp:
            path=Path(temp)/'bad.pth';path.write_bytes(b'synthetic invalid checkpoint')
            e=s.SegmentationEngine(torch.device('cpu'),path)
            with patch.object(e,'_load',side_effect=AssertionError('Hash must reject')):r=e.analyze(Image.new('RGB',(23,17)),'UCLM')
        self.assertEqual(r.error_code,'checkpoint_hash_mismatch');self.assertIsNone(r.predicted_mask)
    def test_inference_failure_no_fallback(self):
        e=s.SegmentationEngine(torch.device('cpu'))
        with patch.object(e,'_load',side_effect=RuntimeError('Explicit failure fixture')):
            r=e.analyze(Image.new('RGB',(23,17)),'UCLM')
        self.assertEqual(r.status,'error');self.assertIsNone(r.overlay)
    def test_schema_rejects_mask_overlay_mismatch(self):
        raw=self.fixture().model_dump();raw['overlay']=s.png_url(Image.new('RGBA',(79,61)))
        with self.assertRaises(ValueError):s.SegmentationResult.model_validate(raw)
    def test_schema_rejects_wrong_native_dimensions(self):
        raw=self.fixture().model_dump();raw['image_width']+=1
        with self.assertRaises(ValueError):s.SegmentationResult.model_validate(raw)
    def test_schema_rejects_fabricated_failure_outputs(self):
        raw=self.fixture().model_dump();raw['status']='error'
        with self.assertRaises(ValueError):s.SegmentationResult.model_validate(raw)
    def test_version_10_unchanged_no_segmentation(self):
        r=StubEngine().analyze(png(),attribution=False);self.assertEqual(r.contract_version,'1.0.0');self.assertNotIn('segmentation',r.model_dump())
        validate_research_response(r.model_dump())
    def test_version_11_non_uclm_preserves_classifiers_and_consensus(self):
        e=StubEngine();a=e.analyze(png(),attribution=False);b=e.analyze(png(),attribution=False,contract_version='1.1.0')
        self.assertEqual(a.models,b.models);self.assertEqual(a.consensus,b.consensus);self.assertEqual(b.segmentation.status,'not_run')
        validate_research_response(b.model_dump())
    def test_version_11_domain_and_image_mismatch_rejected(self):
        r=StubEngine().analyze(png(),attribution=False,contract_version='1.1.0').model_dump()
        for change in [dict(image_width=2),dict(requested_domain='BUSI',domain_relation='cross_dataset')]:
            raw=copy.deepcopy(r);raw['segmentation'].update(change)
            with self.assertRaises(ValueError):validate_research_response(raw)
    def test_unknown_contract_rejected(self):
        with self.assertRaises(ValueError):validate_research_response({'contract_version':'1.2.0'})
    def test_invalid_image_formats_rejected(self):
        for raw in [b'',b'not PNG',b'\x89PNG\r\n\x1a\n']:
            with self.assertRaises(InvalidImage):decode_image(raw)
        buf=io.BytesIO();Image.new('RGB',(8,8)).save(buf,format='GIF')
        with self.assertRaises(InvalidImage):decode_image(buf.getvalue())


if __name__=='__main__':unittest.main(verbosity=2)
