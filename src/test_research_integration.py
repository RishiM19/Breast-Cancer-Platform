"""Isolated integration fixtures. Real checkpoint inference is a separate verifier.

No database is opened. Stub outputs exercise failure/policy behavior, never
provide study predictions or performance measurements.
"""
import importlib.util
import io
import itertools
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, ImageOps
from pydantic import ValidationError

from src import research_inference as ri
from src.research_api import create_research_router


def png(mode="RGB"):
    image=Image.new(mode,(79,61),128 if mode=="L" else (210,90,20,140) if mode=="RGBA" else (210,90,20))
    b=io.BytesIO();image.save(b,format="PNG");return b.getvalue()


class StubModel(torch.nn.Module):
    def __init__(self,logits):
        super().__init__();self.values=torch.tensor(logits,dtype=torch.float32)
    def forward(self,x):return self.values.unsqueeze(0).repeat(len(x),1)


class StubEngine(ri.ResearchEngine):
    def __init__(self):
        super().__init__(device=torch.device("cpu"))
        self.outputs={k:list(range(len(ri.PROFILES[k].classes))) for k in ri.SPECS}
    def _checkpoint(self,k):
        s=ri.SPECS[k];return ri.Checkpoint(filename=ri.PROFILES[k].checkpoint,version=f"sha256:{s.expected_sha256}",
            expected_sha256=s.expected_sha256,sha256=s.expected_sha256,verification="verified")
    def _load(self,k):return StubModel(self.outputs[k])


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior_threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.db_guard=patch.object(sqlite3,"connect",side_effect=AssertionError("Database access forbidden in tests"))
        cls.db_guard.start()
    @classmethod
    def tearDownClass(cls):
        cls.db_guard.stop();torch.set_num_threads(cls.prior_threads)
    def setUp(self):self.engine=StubEngine()
    def client(self,engine=None):
        app=FastAPI();app.include_router(create_research_router(engine or self.engine));return TestClient(app)
    def response(self,domain="unknown"):
        return self.engine.analyze(png(),domain,attribution=False)

    def test_all_frozen_class_indices(self):
        expected={"busi":["Normal","Benign","Malignant"],"uclm":["Normal","Benign","Malignant"],
                  "bra":["Benign","Malignant"],"birads":["2","3","4","5"]}
        for k,v in expected.items():self.assertEqual(list(ri.PROFILES[k].classes),v)

    def test_preprocessing_independent_rgb_sizing_and_normalization(self):
        raw=png();rgb=ri.decode_image(raw)
        for profile in ri.PROFILES.values():
            display,tensor=ri.preprocess(io.BytesIO(raw),profile)
            size=(profile.input_size,)*2
            expected=(ImageOps.pad(rgb,size,method=Image.Resampling.BILINEAR,color=(0,0,0),centering=(.5,.5))
                if profile.preprocessing=="pad" else rgb.resize(size,Image.Resampling.BILINEAR))
            np.testing.assert_array_equal(display,np.asarray(expected))
            actual=(np.asarray(expected,dtype=np.float32).transpose(2,0,1)/255-np.array(ri.MEAN)[:,None,None])/np.array(ri.STD)[:,None,None]
            np.testing.assert_allclose(tensor[0].numpy(),actual,atol=5e-7,rtol=1e-6)

    def test_each_score_belongs_to_own_predicted_label(self):
        r=self.response()
        for model in r.models.values():
            self.assertEqual(model.score,model.probabilities[model.prediction]);self.assertEqual(model.confidence,model.score)
        self.assertIsNone(r.consensus.score);self.assertIsNone(r.consensus.confidence)

    def test_model_loading_occurs_outside_inference_mode_for_gradcam(self):
        original=self.engine._load
        def checked(k):
            self.assertFalse(torch.is_inference_mode_enabled())
            return original(k)
        with patch.object(self.engine,"_load",side_effect=checked):
            r=self.response()
        self.assertEqual(r.status,"ok")

    def test_production_engine_configures_verified_precision(self):
        torch.backends.cudnn.allow_tf32=True
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.benchmark=True
        torch.use_deterministic_algorithms(False)
        StubEngine()
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_native_cross_dataset_and_unknown_eligibility(self):
        for k,s in ri.SPECS.items():
            for d in ("unknown","BUSI","UCLM","BUS-BRA"):
                e=ri.eligibility(k,d)
                self.assertFalse(e.consensus_eligible)
                self.assertEqual(e.research_task_supported,d==s.source_domain and k!="busi")
                self.assertEqual(e.domain_relation,"unknown" if d=="unknown" else "native_research" if d==s.source_domain else "cross_dataset")

    def test_all_agreement_disagreement_and_ties_abstain_deterministically(self):
        results=self.response().models
        for labels in itertools.product(["Normal","Benign","Malignant"],repeat=3):
            # Deliberately untrusted/mutated metadata cannot unlock voting.
            untrusted={name:model.model_copy(update={"prediction":label,"eligibility":{"consensus_eligible":True}})
                       for (name,model),label in zip(results.items(),labels)}
            for domain in ("unknown","BUSI","UCLM","BUS-BRA"):
                a=ri.resolve_consensus(untrusted,domain);b=ri.resolve_consensus(dict(reversed(list(untrusted.items()))),domain)
                self.assertEqual(a,b);self.assertEqual(a.prediction,ri.ABSTENTION)
                self.assertEqual(a.eligible_model_ids,[]);self.assertIn("busi",a.excluded_models)

    def test_birads_never_a_pathology_vote_and_bra_not_normal_capable(self):
        r=self.response("BUS-BRA");self.assertEqual(r.models["BUS-BRA BI-RADS"].task,"birads")
        self.assertEqual(r.models["BUS-BRA"].ordered_class_labels,["Benign","Malignant"])
        self.assertIsNone(r.models["BUS-BRA"].probabilities["Normal"])
        self.assertEqual(r.consensus.participating_models,0)

    def test_not_run_has_no_fabricated_values_or_checkpoint_load(self):
        with patch.object(self.engine,"_load",side_effect=AssertionError("No model requested")):
            r=self.engine.analyze(png(),run_models=[],attribution=False)
        self.assertFalse(r.success);self.assertEqual(r.status,"unavailable")
        for m in r.models.values():self.assertEqual(m.status,"not_run");self.assertIsNone(m.score);self.assertEqual(m.probabilities,{})

    def test_missing_checkpoints_are_individual_missing_statuses(self):
        with tempfile.TemporaryDirectory() as temp:
            engine=ri.ResearchEngine(torch.device("cpu"),Path(temp));r=engine.analyze(png(),attribution=False)
        self.assertFalse(r.success)
        self.assertTrue(all(m.status=="missing" and m.prediction is None for m in r.models.values()))

    def test_checkpoint_hash_mismatch_never_loads_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/ri.PROFILES["busi"].checkpoint).write_bytes(b"invalid fixture checkpoint")
            engine=ri.ResearchEngine(torch.device("cpu"),root)
            with patch.object(engine,"_load",side_effect=AssertionError("Must not load mismatched weights")):
                r=engine.analyze(png(),run_models=["busi"],attribution=False)
        self.assertEqual(r.models["BUSI"].error_code,"checkpoint_hash_mismatch")
        self.assertEqual(r.models["BUSI"].checkpoint.verification,"mismatch")

    def test_nonfinite_model_output_isolated_without_clamping(self):
        self.engine.outputs["busi"]=[float("nan"),0,1]
        r=self.response();self.assertEqual(r.status,"partial");self.assertTrue(r.success)
        self.assertEqual(r.models["BUSI"].status,"error");self.assertIsNone(r.models["BUSI"].confidence)
        self.assertEqual(r.models["BUS-BRA"].status,"ok")

    def test_invalid_output_dimensions_are_error_status(self):
        self.engine.outputs["bra"]=[1,2,3]
        self.assertEqual(self.response().models["BUS-BRA"].status,"error")

    def test_model_loading_error_isolated(self):
        original=self.engine._load
        def fail(k):
            if k=="uclm":raise RuntimeError("fixture loading failure")
            return original(k)
        with patch.object(self.engine,"_load",side_effect=fail):r=self.response()
        self.assertEqual(r.models["UCLM"].status,"error");self.assertEqual(r.models["BUSI"].status,"ok")

    def test_schema_rejects_wrong_class_or_borrowed_score(self):
        m=self.response().models["BUSI"].model_dump()
        for change in ({"prediction":"Benign"},{"confidence":.01},{"score":.01},{"ordered_class_labels":["Benign","Malignant","Normal"]}):
            with self.assertRaises(ValidationError):ri.ModelResult.model_validate(m|change)

    def test_schema_requires_four_records(self):
        r=self.response().model_dump();r["models"].pop("BUS-BRA BI-RADS")
        with self.assertRaises(ValidationError):ri.ResearchResponse.model_validate(r)

    def test_bad_images_empty_truncated_and_text(self):
        for raw in (b"",b"not an image",png()[:25]):
            response=self.client().post("/predict",files={"file":("bad.png",raw,"image/png")})
            self.assertEqual(response.status_code,400)

    def test_image_modes_rgb_grayscale_rgba(self):
        for mode in ("RGB","L","RGBA"):
            self.assertEqual(ri.decode_image(png(mode)).mode,"RGB")

    def test_pixel_limit_and_multiple_frames_rejected(self):
        with patch.object(ri,"MAX_IMAGE_PIXELS",100):
            with self.assertRaises(ri.InvalidImage):ri.decode_image(png())
        b=io.BytesIO();Image.new("RGB",(10,10)).save(b,format="TIFF",save_all=True,append_images=[Image.new("RGB",(10,10))])
        with self.assertRaises(ri.InvalidImage):ri.decode_image(b.getvalue())

    def test_upload_byte_limit_413(self):
        with patch("src.research_api.MAX_UPLOAD_BYTES",20):
            r=self.client().post("/predict",files={"file":("large.png",png(),"image/png")})
        self.assertEqual(r.status_code,413)

    def test_file_only_post_backward_compatible_and_unknown_default(self):
        with patch.object(ri,"gradcam",side_effect=RuntimeError("Attribution unavailable fixture")):
            response=self.client().post("/predict",files={"file":("scan.png",png(),"image/png")})
        self.assertEqual(response.status_code,200);r=ri.ResearchResponse.model_validate(response.json())
        self.assertEqual(r.source_domain,"unknown");self.assertEqual(r.image.width,79)
        self.assertEqual(set(r.models),{"BUSI","UCLM","BUS-BRA","BUS-BRA BI-RADS"})

    def test_invalid_domain_422_and_missing_file_422(self):
        self.assertEqual(self.client().post("/predict",data={"source_domain":"claimed-native"},files={"file":("x.png",png())}).status_code,422)
        self.assertEqual(self.client().post("/predict").status_code,422)

    def test_contract_discovery_and_openapi_schema(self):
        c=self.client();descriptor=c.get("/inference-contract").json()
        self.assertEqual(descriptor["contract_version"],"1.0.0")
        self.assertIn("ResearchResponse",c.get("/openapi.json").json()["components"]["schemas"])
        self.assertTrue(all(not s["consensus_eligible"] for s in descriptor["models"].values()))

    def test_attribution_failure_has_no_synthetic_fallback_or_diagnostic_metrics(self):
        with patch.object(ri,"gradcam",side_effect=RuntimeError("No gradient fixture")):
            r=self.engine.analyze(png())
        self.assertEqual(r.attribution.status,"error");self.assertEqual(r.heatmap_base64,"")
        self.assertNotIn("size_cm",r.model_dump());self.assertNotIn("stage",r.model_dump())
        self.assertNotIn("biopsy",json.dumps(r.report_details).lower())

    def test_main_import_mounts_actual_router_without_database_access(self):
        spec=importlib.util.spec_from_file_location("integration_main_isolated",ri.ROOT/"main.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertIn(module.init_db,module.app.router.on_startup)
        paths={r.path for r in module.app.routes}
        self.assertTrue({"/predict","/inference-contract","/patients","/patients/{patient_id}","/feedback/submit","/feedback/stats"}<=paths)
        self.assertEqual(TestClient(module.app).get("/inference-contract").status_code,200)


if __name__=="__main__":unittest.main(verbosity=2)
