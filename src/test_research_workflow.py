"""Actual FastAPI startup/SQL/review tests on newly created synthetic-only DBs."""
import copy
import importlib.util
import json
import os
import sqlite3
import unittest
import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from src.research_api import create_research_router
from src.research_inference import ROOT, ABSTENTION
from src.research_storage import database_path, frozen_analysis
from src.test_research_integration import StubEngine, png

OUT=Path(os.environ.get("LUMINA_TEST_ARTIFACT_ROOT",ROOT/"data/processed/lumina_integration_stage2")).resolve()
if OUT not in {(ROOT/f"data/processed/lumina_integration_stage{s}").resolve() for s in (2,3)} | {(ROOT/'data/processed/lumina_ui_verdict_upgrade').resolve()}:
    raise ValueError("Tests require a dedicated integration artifact directory")
ISOLATION_EVENTS=[]


def load_app():
    spec=importlib.util.spec_from_file_location("workflow_main_"+uuid.uuid4().hex,ROOT/"main.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.db=OUT/"databases"/("unit_"+uuid.uuid4().hex+".db")
        self.assertFalse(self.db.exists());self.events=[]
        self.stack.enter_context(patch.dict("os.environ",{"LUMINA_TEST_MODE":"1","LUMINA_TEST_DB_PATH":str(self.db)}))
        self.original_connect=sqlite3.connect
        def guarded(path,*args,**kwargs):
            self.assertEqual(Path(path).resolve(),self.db.resolve(),"No production/other database connection allowed")
            ISOLATION_EVENTS.append(dict(test=self.id(),database=str(self.db)))
            return self.original_connect(path,*args,**kwargs)
        self.stack.enter_context(patch.object(sqlite3,"connect",side_effect=guarded))
        engine=StubEngine()
        self.stack.enter_context(patch("src.research_api.create_research_router",side_effect=lambda:create_research_router(engine)))
        self.main=load_app();self.assertEqual(self.main.DB_PATH,self.db)
        self.main.app.add_event_handler("startup",lambda:self.events.append("startup"))
        self.main.app.add_event_handler("shutdown",lambda:self.events.append("shutdown"))
        self.client=self.stack.enter_context(TestClient(self.main.app))
        saved=json.loads((ROOT/"data/processed/lumina_integration_stage1/real_inference_responses.json").read_text())[0]["response"]
        self.payload=dict(id="SYNTHETIC-"+uuid.uuid4().hex,name="Synthetic Unit Case",age="40",date="2026-10-25T00:00:00Z",
            prediction=ABSTENTION,confidence=0,size_cm=0,stage="N/A",report_details=saved["report_details"],
            original_image="",heatmap_image=saved["heatmap_base64"],multi_model_result=saved)
    def tearDown(self):
        self.stack.close();self.assertEqual(self.events,["startup","shutdown"])
    def create(self,payload=None):
        p=payload or self.payload;r=self.client.post("/patients",json=p);self.assertEqual(r.status_code,200,r.text);return p
    def fetch(self):return self.client.get("/patients").json()["patients"]
    def review(self,**change):
        body=dict(case_id=self.payload["id"],doctor_id="SYNTHETIC-REVIEWER",reasoning="Synthetic review fixture only",feedback_type="reviewed")|change
        return self.client.post("/feedback/submit",json=body)

    def test_startup_initializes_empty_existing_schema_and_shutdown(self):
        self.assertEqual(self.fetch(),[])
        conn=sqlite3.connect(str(self.db))
        try:
            self.assertEqual({r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")},{"patients","feedback_logs"})
            self.assertEqual(conn.execute("SELECT count(*) FROM feedback_logs").fetchone()[0],0)
        finally:conn.close()
        self.assertEqual(self.events,["startup"])

    def test_missing_relative_outside_or_production_override_fails_closed(self):
        for env in ({"LUMINA_TEST_MODE":"true"},{"LUMINA_TEST_MODE":""},{"LUMINA_TEST_MODE":"1"},{"LUMINA_TEST_MODE":"1","LUMINA_TEST_DB_PATH":"relative.db"},
                    {"LUMINA_TEST_DB_PATH":str(self.db)}, {"LUMINA_TEST_MODE":"1","LUMINA_TEST_DB_PATH":str(ROOT/"lumina_database.db")}):
            with self.assertRaises(RuntimeError):database_path(ROOT,env)

    def test_full_versioned_result_survives_json_sql_roundtrip(self):
        self.create();r=self.fetch()[0]
        self.assertEqual(r["multi_model_result"],self.payload["multi_model_result"])
        self.assertEqual(r["heatmap_image"],self.payload["heatmap_image"])
        self.assertIsNone(r["multi_model_result"]["consensus"]["confidence"])
        self.assertEqual(r["confidence"],0);self.assertEqual(r["review_status"],"unreviewed")

    def test_review_only_persists_no_assessment_and_keeps_original(self):
        self.create();r=self.review();self.assertEqual(r.status_code,200,r.text)
        stored=self.fetch()[0];review=stored["multi_model_result"]["human_review"]
        self.assertIsNone(review["prediction"]);self.assertIsNone(review["declared_confidence"])
        self.assertFalse(review["identity_verified"]);self.assertEqual(stored["review_status"],"reviewed")
        self.assertEqual(frozen_analysis(stored["multi_model_result"]),self.payload["multi_model_result"])
        self.assertEqual(stored["prediction"],ABSTENTION)

    def test_independent_assessment_does_not_change_ai_or_report(self):
        self.create();self.assertEqual(self.review(feedback_type="independent_assessment",corrected_prediction="Benign").status_code,200)
        r=self.fetch()[0];self.assertEqual(r["prediction"],ABSTENTION)
        self.assertEqual(r["report_details"],self.payload["report_details"])
        self.assertEqual(r["multi_model_result"]["human_review"]["prediction"],"Benign")
        self.assertEqual(r["multi_model_result"]["models"],self.payload["multi_model_result"]["models"])

    def test_review_history_append_and_feedback_retrieval(self):
        self.create();self.review();self.review(feedback_type="independent_assessment",corrected_prediction="Malignant",reasoning="Second synthetic assessment")
        history=self.fetch()[0]["multi_model_result"]["review_history"]
        self.assertEqual(len(history),2);self.assertEqual(history[0]["kind"],"reviewed")
        stats=self.client.get("/feedback/stats").json();self.assertEqual(stats["total_feedback"],2)
        self.assertEqual(len(stats["recent_feedback"]),2)

    def test_incomplete_invalid_and_birads_review_rejected_without_writes(self):
        self.create()
        for change in ({"reasoning":" "},{"doctor_id":" "},{"feedback_type":"independent_assessment"},
                       {"feedback_type":"independent_assessment","corrected_prediction":"4"},
                       {"corrected_confidence":1.1},{"feedback_type":"made_up"}):
            self.assertEqual(self.review(**change).status_code,422,change)
        self.assertEqual(self.client.get("/feedback/stats").json()["total_feedback"],0)
        self.assertEqual(self.fetch()[0]["review_status"],"unreviewed")

    def test_review_only_cannot_claim_diagnostic_assessment(self):
        self.create();self.assertEqual(self.review(corrected_prediction="Benign").status_code,422)

    def test_missing_record_review_is_404(self):
        self.assertEqual(self.review().status_code,404)

    def test_empty_statistics_never_invent_zero_accuracy(self):
        r=self.client.get("/feedback/stats").json()
        self.assertIsNone(r["accuracy_percentage"]);self.assertIsNone(r["avg_model_confidence"])
        self.assertEqual(r["counts"]["total_records"],0);self.assertEqual(r["recent_errors"],[])

    def test_abstentions_and_placeholders_excluded_from_accuracy(self):
        self.create();self.review(feedback_type="independent_assessment",corrected_prediction="Benign")
        r=self.client.get("/feedback/stats").json()
        self.assertEqual(r["counts"]["abstentions"],1);self.assertEqual(r["counts"]["reviewed_records"],1)
        self.assertIsNone(r["accuracy_percentage"]);self.assertIsNone(r["breakdown"][0]["avg_model_confidence"])
        self.assertEqual(r["recent_errors"],[]);self.assertEqual(r["total_predictions_reviewed"],0)

    def test_legacy_row_readable_and_review_does_not_upgrade_or_relabel(self):
        legacy=self.payload|dict(prediction="Benign",confidence=.8,multi_model_result={},heatmap_image="",report_details={"Findings":"Historical synthetic note"})
        self.create(legacy);self.review(feedback_type="independent_assessment",corrected_prediction="Malignant")
        r=self.fetch()[0];self.assertEqual(r["prediction"],"Benign");self.assertEqual(r["confidence"],.8)
        self.assertEqual(r["record_kind"],"historical");self.assertNotIn("contract_version",r["multi_model_result"])
        self.assertEqual(r["report_details"],legacy["report_details"])

    def test_malformed_historical_json_readable_not_silently_repaired(self):
        legacy=self.payload|dict(prediction="Benign",confidence=.8,multi_model_result={},heatmap_image="")
        self.create(legacy);conn=sqlite3.connect(str(self.db))
        try:conn.execute("UPDATE patients SET multi_model_json='not-json'");conn.commit()
        finally:conn.close()
        r=self.fetch()[0];self.assertTrue(r["storage_warnings"]);self.assertEqual(r["record_kind"],"historical")
        self.assertEqual(self.review().status_code,409)
        conn=sqlite3.connect(str(self.db))
        try:self.assertEqual(conn.execute("SELECT multi_model_json FROM patients").fetchone()[0],"not-json")
        finally:conn.close()

    def test_replacing_frozen_models_or_top_label_rejected(self):
        self.create()
        for field,value in (("prediction","Benign"),("confidence",.9),("report_details",{"Findings":"Altered"}),("original_image","different")):
            p=copy.deepcopy(self.payload);p[field]=value
            self.assertEqual(self.client.post("/patients",json=p).status_code,409)
        p=copy.deepcopy(self.payload);p["multi_model_result"]["models"]["BUSI"]["prediction"]="Normal"
        self.assertEqual(self.client.post("/patients",json=p).status_code,409)

    def test_supported_name_age_update_preserves_analysis(self):
        self.create();p=self.payload|dict(name="Synthetic Updated Identity",age="41")
        self.assertEqual(self.client.post("/patients",json=p).status_code,200)
        r=self.fetch()[0];self.assertEqual(r["name"],p["name"]);self.assertEqual(r["multi_model_result"],self.payload["multi_model_result"])

    def test_unsupported_stored_contract_is_historical_and_not_upgraded(self):
        self.create();conn=sqlite3.connect(str(self.db))
        raw=copy.deepcopy(self.payload['multi_model_result']);raw['contract_version']='unsupported'
        try:conn.execute('UPDATE patients SET multi_model_json=?',(json.dumps(raw),));conn.commit()
        finally:conn.close()
        row=self.fetch()[0];self.assertEqual(row['record_kind'],'historical');self.assertTrue(row['storage_warnings'])
        self.assertEqual(row['multi_model_result']['contract_version'],'unsupported')
        self.assertEqual(self.review().status_code,409)
        self.assertEqual(self.client.get('/feedback/stats').json()['counts']['historical_records'],1)

    def test_initial_human_review_cannot_be_forged_inside_analysis_save(self):
        p=copy.deepcopy(self.payload);p["multi_model_result"]["human_review"]={"status":"recorded","prediction":"Benign"}
        self.assertEqual(self.client.post("/patients",json=p).status_code,422)

    def test_sql_failure_rolls_back_feedback_and_review_atomically(self):
        self.create();conn=sqlite3.connect(str(self.db))
        try:conn.execute("CREATE TRIGGER reject_review BEFORE UPDATE ON patients BEGIN SELECT RAISE(ABORT,'synthetic rollback fixture'); END;");conn.commit()
        finally:conn.close()
        r=self.review();self.assertEqual(r.status_code,503);self.assertNotIn(str(self.db),r.text)
        self.assertEqual(self.client.get("/feedback/stats").json()["total_feedback"],0)
        self.assertEqual(self.fetch()[0]["review_status"],"unreviewed")

    def test_unavailable_model_is_not_a_negative_prediction(self):
        p=copy.deepcopy(self.payload);m=p["multi_model_result"]["models"]["UCLM"]
        m.update(status="error",prediction=None,predicted_class_index=None,score=None,confidence=None,probabilities={},logits=None,error_code="synthetic_failure_fixture")
        p["multi_model_result"]["status"]="partial"
        self.create(p);r=self.client.get("/feedback/stats").json()
        self.assertEqual(r["counts"]["unavailable_model_observations"],{"UCLM":1});self.assertIsNone(r["accuracy_percentage"])

    def test_delete_only_synthetic_fixture_and_associated_feedback(self):
        self.create();self.review();self.assertEqual(self.client.delete('/patients/'+self.payload['id']).status_code,200)
        self.assertEqual(self.fetch(),[]);self.assertEqual(self.client.get('/feedback/stats').json()['total_feedback'],0)
        self.assertEqual(self.client.delete('/patients/'+self.payload['id']).status_code,404)

    def test_actual_contract_predict_upload_and_invalid_image_routes(self):
        self.assertEqual(self.client.get('/inference-contract').status_code,200)
        self.assertEqual(self.client.post('/predict',files={'file':('bad.png',b'not image')}).status_code,400)
        r=self.client.post('/predict',files={'file':('fixture.png',png(),'image/png')})
        self.assertEqual(r.status_code,200);self.assertEqual(r.json()['consensus']['prediction'],ABSTENTION)

    def test_test_frontend_cors_origin_only_in_isolated_mode(self):
        r=self.client.options('/patients',headers={'Origin':'http://127.0.0.1:5177','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'})
        self.assertEqual(r.headers.get('access-control-allow-origin'),'http://127.0.0.1:5177')


if __name__=='__main__':unittest.main(verbosity=2)
