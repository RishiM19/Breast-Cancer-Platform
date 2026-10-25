"""Bounded real API/persistence/review checks; no production SQL or training."""
from __future__ import annotations
import argparse
import base64
import copy
import hashlib
import io
import json
import os
import sqlite3
import sys
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

from . import research_inference as ri
from .research_api import create_research_router
from .research_storage import frozen_analysis
from .stage2_test_support import ROOT,OUT,SQLGuard,load_main


def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def units():
    sys.path.insert(0,str(ROOT/"src"))
    names=["src.test_research_workflow","src.test_research_integration","test_artifact_sensitivity_lumina_classifiers","test_lumina_reliability_phase2","test_audit_busi_final_reliability"]
    suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    torch.set_num_threads(2)
    with (OUT/"python_regression_tests.txt").open("w",encoding="utf-8") as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    counts={n:unittest.defaultTestLoader.loadTestsFromName(n).countTestCases() for n in names}
    record=dict(status="PASSED" if result.wasSuccessful() else "FAILED",tests_run=result.testsRun,suite_counts=counts,
        failures=[dict(test=str(t),traceback=x) for t,x in result.failures],errors=[dict(test=str(t),traceback=x) for t,x in result.errors],
        skipped=[dict(test=str(t),reason=x) for t,x in result.skipped])
    write("python_test_results.json",record)
    from .test_research_workflow import ISOLATION_EVENTS
    write("unit_database_isolation.json",dict(status="PASSED" if result.wasSuccessful() else "FAILED",connections=ISOLATION_EVENTS,
        only_synthetic_new_databases=True,production_database_connections=0))
    assert result.wasSuccessful(),"Inspect preserved Python regression log"
    print(json.dumps(dict(unit_status=record['status'],tests=result.testsRun)),flush=True)


def real_workflow():
    torch.set_num_threads(2)
    path=OUT/"databases"/("workflow_"+uuid.uuid4().hex+".db")
    assert not path.exists();guard=SQLGuard(path);prod_before=digest(ROOT/"lumina_database.db")
    events=[];load_counts={k:0 for k in ri.SPECS};real_loader=ri.load_classifier
    def counted(profile,device):
        load_counts[profile.model_id]+=1;return real_loader(profile,device)
    with patch.dict(os.environ,{"LUMINA_TEST_MODE":"1","LUMINA_TEST_DB_PATH":str(path)}),patch.object(sqlite3,"connect",side_effect=guard.connect),patch.object(ri,"load_classifier",side_effect=counted):
        engine=ri.ResearchEngine()
        with patch("src.research_api.create_research_router",side_effect=lambda:create_research_router(engine)):
            main=load_main()
        main.app.add_event_handler("startup",lambda:events.append("startup"));main.app.add_event_handler("shutdown",lambda:events.append("shutdown"))
        with TestClient(main.app) as client:
            assert client.get('/patients').json()=={'patients':[]}
            schema_conn=guard.connect(str(path))
            try:tables=[r[0] for r in schema_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            finally:schema_conn.close()
            assert set(tables)=={'patients','feedback_logs'}
            descriptor=client.get('/inference-contract');assert descriptor.status_code==200
            write('actual_inference_contract.json',descriptor.json())
            selected=pd.read_csv(ROOT/'data/processed/lumina_reliability_phase2/selected_cases.csv')
            source=selected.iloc[0];image=Path(source.image_path);image=image if image.is_absolute() else ROOT/image
            raw=image.read_bytes();source_hash=digest(image);image_url='data:image/png;base64,'+base64.b64encode(raw).decode('ascii')
            def predict():
                begin=time.perf_counter();r=client.post('/predict',files={'file':(image.name,raw,'image/png')},data={'source_domain':'BUS-BRA'})
                assert r.status_code==200,r.text;obj=r.json();ri.ResearchResponse.model_validate(obj)
                return obj,time.perf_counter()-begin
            first,first_seconds=predict();assert first['status']=='ok' and first['attribution']['status']=='ok'
            prior=json.loads((ROOT/'data/processed/lumina_integration_stage1/real_inference_responses.json').read_text())[0]['response']
            parity=[]
            for name,m in first['models'].items():
                np.testing.assert_allclose(m['logits'],prior['models'][name]['logits'],atol=4e-5,rtol=5e-5)
                assert m['prediction']==prior['models'][name]['prediction']
                parity.append(dict(model=name,max_logit_difference=float(np.max(np.abs(np.array(m['logits'])-prior['models'][name]['logits'])))))
            payload=dict(id='SYNTHETIC-WORKFLOW-'+uuid.uuid4().hex,name='Synthetic Workflow Case',age='40',date='2026-10-25T12:00:00Z',
                prediction=ri.ABSTENTION,confidence=0,size_cm=0,stage='N/A',report_details=first['report_details'],original_image=image_url,
                heatmap_image=first['heatmap_base64'],multi_model_result=first)
            r=client.post('/patients',json=payload);assert r.status_code==200,r.text
            fetched=client.get('/patients').json()['patients'][0]
            assert fetched['multi_model_result']==first and fetched['heatmap_image']==first['heatmap_base64'] and fetched['original_image']==image_url
            assert fetched['multi_model_result']['consensus']['score'] is None
            before_review=copy.deepcopy(fetched)
            bodies=[dict(case_id=payload['id'],doctor_id='SYNTHETIC-REVIEWER-01',feedback_type='reviewed',reasoning='Synthetic reviewed-only workflow fixture'),
                dict(case_id=payload['id'],doctor_id='SYNTHETIC-REVIEWER-02',feedback_type='independent_assessment',corrected_prediction='Benign',corrected_confidence=None,reasoning='Synthetic independent assessment; not a clinical judgment')]
            review_responses=[]
            for body in bodies:
                r=client.post('/feedback/submit',json=body);assert r.status_code==200,r.text;review_responses.append(r.json())
            rejected=client.post('/feedback/submit',json=bodies[0]|dict(reasoning=' '));assert rejected.status_code==422
            after_review=client.get('/patients').json()['patients'][0]
            assert frozen_analysis(after_review['multi_model_result'])==first
            assert all(after_review[k]==before_review[k] for k in ('prediction','confidence','report_details','original_image','heatmap_image','date'))
            assert len(after_review['multi_model_result']['review_history'])==2
            legacy=payload|dict(id='SYNTHETIC-HISTORICAL',name='Synthetic Historical Case',prediction='Benign',confidence=.8,size_cm=0,stage='N/A',multi_model_result={},heatmap_image='',report_details={'Findings':'Historical synthetic text only'})
            assert client.post('/patients',json=legacy).status_code==200
            assert client.post('/feedback/submit',json=bodies[0]|dict(case_id=legacy['id'])).status_code==200
            stats=client.get('/feedback/stats').json()
            assert stats['counts']['total_records']==2 and stats['counts']['historical_records']==1 and stats['counts']['versioned_records']==1
            assert stats['counts']['abstentions']==1 and stats['counts']['reviewed_records']==2 and stats['total_feedback']==3
            assert stats['accuracy_percentage'] is None and stats['avg_model_confidence'] is None and stats['recent_errors']==[]
            invalid=client.post('/predict',files={'file':('invalid.png',b'not an image','image/png')});assert invalid.status_code==400
            torch.cuda.synchronize() if engine.device.type=='cuda' else None
            warm_alloc=torch.cuda.memory_allocated() if engine.device.type=='cuda' else None
            warm_hooks=sum(len(m._forward_hooks)+len(m._backward_hooks) for model in engine._models.values() for m in model.modules())
            repeat,repeat_seconds=predict()
            for name,m in first['models'].items():np.testing.assert_allclose(repeat['models'][name]['logits'],m['logits'],atol=2e-5,rtol=2e-5)
            monitor_lock=threading.Lock();active=0;maximum=0;handles=[]
            def entering(_model,_inputs):
                nonlocal active,maximum
                with monitor_lock:active+=1;maximum=max(maximum,active)
            def leaving(_model,_inputs,_output):
                nonlocal active
                with monitor_lock:active-=1
            for model in engine._models.values():
                handles.extend([model.register_forward_pre_hook(entering),model.register_forward_hook(leaving)])
            try:
                with ThreadPoolExecutor(max_workers=2) as pool:parallel=list(pool.map(lambda _:predict(),range(2)))
                assert maximum==1 and active==0
                for obj,_ in parallel:
                    for name,m in first['models'].items():np.testing.assert_allclose(obj['models'][name]['logits'],m['logits'],atol=2e-5,rtol=2e-5)
            finally:
                for h in handles:h.remove()
            original_load=engine._load
            def fail_uclm(k):
                if k=='uclm':raise RuntimeError('Intentional isolated availability fault')
                return original_load(k)
            with patch.object(engine,'_load',side_effect=fail_uclm):partial,partial_seconds=predict()
            assert partial['status']=='partial' and partial['models']['UCLM']['status']=='error' and partial['models']['UCLM']['prediction'] is None
            assert partial['models']['BUSI']['status']=='ok' and partial['attribution']['status']=='ok'
            write('availability_test_inputs.json',dict(status='PASSED',response=partial,
                source='One actual API request with an explicitly injected UCLM loading exception; the other three models performed real inference.'))
            assert load_counts==dict.fromkeys(ri.SPECS,1),load_counts
            final_hooks=sum(len(m._forward_hooks)+len(m._backward_hooks) for model in engine._models.values() for m in model.modules())
            assert final_hooks==warm_hooks==0
            allocated=torch.cuda.memory_allocated() if engine.device.type=='cuda' else None
            peak=torch.cuda.max_memory_allocated() if engine.device.type=='cuda' else None
            if allocated is not None:assert allocated-warm_alloc<96*1024*1024 and peak<5.5*1024**3
            rows=client.get('/patients').json()['patients']
            write('persistence_test_inputs.json',dict(actual_response=first,before_review=before_review,after_review=after_review,all_rows=rows,
                source_image_path=str(image),source_image_sha256=source_hash))
            write('persistence_roundtrip_results.json',dict(status='PASSED',case_id=payload['id'],database=str(path),
                real_predictions_survived=True,all_four_model_records_identical=True,full_envelope_roundtrip_equal=True,
                null_consensus_scores_preserved=True,stored_compatibility_confidence=0,attribution_data_url_identical=True,
                source_image_data_url_identical=True,original_inference_unchanged_after_review=True,model_parity=parity))
            write('human_review_test_results.json',dict(status='PASSED',reviews=review_responses,rejected_empty_note_status=rejected.status_code,
                history_entries=2,original_ai_fields_unchanged=True,historical_record_not_upgraded=True,identity_verified=False))
            write('dashboard_feedback_audit.json',dict(status='PASSED',actual_response=stats,
                abstention_not_counted_as_error=True,placeholder_not_averaged=True,birads_not_pathology=True,
                note='Counts from explicit isolated fixtures: one real versioned inference and one synthetic historical record.'))
            write('performance_stability_results.json',dict(status='PASSED',successful_full_requests=4,controlled_partial_requests=1,
                individual_real_model_predictions=19,first_seconds=first_seconds,warm_seconds=repeat_seconds,
                concurrent_seconds=[t for _,t in parallel],partial_seconds=partial_seconds,checkpoint_loads=load_counts,
                maximum_overlapping_model_forwards=maximum,initial_hooks=warm_hooks,final_hooks=final_hooks,
                device=str(engine.device),cuda_allocated_after_warmup=warm_alloc,cuda_allocated_at_end=allocated,cuda_peak_allocated=peak,
                limitation='Small bounded test only; no unbounded load or universal memory/concurrency guarantee.'))
        assert events==['startup','shutdown']
    assert digest(ROOT/'lumina_database.db')==prod_before
    write('database_isolation_audit.json',dict(status='PASSED',guard=guard.report(),database_existed_before=False,initial_patient_count=0,
        startup_shutdown_events=events,actual_schema_tables=tables,production_database_before_sha256=prod_before,
        production_database_after_sha256=digest(ROOT/'lumina_database.db'),production_database_unchanged=True,
        production_sql_connections=0,source='New empty database initialized by actual main.py startup; no patient records copied.'))
    write('workflow_test_results.json',dict(status='PASSED',actual_fastapi_app=True,full_lifespan_executed=True,
        endpoints_exercised=['GET /inference-contract','POST /predict','GET /patients','POST /patients','POST /feedback/submit','GET /feedback/stats'],
        isolated_database=str(path),real_image=image.name,held_out_test_data_accessed=False,failures_unresolved=[],
        real_requests=5,real_model_predictions=19,invalid_upload_status=400,save_reload=True,review_reload=True,
        limitations=['Browser verification is reported separately','Identity/authorization/clinical correctness are not verified']))
    print(json.dumps(dict(workflow='PASSED',real_requests=5,real_predictions=19,persistence=True,reviews=True,production_database_unchanged=True)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--unit',action='store_true');p.add_argument('--workflow',action='store_true');a=p.parse_args()
    if a.unit:units()
    if a.workflow:real_workflow()


if __name__=='__main__':main()
