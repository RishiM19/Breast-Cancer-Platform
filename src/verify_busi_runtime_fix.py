"""Real API/save/review and prior regressions for BUSI result preservation."""
import base64,copy,json,os,sqlite3,sys,unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from .research_inference import ROOT,ResearchEngine,ABSTENTION
from .research_api import create_research_router
from .research_storage import frozen_analysis
from .stage2_test_support import SQLGuard,load_main

OUT=ROOT/'data/processed/busi_runtime_diagnosis_20261026'
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')

def unit():
    sys.path.insert(0,str(ROOT/'src'))
    from . import test_research_workflow as workflow
    workflow.OUT=OUT  # Test artifact destination only; assertions/SQL guard unchanged.
    names=['src.test_research_decision_engine','src.test_research_segmentation','src.test_research_integration','src.test_artifact_sensitivity_lumina_classifiers','src.test_lumina_reliability_phase2','src.test_audit_busi_final_reliability','src.test_research_workflow']
    suite=unittest.TestLoader().loadTestsFromNames(names)
    with (OUT/'python_regression.log').open('w',encoding='utf-8') as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    write('python_regression_results.json',dict(tests_run=result.testsRun,passed=result.testsRun-len(result.errors)-len(result.failures),errors=len(result.errors),failed=len(result.failures),skipped=len(result.skipped),modules=names,failures=[(str(t),s) for t,s in result.errors+result.failures]))
    if not result.wasSuccessful():raise AssertionError('Regression failure')


def api():
    db=OUT/'databases/api.db'
    if db.exists():raise FileExistsError('Preserve earlier evidence; synthetic database already exists')
    engine=ResearchEngine();guard=SQLGuard(db);checks=[];payloads=[]
    def check(name,condition):
        checks.append(dict(name=name,status='PASSED' if condition else 'FAILED'))
        if not condition:raise AssertionError(name)
    cases=json.loads((OUT/'representative_failures.json').read_text())[:3];assert cases
    with patch.dict(os.environ,{'LUMINA_TEST_MODE':'1','LUMINA_TEST_DB_PATH':str(db)}),patch.object(sqlite3,'connect',guard.connect):
        with patch('src.research_api.create_research_router',side_effect=lambda:create_research_router(engine)):main=load_main()
        with TestClient(main.app) as client:
            for i,case in enumerate(cases):
                raw=Path(case['image_path']).read_bytes()
                response=client.post('/predict',data=dict(source_domain='BUSI',contract_version='1.2.0'),files={'file':(case['image'],raw,'image/png')})
                check(f'Genuine failure example{i} API success',response.status_code==200)
                actual=response.json();b=actual['models']['BUSI'];v=actual['research_verdict']
                check(f'Example{i} correct BUSI versus wrong UCLM is reproduced',b['prediction']==case['true_label'] and v['prediction']!=case['true_label'] and v['primary_model_id']=='uclm')
                check(f'Example{i} BUSI own label/index/score remains correct',b['ordered_class_labels']==['Normal','Benign','Malignant'] and b['ordered_class_labels'][b['predicted_class_index']]==b['prediction'] and b['score']==b['probabilities'][b['prediction']])
                check(f'Example{i} consensus restriction intact',actual['consensus']['status']=='abstained' and b['eligibility']['consensus_eligible'] is False and actual['segmentation']['status']=='not_run')
                payloads.append(dict(case=case,response=actual))
            actual=payloads[0]['response'];case=cases[0];raw=Path(case['image_path']).read_bytes()
            patient=dict(id='SYNTHETIC-BUSI-RUNTIME-REGRESSION',name='Synthetic BUSI Runtime Case',age='40',date='2026-10-26T00:00:00Z',prediction=ABSTENTION,confidence=0,size_cm=0,stage='N/A',report_details=actual['report_details'],original_image='data:image/png;base64,'+base64.b64encode(raw).decode(),heatmap_image=actual['heatmap_base64'],multi_model_result=actual)
            check('Save exact original model envelope',client.post('/patients',json=patient).status_code==200)
            before=client.get('/patients').json()['patients'][0];check('BUSI and UCLM outputs survive SQL reload',before['multi_model_result']==actual)
            check('Review submission succeeds',client.post('/feedback/submit',json=dict(case_id=patient['id'],doctor_id='SYNTHETIC-BUSI-REVIEWER',reasoning='Regression workflow only; preserve BUSI and UCLM as separate outputs.',feedback_type='reviewed')).status_code==200)
            after=client.get('/patients').json()['patients'][0];check('Review cannot alter BUSI or routed verdict',frozen_analysis(after['multi_model_result'])==actual)
            changed=copy.deepcopy(patient);changed['multi_model_result']['models']['BUSI']['prediction']='Altered'
            check('Overwriting original BUSI output is rejected409',client.post('/patients',json=changed).status_code==409)
            for version in ['1.0.0','1.1.0']:
                r=client.post('/predict',data=dict(source_domain='BUSI',contract_version=version),files={'file':(case['image'],raw)})
                check(f'Older contract{version} retains BUSI mapping and no fabricated verdict',r.status_code==200 and r.json()['models']['BUSI']['prediction']==actual['models']['BUSI']['prediction'] and 'research_verdict' not in r.json())
            write('api_display_fixture.json',dict(response=actual,patient=after,case=case))
            write('actual_api_examples.json',payloads)
    write('api_regression_results.json',dict(tests_run=len(checks),passed=sum(c['status']=='PASSED' for c in checks),failed=sum(c['status']=='FAILED' for c in checks),checks=checks,sql_isolation=guard.report(),real_model_inference=True))


if __name__=='__main__':
    if '--unit' in sys.argv:unit()
    else:api()
