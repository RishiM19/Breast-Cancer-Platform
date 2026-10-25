"""Execute upgrade tests and real frozen inference only in explicitly isolated synthetic SQL."""
import base64,copy,csv,json,os,sqlite3,time,unittest,sys
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from .research_inference import ROOT,ResearchEngine,validate_research_response,ABSTENTION
from .research_decision_engine import decision_policy
from .research_storage import frozen_analysis
from .stage2_test_support import SQLGuard,load_main

OUT=ROOT/'data/processed/lumina_ui_verdict_upgrade'
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')

class Checks:
    def __init__(self):self.checks=[]
    def check(self,name,condition,**evidence):
        self.checks.append(dict(name=name,status='PASSED' if condition else 'FAILED',evidence=evidence))
        if not condition:raise AssertionError(name)
    def result(self):return dict(tests_run=len(self.checks),passed=sum(c['status']=='PASSED' for c in self.checks),failed=sum(c['status']=='FAILED' for c in self.checks),checks=self.checks)

def unit_tests():
    os.environ['LUMINA_TEST_ARTIFACT_ROOT']=str(OUT);sys.path.insert(0,str(ROOT/'src'))
    names=['src.test_research_decision_engine','src.test_research_segmentation','src.test_research_integration','src.test_artifact_sensitivity_lumina_classifiers','src.test_lumina_reliability_phase2','src.test_audit_busi_final_reliability','src.test_research_workflow']
    suite=unittest.TestLoader().loadTestsFromNames(names)
    with (OUT/'python_regression.log').open('w',encoding='utf-8') as log:r=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    write('python_test_results.json',dict(tests_run=r.testsRun,passed=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),failed=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),modules=names,failures=[(str(t),s) for t,s in r.failures+r.errors]))
    decision=unittest.TestLoader().loadTestsFromName('src.test_research_decision_engine').countTestCases()
    write('decision_engine_test_results.json',dict(tests_run=decision,passed=decision if r.wasSuccessful() else None,status='PASSED' if r.wasSuccessful() else 'See combined Python log',included_in_python_total=True,fixtures='Explicit synthetic routing/availability scores, not research performance data.'))
    if not r.wasSuccessful():raise AssertionError('Python suite failed')

def actual_workflow():
    db=OUT/'databases/api.db'
    if db.exists():raise FileExistsError('Preserve existing test database; do not overwrite')
    engine=ResearchEngine();guard=SQLGuard(db);compat=Checks();persistence=Checks();api=Checks();responses=[]
    from .research_api import create_research_router
    factory=create_research_router
    rows=list(csv.DictReader((ROOT/'data/processed/lumina_integration_stage3/selected_development_examples.csv').open()))
    image=(ROOT/rows[0]['image_path']).read_bytes()
    with patch.dict(os.environ,{'LUMINA_TEST_MODE':'1','LUMINA_TEST_DB_PATH':str(db)}),patch.object(sqlite3,'connect',guard.connect):
        with patch('src.research_api.create_research_router',side_effect=lambda:factory(engine)):main=load_main()
        with TestClient(main.app) as client:
            def predict(version,domain='UCLM',raw=image):
                start=time.perf_counter();r=client.post('/predict',data=dict(contract_version=version,source_domain=domain),files={'file':('permitted-validation.png',raw,'image/png')})
                if r.status_code!=200:raise AssertionError(r.text)
                result=r.json();validate_research_response(result);responses.append(dict(contract_version=version,declared_domain=domain,seconds=time.perf_counter()-start,response=result));return result
            a=predict('1.0.0');b=predict('1.1.0');c=predict('1.2.0')
            for version in ['1.0.0','1.1.0','1.2.0']:compat.check(f'Discovery {version}',client.get('/inference-contract',params={'version':version}).json()['contract_version']==version)
            compat.check('Default legacy client unchanged',client.post('/predict',files={'file':('x.png',image)}).json()['contract_version']=='1.0.0')
            compat.check('1.0 contains neither segmentation nor new verdict','segmentation' not in a and 'research_verdict' not in a)
            compat.check('1.1 retains segmentation without new verdict',b['segmentation']['status']=='ok' and 'research_verdict' not in b)
            compat.check('1.2 appends verdict without changing classifiers',a['models']==b['models']==c['models'])
            compat.check('Consensus remains identical and abstained',a['consensus']==b['consensus']==c['consensus'] and c['consensus']['score'] is None)
            compat.check('Attribution unchanged and genuine',a['attribution']==c['attribution'] and a['heatmap_base64']==c['heatmap_base64'] and c['attribution']['status']=='ok')
            compat.check('Segmentation unchanged and genuine',b['segmentation']==c['segmentation'] and c['segmentation']['status']=='ok')
            compat.check('Unknown versions reject422',client.post('/predict',data={'contract_version':'9.0.0'},files={'file':('x.png',image)}).status_code==422)
            api.check('UCLM native three-class primary and own score',c['research_verdict']['primary_model_id']=='uclm' and c['research_verdict']['prediction']==c['models']['UCLM']['prediction'] and c['research_verdict']['primary_model_score']==c['models']['UCLM']['score'])
            for domain in ['BUS-BRA','BUSI','unknown']:
                r=predict('1.2.0',domain);v=r['research_verdict'];expected='bra' if domain=='BUS-BRA' else 'uclm'
                api.check(f'{domain} routing and domain restriction',v['primary_model_id']==expected and v['experimental_fallback']==(domain in ['BUSI','unknown']) and r['segmentation']['status']=='not_run' and all(m['status']=='ok' for m in r['models'].values()))
                if domain=='BUS-BRA':api.check('Binary primary cannot claim Normal',v['prediction'] in ['Benign','Malignant'] and v['supported_primary_classes']==['Benign','Malignant'])
            # Same fixed validation cohort selected before Stage 3, no image search or locked inputs.
            for row in rows[1:]:predict('1.2.0','UCLM',(ROOT/row['image_path']).read_bytes())
            repeated=predict('1.2.0');api.check('Repeated output deterministic',repeated==c)
            api.check('All four class labels and own scores retained',all(m['score']==m['probabilities'][m['prediction']] and m['eligibility']['consensus_eligible'] is False for m in c['models'].values()))
            api.check('Invalid image rejected400',client.post('/predict',data={'contract_version':'1.2.0'},files={'file':('invalid.png',b'not an image')}).status_code==400)
            missing=ResearchEngine(model_root=OUT/'missing_checkpoints').analyze(image,'UCLM',contract_version='1.2.0').model_dump()
            api.check('Complete missing inference never invents verdict',missing['research_verdict']['status']=='unavailable' and missing['research_verdict']['prediction'] is None and missing['research_verdict']['primary_model_score'] is None)
            original='data:image/png;base64,'+base64.b64encode(image).decode()
            def payload(value,record_id):return dict(id=record_id,name='Synthetic Verdict API',age='40',date='2026-10-25T12:00:00Z',prediction=ABSTENTION,confidence=0,size_cm=0,stage='N/A',report_details=value['report_details'],original_image=original,heatmap_image=value['heatmap_base64'],multi_model_result=value)
            p=payload(c,'SYNTHETIC-VERDICT-API');persistence.check('Save version1.2',client.post('/patients',json=p).status_code==200)
            before=client.get('/patients').json()['patients'][0];persistence.check('Full envelope fetch exact',before['multi_model_result']==c)
            review=dict(case_id=p['id'],doctor_id='SYNTHETIC-REVIEWER',reasoning='Synthetic independent workflow review; keep AI frozen.',feedback_type='independent_assessment',corrected_prediction='Malignant')
            persistence.check('Independent review recorded',client.post('/feedback/submit',json=review).status_code==200)
            after=client.get('/patients').json()['patients'][0]
            persistence.check('Verdict frozen after human review',after['multi_model_result']['research_verdict']==c['research_verdict'])
            persistence.check('All original outputs frozen',frozen_analysis(after['multi_model_result'])==c)
            persistence.check('Human finding separate',after['multi_model_result']['human_review']['prediction']=='Malignant' and after['prediction']==ABSTENTION)
            tampered=copy.deepcopy(p);tampered['multi_model_result']['research_verdict']['primary_model_score']=.123
            persistence.check('Verdict overwrite rejected409',client.post('/patients',json=tampered).status_code==409)
            persistence.check('Rejected overwrite preserves verdict',frozen_analysis(client.get('/patients').json()['patients'][0]['multi_model_result'])==c)
            for old,version in [(a,'1.0.0'),(b,'1.1.0')]:
                old_p=payload(old,'SYNTHETIC-OLD-'+version);persistence.check('Save old contract '+version,client.post('/patients',json=old_p).status_code==200)
                row=next(r for r in client.get('/patients').json()['patients'] if r['id']==old_p['id']);persistence.check('Old contract '+version+' not reclassified',row['multi_model_result']==old and 'research_verdict' not in row['multi_model_result'])
            persistence.check('Review history persisted',len(after['multi_model_result']['review_history'])==1)
            write('persistence_inputs.json',dict(actual_response=c,before_review=before,after_review=after))
    hooks=sum(len(m._forward_hooks)+len(m._backward_hooks) for model in engine._models.values() for m in model.modules())
    api.check('Cached models and no hook accumulation',len(engine._models)==4 and engine.segmentation.load_count==1 and hooks==0)
    write('research_decision_policy.json',decision_policy());write('contract_compatibility_results.json',compat.result())
    write('persistence_roundtrip_results.json',persistence.result()|dict(sql_isolation=guard.report(),synthetic_only=True))
    write('api_inference_results.json',api.result()|dict(sql_isolation=guard.report(),segmentation_loads=engine.segmentation.load_count,hooks_remaining=hooks))
    write('genuine_inference_responses.json',responses);write('unavailable_inference_fixture.json',missing)


if __name__=='__main__':
    if '--unit' in sys.argv:unit_tests()
    else:actual_workflow()
