"""Genuine three-class API trace fixtures from permitted validation inputs only."""
import base64,csv,json
from fastapi import FastAPI
from fastapi.testclient import TestClient
from .research_api import create_research_router
from .research_inference import ResearchEngine,ABSTENTION
from .diagnose_busi_runtime import ROOT,OUT

def main():
    rows=list(csv.DictReader((OUT/'validation_predictions.csv').open()))
    selected=[next(r for r in rows if r['true_label']==label and r['busi_correct']=='True') for label in ['Normal','Benign','Malignant']]
    app=FastAPI();engine=ResearchEngine();app.include_router(create_research_router(engine));fixtures=[]
    with TestClient(app) as client:
        for i,row in enumerate(selected):
            raw=(ROOT/row['image_path']).read_bytes();r=client.post('/predict',data={'source_domain':'BUSI','contract_version':'1.2.0'},files={'file':(row['image'],raw,'image/png')});assert r.status_code==200
            result=r.json();assert result['models']['BUSI']['prediction']==row['true_label']
            original='data:image/png;base64,'+base64.b64encode(raw).decode()
            record=dict(id='SYNTHETIC-BUSI-TRACE-'+str(i),name='Synthetic Trace '+row['true_label'],age='40',date='2026-10-26T00:00:00Z',prediction=ABSTENTION,confidence=0,size_cm=0,stage='N/A',report_details=result['report_details'],original_image=original,heatmap_image=result['heatmap_base64'],multi_model_result=result,record_kind='versioned_research')
            fixtures.append(dict(image=row['image'],path=row['image_path'],expected=row['true_label'],response=result,patient=record))
    (OUT/'three_class_trace_fixtures.json').write_text(json.dumps(fixtures,indent=2),encoding='utf-8')
    print([(f['image'],f['response']['models']['BUSI']['prediction']) for f in fixtures])

if __name__=='__main__':main()
