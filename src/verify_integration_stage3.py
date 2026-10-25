"""Bounded executed Stage 3 evidence; validation-only inputs and guarded synthetic SQL."""
import ast
import base64
import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
import sqlite3
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from . import research_segmentation as seg
from .research_inference import ROOT, ResearchEngine, validate_research_response, ABSTENTION
from .research_storage import frozen_analysis
from .stage2_test_support import SQLGuard, load_main

OUT=ROOT/'data/processed/lumina_integration_stage3'


def write(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')


class Checks:
    def __init__(self):self.checks=[]
    def check(self,name,condition,**evidence):
        self.checks.append(dict(name=name,status='PASSED' if bool(condition) else 'FAILED',evidence=evidence))
        if not condition:raise AssertionError(name)
    def report(self):
        return dict(tests_run=len(self.checks),passed=sum(c['status']=='PASSED' for c in self.checks),
            failed=sum(c['status']=='FAILED' for c in self.checks),checks=self.checks)


def scientific():
    engine=ResearchEngine();device=engine.device
    rows=list(csv.DictReader((OUT/'selected_development_examples.csv').open()))
    # Load the preserved pre-edit dataset implementation independently.
    spec=importlib.util.spec_from_file_location('stage3_original_dataset',OUT/'before/src/uclm_segmentation_dataset_v2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    dataset=module.UCLMSegmentationDatasetV2(OUT/'selected_development_examples.csv',ROOT,size=256,augment=False)
    # Exact original architecture class nodes, not a guessed replacement.
    tree=ast.parse((OUT/'before/src/train_uclm_segmentation_v2.py').read_text())
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in {'DoubleConv','DownBlock','UpBlock','UNet'}]
    namespace=dict(nn=torch.nn,F=torch.nn.functional,torch=torch)
    exec(compile(ast.Module(body=classes,type_ignores=[]),'preserved_UNet_classes','exec'),namespace)
    checkpoint=ROOT/'models'/seg.FILENAME
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    ref=namespace['UNet'](base=int(ck['base_channels']),num_classes=2).to(device)
    load=ref.load_state_dict(ck['model_state_dict'],strict=True);ref.eval()
    source=(ROOT/'src/evaluate_uclm_segmentation_v3_binary.py').read_text(encoding='utf-8')
    nodes=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.With) and 'torch.autocast' in ast.unparse(n.items[0].context_expr)]
    assert len(nodes)==1
    code=compile(ast.Module(body=nodes,type_ignores=[]),'original_evaluator_forward_AST','exec')
    checks=Checks();cases=[];all_hashes=[]
    for i,row in enumerate(rows):
        image_path=ROOT/row['image_path'];mask_path=ROOT/row['mask_path'];image=Image.open(image_path).convert('RGB')
        for p in [image_path,mask_path]:all_hashes.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        tensor,target=dataset[i];new=seg.preprocess_image(image)
        checks.check(f'{row["image"]}: tensor parity vs pre-edit dataset',torch.equal(new[0],tensor))
        captured=[];handle=ref.register_forward_hook(lambda m,a,o:captured.append(o.detach().clone()))
        env=dict(torch=torch,device=device,model=ref,images=tensor.unsqueeze(0))
        with torch.inference_mode():exec(code,env)
        handle.remove();logits=captured.pop();prob=logits.float().softmax(1)[0,1].cpu().numpy()
        t=time.perf_counter();result=engine.segmentation.analyze(image,'UCLM');elapsed=time.perf_counter()-t
        checks.check(f'{row["image"]}: genuine pinned segmentation',result.status=='ok',status=result.status)
        p=np.load(io.BytesIO(base64.b64decode(result.foreground_probability_npy_base64)),allow_pickle=False)
        diff=float(np.max(np.abs(p-prob)))
        checks.check(f'{row["image"]}: foreground probability parity',diff==0,max_absolute_difference=diff)
        expected=Image.fromarray(env['pred'][0].numpy().astype(np.uint8)*255).resize(image.size,Image.Resampling.NEAREST)
        actual=seg.decode_png(result.predicted_mask,'L',image.size)
        checks.check(f'{row["image"]}: original evaluator mask parity',np.array_equal(actual,np.asarray(expected)))
        checks.check(f'{row["image"]}: native mask dimensions',actual.shape==(image.height,image.width))
        alpha=seg.decode_png(result.overlay,'RGBA',image.size)
        checks.check(f'{row["image"]}: transparent overlay alignment',np.array_equal(alpha[:,:,3]>0,actual>0))
        checks.check(f'{row["image"]}: threshold behavior',np.array_equal(env['pred'][0].numpy(),(logits[0,1]>logits[0,0]).cpu().numpy()))
        checks.check(f'{row["image"]}: empty-mask flag',result.empty_mask==bool(not actual.any()))
        prefix=OUT/'examples'/image_path.stem;prefix.parent.mkdir(exist_ok=True)
        image.save(str(prefix)+'_original.png');expected.save(str(prefix)+'_predicted_mask.png')
        Image.fromarray(alpha).save(str(prefix)+'_transparent_overlay.png')
        ref_mask=Image.open(mask_path).convert('RGB');gt=(module.rgb_to_labels(np.asarray(ref_mask))!=0).astype(np.uint8)*255
        Image.fromarray(gt).save(str(prefix)+'_reference_annotation.png')
        blend=Image.alpha_composite(image.convert('RGBA'),Image.fromarray(alpha)).convert('RGB')
        sheet=Image.new('RGB',(1200,330),'white');draw=ImageDraw.Draw(sheet)
        for j,(im,title) in enumerate([(image,'Original validation image'),(Image.fromarray(actual),'AI-predicted foreground'),(blend,'AI overlay on original'),(Image.fromarray(gt),'Reference annotation ONLY')]):
            im=im.convert('RGB');im.thumbnail((298,298));sheet.paste(im,(j*300,30));draw.text((j*300+3,7),title,fill='black')
        sheet.save(str(prefix)+'_comparison.png')
        cases.append(dict(image=row['image'],declared_label=row['label'],input_dimensions=list(image.size),
            probability_max_abs_difference=diff,mask_pixels_different=int(np.count_nonzero(actual!=np.asarray(expected))),
            genuine_empty_mask=result.empty_mask,foreground_pixels=int((actual>0).sum()),inference_seconds=elapsed,
            reference_annotation_used_for_inference=False,clinical_performance_estimated=False))
    checks.check('Model cached across three examples',engine.segmentation.load_count==1,load_count=engine.segmentation.load_count)
    audit=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        metadata={k:v for k,v in ck.items() if k!='model_state_dict'},architecture='Original V2 UNet reused with 2 output classes, base=32',
        parameter_count=sum(p.numel() for p in ref.parameters()),strict_load=dict(missing_keys=load.missing_keys,unexpected_keys=load.unexpected_keys),
        preprocessing=seg.PREPROCESSING,saved_tuned_threshold=False,evaluation_decision='two-logit argmax; ties background; exact-arithmetic foreground >0.5',
        visualization_source_issue='visualize_uclm_segmentation_v3.py points to older 3-class checkpoint and locked test CSV; inspected only, never executed',
        gpu_available=torch.cuda.is_available(),device=str(device),gpu_name=torch.cuda.get_device_name() if device.type=='cuda' else None,
        source_evaluator_sha256=hashlib.sha256(source.encode()).hexdigest(),executed_original_evaluator_snippet=ast.unparse(nodes[0]),
        evaluator_main_executed=False,locked_test_images_accessed=False)
    write('segmentation_model_audit.json',audit);write('segmentation_inference_tests.json',checks.report()|dict(cases=cases,interpretation='Regression consistency on 3 preselected validation examples, not a new accuracy study.'))
    write('development_input_hashes.json',all_hashes)
    del ref;return engine,rows


def workflow(engine,rows,database_name='api.db'):
    if Path(database_name).name!=database_name or Path(database_name).suffix!='.db':
        raise ValueError('Pass a synthetic test database filename only')
    checks=Checks();persist=Checks();db=OUT/'databases'/database_name
    if db.exists():raise FileExistsError('Preserve prior test evidence; use a new explicit synthetic DB')
    raw=(ROOT/rows[0]['image_path']).read_bytes()
    guard=SQLGuard(db);durations=[];ids_before=None
    torch.cuda.reset_peak_memory_stats() if engine.device.type=='cuda' else None
    with patch.dict(os.environ,{'LUMINA_TEST_MODE':'1','LUMINA_TEST_DB_PATH':str(db)}),patch.object(sqlite3,'connect',guard.connect):
        from .research_api import create_research_router
        factory=create_research_router
        with patch('src.research_api.create_research_router',side_effect=lambda:factory(engine)):
            main=load_main()
        with TestClient(main.app) as client:
            checks.check('Isolated empty database startup',client.get('/patients').json()['patients']==[])
            descriptor=client.get('/inference-contract',params={'version':'1.1.0'}).json()
            checks.check('Contract 1.1 discovery',descriptor['contract_version']=='1.1.0' and descriptor['segmentation']['eligible_domains']==['UCLM'])
            checks.check('Legacy discovery remains 1.0',client.get('/inference-contract').json()['contract_version']=='1.0.0')
            def predict(domain='UCLM',version='1.1.0'):
                t=time.perf_counter();r=client.post('/predict',data=dict(source_domain=domain,contract_version=version),files={'file':('development.png',raw,'image/png')});durations.append(time.perf_counter()-t)
                if r.status_code!=200:raise AssertionError(r.text)
                return r.json()
            actual=predict();validate_research_response(actual)
            write('actual_uclm_response.json',actual)
            checks.check('Genuine four classifiers, BRA Grad-CAM and UCLM segmentation',all(m['status']=='ok' for m in actual['models'].values()) and actual['attribution']['status']=='ok' and actual['segmentation']['status']=='ok')
            checks.check('All four own-label scores and BUSI excluded',all(m['score']==m['probabilities'][m['prediction']] and m['eligibility']['consensus_eligible'] is False for m in actual['models'].values()))
            checks.check('All consensus abstains',actual['consensus']['status']=='abstained' and actual['consensus']['score'] is None)
            identities={k:id(v) for k,v in engine._models.items()};ids_before=identities
            repeat=predict()
            checks.check('Repeated request same frozen outputs',actual==repeat)
            with ThreadPoolExecutor(max_workers=2) as pool:concurrent=list(pool.map(lambda _:predict(),range(2)))
            checks.check('Two concurrent requests deterministic and serialized',all(r==actual for r in concurrent))
            for domain in ['unknown','BUSI','BUS-BRA']:
                r=predict(domain);checks.check(f'{domain} classifiers preserved, segmentation not run',all(m['status']=='ok' for m in r['models'].values()) and r['segmentation']['status']=='not_run' and r['segmentation']['predicted_mask'] is None)
            legacy=predict('UCLM','1.0.0');checks.check('Actual legacy caller preserved',legacy['contract_version']=='1.0.0' and 'segmentation' not in legacy and legacy['models']==actual['models'])
            with patch.object(engine.segmentation,'path',OUT/'missing_segmentation_fixture.pth'):
                missing=predict();checks.check('Missing checkpoint keeps classifiers and CAM',missing['status']=='partial' and missing['segmentation']['status']=='missing' and missing['segmentation']['overlay'] is None and all(m['status']=='ok' for m in missing['models'].values()) and missing['attribution']['status']=='ok')
                write('missing_checkpoint_response.json',missing)
            checks.check('Invalid image API400',client.post('/predict',data={'contract_version':'1.1.0'},files={'file':('bad.png',b'not image')}).status_code==400)
            oriented=Image.open(io.BytesIO(raw)).convert('RGB');exif=oriented.getexif();exif[274]=6
            buf=io.BytesIO();oriented.save(buf,format='JPEG',exif=exif)
            t=time.perf_counter()
            rotation=client.post('/predict',data=dict(source_domain='UCLM',contract_version='1.1.0'),files={'file':('orientation.jpg',buf.getvalue(),'image/jpeg')})
            durations.append(time.perf_counter()-t);rotated=rotation.json()
            checks.check('Non-default EXIF declines masks, retains classifiers and attribution',rotation.status_code==200 and rotated['status']=='partial' and rotated['segmentation']['status']=='not_run' and rotated['segmentation']['error_code']=='unsupported_image_orientation' and rotated['segmentation']['overlay'] is None and all(m['status']=='ok' for m in rotated['models'].values()) and rotated['attribution']['status']=='ok')
            checks.check('Unsupported contract API422',client.post('/predict',data={'contract_version':'2.0.0'},files={'file':('x.png',raw)}).status_code==422)
            original='data:image/png;base64,'+base64.b64encode(raw).decode()
            payload=dict(id='SYNTHETIC-STAGE3-API',name='Synthetic Stage3',age='0',date='2026-10-25T00:00:00Z',prediction=ABSTENTION,confidence=0,size_cm=0,stage='N/A',report_details=actual['report_details'],original_image=original,heatmap_image=actual['heatmap_base64'],multi_model_result=actual)
            persist.check('Save 1.1 response',client.post('/patients',json=payload).status_code==200)
            before=client.get('/patients').json()['patients'][0]
            persist.check('Exact frozen envelope roundtrip',before['multi_model_result']==actual and before['record_kind']=='versioned_research')
            persist.check('Original image preserved',before['original_image']==original)
            review=client.post('/feedback/submit',json=dict(case_id=payload['id'],doctor_id='SYNTHETIC-UNVERIFIED',reasoning='Independent synthetic workflow review',feedback_type='independent_assessment',corrected_prediction='Benign'))
            persist.check('Independent review recorded',review.status_code==200)
            after=client.get('/patients').json()['patients'][0]
            persist.check('All AI fields frozen after review',frozen_analysis(after['multi_model_result'])==actual)
            persist.check('Segmentation bytes, probability map and provenance unchanged',after['multi_model_result']['segmentation']==actual['segmentation'])
            persist.check('Human assessment remains separate',after['multi_model_result']['human_review']['prediction']=='Benign' and after['prediction']==ABSTENTION)
            persist.check('Unverified identity explicit',after['multi_model_result']['human_review']['identity_verified'] is False)
            altered=copy.deepcopy(payload);altered['multi_model_result']['segmentation']=missing['segmentation']
            persist.check('Overwrite mask rejected409',client.post('/patients',json=altered).status_code==409)
            persist.check('Rejected overwrite leaves frozen result',frozen_analysis(client.get('/patients').json()['patients'][0]['multi_model_result'])==actual)
            old=copy.deepcopy(payload);old.update(id='SYNTHETIC-STAGE3-LEGACY',multi_model_result=legacy)
            persist.check('Save older 1.0 contract',client.post('/patients',json=old).status_code==200)
            all_rows=client.get('/patients').json()['patients'];old_row=next(r for r in all_rows if r['id']==old['id'])
            persist.check('Reload older 1.0 without upgrade',old_row['multi_model_result']==legacy and old_row['record_kind']=='versioned_research')
            stats=client.get('/feedback/stats').json()
            persist.check('Workflow counts, no diagnostic accuracy',stats['counts']['total_records']==2 and stats['counts']['reviewed_records']==1 and stats['diagnostic_accuracy_available'] is False)
            write('persistence_test_inputs.json',dict(actual_response=actual,before_review=before,after_review=after,all_rows=all_rows))
            # Reload segmentation after explicit missing simulation; only expected extra load.
            restored=predict();checks.check('Recovers from missing checkpoint without fallback',restored==actual)
            hooks=sum(len(m._forward_hooks)+len(m._backward_hooks) for model in engine._models.values() for m in model.modules())
            checks.check('No Grad-CAM hooks remain',hooks==0,remaining_hooks=hooks)
            checks.check('Classifier models cached without recreation',ids_before=={k:id(v) for k,v in engine._models.items()})
            checks.check('Segmentation caching and expected recovery reload only',engine.segmentation.load_count==2,loads=engine.segmentation.load_count,expected='one initial load, one after missing-checkpoint invalidation')
    resources=dict(request_seconds=durations,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if engine.device.type=='cuda' else None,
        gpu_current_allocated_bytes=torch.cuda.memory_allocated() if engine.device.type=='cuda' else None,concurrent_requests=2,gradcam_hooks_remaining=hooks,
        segmentation_load_count=engine.segmentation.load_count,classifier_cache_ids_unchanged=True)
    checks.check('Bounded GPU peak below 2 GiB',resources['gpu_peak_allocated_bytes'] is None or resources['gpu_peak_allocated_bytes']<2*1024**3,**resources)
    write('api_integration_results.json',checks.report()|dict(sql_isolation=guard.report(),resources=resources,actual_inference=True))
    write('persistence_roundtrip_results.json',persist.report()|dict(sql_isolation=guard.report(),synthetic_only=True))
    write('inference_contract_update.json',dict(previous_version='1.0.0',new_version='1.1.0',default_version='1.0.0',frontend_requests='1.1.0',classifier_record_version='1.0.0',policy_version='1.0.0',segmentation_separate_from_models_and_consensus=True,old_records_verified=True,schema=descriptor))


def unit_regressions():
    import sys
    sys.path.insert(0,str(ROOT/'src'))  # Preserve historical script-style test imports.
    os.environ['LUMINA_TEST_ARTIFACT_ROOT']=str(OUT)
    names=['src.test_research_segmentation','src.test_research_integration','src.test_artifact_sensitivity_lumina_classifiers','src.test_lumina_reliability_phase2','src.test_audit_busi_final_reliability','src.test_research_workflow']
    suite=unittest.TestLoader().loadTestsFromNames(names)
    with (OUT/'python_regression.log').open('w',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    write('python_regression_results.json',dict(tests_run=result.testsRun,passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),modules=names,failures=[(str(t),trace) for t,trace in result.failures+result.errors]))
    if not result.wasSuccessful():raise AssertionError('Python regressions failed; inspect log')


def cpu_parity():
    torch.set_num_threads(1)
    rows=list(csv.DictReader((OUT/'selected_development_examples.csv').open()))
    image=Image.open(ROOT/rows[0]['image_path']).convert('RGB');device=torch.device('cpu')
    e=seg.SegmentationEngine(device);checks=Checks();r=e.analyze(image,'UCLM')
    checks.check('Genuine CPU float32 inference',r.status=='ok')
    tree=ast.parse((OUT/'before/src/train_uclm_segmentation_v2.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef)]
    ns=dict(nn=torch.nn,F=torch.nn.functional,torch=torch)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'preserved_CPU_UNet','exec'),ns)
    model=ns['UNet'](base=32,num_classes=2);model.load_state_dict(torch.load(e.path,map_location='cpu',weights_only=True)['model_state_dict'],strict=True);model.eval()
    with torch.inference_mode():
        with torch.autocast(device_type='cpu',dtype=torch.float16,enabled=False):logits=model(seg.preprocess_image(image))
        reference=seg.encoded_outputs(logits,image.size)
    checks.check('CPU reference mask and probabilities exact',all(getattr(r,k)==v for k,v in reference.items()))
    repeated=e.analyze(image,'UCLM')
    checks.check('CPU deterministic repeat',repeated==r)
    checks.check('CPU model cached once',e.load_count==1)
    write('cpu_inference_results.json',checks.report()|dict(device='cpu',precision='float32',image=rows[0]['image'],clinical_performance_estimated=False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group();mode.add_argument('--unit',action='store_true');mode.add_argument('--cpu',action='store_true');parser.add_argument('--db',default='api.db');args=parser.parse_args()
    if args.unit:unit_regressions()
    elif args.cpu:cpu_parity()
    else:
        engine,rows=scientific();workflow(engine,rows,args.db)
