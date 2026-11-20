"""Validation-only frozen BUSI runtime parity; never invokes all-split factories or training."""
import ast,csv,hashlib,io,json,os,time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
from .research_inference import ROOT,ResearchEngine,sha256
from .lumina_gradcam_all_classifiers import PROFILES,preprocess

OUT=ROOT/'data/processed/busi_runtime_diagnosis_20261026'
ORIGINAL=Path('C:/Breast_Cancer_Project')


def main():
    engine=ResearchEngine();start=time.perf_counter()
    expected='8e59e91c25cc6e591ae042d98951bda05b5c2fd31a49c78f4ce47b0d34d060e5'
    actual=sha256(ROOT/'models/best_model.pth');original_hash=sha256(ORIGINAL/'models/best_model.pth')
    assert actual==original_hash==expected
    source=(ORIGINAL/'src/dataset.py').read_text(encoding='utf-8-sig');tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BUSIDataset')
    factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_dataloaders')
    transform=next(n for n in ast.walk(factory) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='data_transforms' for t in n.targets))
    ns=dict(os=os,pd=pd,torch=torch,Dataset=Dataset,Image=Image,transforms=transforms,IMAGE_SIZE=256)
    exec(compile(ast.Module(body=[cls,transform],type_ignores=[]),'recovered_BUSI_validation_only','exec'),ns)
    data=ns['BUSIDataset'](str(ROOT/'data/processed/val.csv'),str(ROOT),ns['data_transforms'])
    manifest=list(csv.DictReader((ROOT/'data/processed/val.csv').open()))
    category_tree=ast.parse((ORIGINAL/'src/process_data.py').read_text(encoding='utf-8-sig'))
    categories=next(ast.literal_eval(n.value) for n in category_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='categories' for t in n.targets))
    assert categories==dict(normal=0,benign=1,malignant=2)
    classes=['Normal','Benign','Malignant'];assert list(PROFILES['busi'].classes)==classes
    reference=efficientnet_b0(weights=None);reference.classifier[1]=torch.nn.Linear(reference.classifier[1].in_features,3)
    loaded=reference.load_state_dict(torch.load(ROOT/'models/best_model.pth',map_location='cpu',weights_only=True),strict=True)
    reference.to(engine.device).eval()
    previous=list(csv.DictReader((ROOT/'data/processed/busi_final_reliability_audit/all_accessible_predictions.csv').open()))
    previous={str(Path(r['path']).resolve()).casefold():r for r in previous if r['split']=='val'}
    rows=[];max_tensor=0.;max_logit=0.;max_prob=0.;failures=[]
    with torch.inference_mode():
        for i,entry in enumerate(manifest):
            path=(ROOT/entry['filepath']).resolve();tensor,truth=data[i]
            _,runtime_tensor=preprocess(path,PROFILES['busi']);tensor_error=float((tensor-runtime_tensor[0]).abs().max());max_tensor=max(max_tensor,tensor_error)
            assert tensor_error==0
            ref_logits=reference(tensor.unsqueeze(0).to(engine.device))[0].float().cpu();ref_scores=ref_logits.softmax(0)
            # Avoid enclosing model creation in inference_mode; the application also uses Grad-CAM.
            with torch.inference_mode(False):
                result=engine.analyze(path.read_bytes(),'BUSI',run_models={'busi','uclm'},attribution=False,contract_version='1.2.0')
            b=result.models['BUSI'];u=result.models['UCLM'];verdict=result.research_verdict
            assert b.status==u.status=='ok'
            logit_error=float((torch.tensor(b.logits)-ref_logits).abs().max());prob_error=max(abs(b.probabilities[c]-float(ref_scores[j])) for j,c in enumerate(classes))
            max_logit=max(max_logit,logit_error);max_prob=max(max_prob,prob_error)
            prior=previous[str(path).casefold()]
            assert b.predicted_class_index==int(ref_logits.argmax())==int(prior['predicted_index'])
            assert b.prediction==classes[b.predicted_class_index] and b.score==b.probabilities[b.prediction]
            assert int(entry['label'])==truth and entry['label_name'].lower()==classes[truth].lower()
            assert verdict.primary_model_id=='uclm' and verdict.prediction==u.prediction and verdict.primary_model_score==u.score
            row=dict(validation_index=i,image=path.name,image_path=str(path),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),true_index=truth,true_label=classes[truth],
                busi_index=b.predicted_class_index,busi_prediction=b.prediction,busi_score=b.score,busi_correct=b.prediction==classes[truth],
                uclm_prediction=u.prediction,uclm_score=u.score,uclm_correct=u.prediction==classes[truth],
                displayed_final_prediction=verdict.prediction,final_primary_model=verdict.primary_model_id,
                previous_busi_prediction=prior['predicted_class'],tensor_max_error=tensor_error,logit_max_error=logit_error,probability_max_error=prob_error)
            rows.append(row)
            if row['busi_correct'] and not row['uclm_correct'] and len(failures)<6:failures.append(row)
    def confusion(key):return [[sum(r['true_index']==a and r[key]==classes[b] for r in rows) for b in range(3)] for a in range(3)]
    result=dict(validation_images=len(rows),interpretation='Runtime/debugging reproduction on reused contaminated validation data; not independent accuracy, calibration or clinical validation.',
        checkpoint_sha256=actual,original_checkpoint_sha256=original_hash,original_category_mapping=categories,current_class_order=list(PROFILES['busi'].classes),
        strict_load=dict(missing_keys=loaded.missing_keys,unexpected_keys=loaded.unexpected_keys),original_dataset_factory_executed=False,locked_test_data_accessed=False,
        preprocessing_tensor_max_error=max_tensor,busi_reference_logit_max_error=max_logit,busi_reference_probability_max_error=max_prob,
        busi_correct=sum(r['busi_correct'] for r in rows),uclm_fallback_correct=sum(r['uclm_correct'] for r in rows),
        old_wrong_mapping_correct=sum(classes[r['true_index']]==['Benign','Malignant','Normal'][r['busi_index']] for r in rows),
        busi_confusion=confusion('busi_prediction'),uclm_fallback_confusion=confusion('uclm_prediction'),
        all_busi_predictions_match_previous_audit=all(r['busi_prediction']==r['previous_busi_prediction'] for r in rows),
        busi_vs_displayed_final_disagreements=sum(r['busi_prediction']!=r['displayed_final_prediction'] for r in rows),
        busi_correct_but_final_wrong=sum(r['busi_correct'] and not r['uclm_correct'] for r in rows),
        busi_errors_remaining=[r for r in rows if not r['busi_correct']],
        source_domain_BUSI_primary_models=dict(Counter(r['final_primary_model'] for r in rows)),device=str(engine.device),elapsed_seconds=time.perf_counter()-start)
    (OUT/'runtime_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (OUT/'validation_predictions.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'representative_failures.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['busi_errors_remaining']},indent=2))


if __name__=='__main__':main()
