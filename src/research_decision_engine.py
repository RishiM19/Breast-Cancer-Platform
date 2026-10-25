"""Transparent single-model selection, never a validated ensemble or medical diagnosis."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

METHOD_ID = 'domain_aware_research_prediction_v1'
METHOD_VERSION = '1.0.0'
DOMAINS = ('unknown', 'UCLM', 'BUSI', 'BUS-BRA')
MODEL_NAMES = {'busi':'BUSI', 'uclm':'UCLM', 'bra':'BUS-BRA'}
TRAINING_DOMAINS = {'busi':'BUSI', 'uclm':'UCLM', 'bra':'BUS-BRA'}
WARNING = 'Research use only. Provisional model-derived finding; not a diagnosis or clinically validated consensus.'


class DecisionModel(BaseModel):
    model_config = ConfigDict(extra='forbid',allow_inf_nan=False)


class SecondaryObservation(DecisionModel):
    model_id: Literal['busi','uclm','bra']
    role: Literal['historical_comparison','secondary_pathology_observation']
    status: Literal['ok','missing','not_run','error','absent']
    prediction: Literal['Normal','Benign','Malignant'] | None
    score: float | None = Field(ge=0,le=1)
    comparison: Literal['agrees','disagrees','not_comparable','unavailable']
    note: str


class AgreementIndicators(DecisionModel):
    indicator: Literal['agreement','disagreement','not_comparable','unavailable']
    agreeing_model_ids: list[str]
    disagreeing_model_ids: list[str]
    not_comparable_model_ids: list[str]
    unavailable_model_ids: list[str]
    semantics: str = 'Descriptive label comparison only; no voting, averaging or confidence adjustment. BUSI historical and BI-RADS excluded from these lists.'


class ResearchVerdict(DecisionModel):
    method_id: Literal['domain_aware_research_prediction_v1'] = METHOD_ID
    method_version: Literal['1.0.0'] = METHOD_VERSION
    status: Literal['ok','unavailable']
    prediction: Literal['Normal','Benign','Malignant'] | None
    primary_model_id: Literal['uclm','bra']
    primary_model_score: float | None = Field(ge=0,le=1)
    score_semantics: Literal['uncalibrated_softmax'] = 'uncalibrated_softmax'
    requested_domain: Literal['unknown','UCLM','BUSI','BUS-BRA']
    training_domain_relation: Literal['native_research','cross_dataset','unknown']
    source_domain_verification: Literal['caller_declared_unverified'] = 'caller_declared_unverified'
    experimental_fallback: bool
    supported_primary_classes: list[str]
    secondary_model_observations: list[SecondaryObservation]
    agreement: AgreementIndicators
    applicability_limitations: list[str]
    research_only: Literal[True] = True
    research_warning: Literal[WARNING] = WARNING
    error_code: str | None = None

    @model_validator(mode='after')
    def coherent(self):
        if self.status=='ok':
            if self.prediction is None or self.primary_model_score is None or self.error_code is not None:
                raise ValueError('Successful verdict requires a primary label and its own score')
        elif self.prediction is not None or self.primary_model_score is not None:
            raise ValueError('Unavailable verdict cannot invent a label or score')
        return self


def _record(models, model_id):
    record=models.get(MODEL_NAMES[model_id])
    return record.model_dump() if hasattr(record,'model_dump') else record or {}


def compute_research_verdict(models, source_domain):
    if source_domain not in DOMAINS:raise ValueError('Unsupported source domain')
    primary_id='bra' if source_domain=='BUS-BRA' else 'uclm'
    classes=['Benign','Malignant'] if primary_id=='bra' else ['Normal','Benign','Malignant']
    primary=_record(models,primary_id)
    available=primary.get('status')=='ok' and primary.get('prediction') in classes
    label=primary.get('prediction') if available else None
    score=primary.get('score') if available else None
    if available and (score is None or score!=primary.get('probabilities',{}).get(label)):
        raise ValueError('Primary model score must belong to its own predicted label')
    relation='unknown' if source_domain=='unknown' else 'native_research' if source_domain==TRAINING_DOMAINS[primary_id] else 'cross_dataset'
    limitations=[WARNING,'Dataset origin is self-declared and unverified. Domain selection does not validate a medical prediction.',
        'This is a transparent single-model selection policy, not a weighted ensemble. Softmax scores are uncalibrated.',
        'Agreement and disagreement are observations, not calibrated confidence; original predictions remain unchanged.']
    if primary_id=='bra':limitations.append('BUS-BRA is binary: the primary model cannot classify Normal. A Benign result is not a Normal finding.')
    if source_domain in ('unknown','BUSI'):
        limitations.append('Experimental UCLM three-class fallback outside verified native applicability. A Normal output is not a reliable negative or verified Normal finding on this upload.')
    if source_domain=='BUSI':limitations.append('BUSI is retained only as a historical comparison because its validation contains duplication and label conflicts; it never votes or supplies the primary score.')
    observations=[];agree=[];disagree=[];not_comparable=[];unavailable=[]
    for model_id in ('busi','uclm','bra'):
        if model_id==primary_id:continue
        record=_record(models,model_id);ok=record.get('status')=='ok'
        comparison='unavailable'
        if available and ok:
            comparison='not_comparable' if label=='Normal' and model_id=='bra' else 'agrees' if record['prediction']==label else 'disagrees'
        historical=model_id=='busi'
        observations.append(SecondaryObservation(model_id=model_id,role='historical_comparison' if historical else 'secondary_pathology_observation',
            status=record.get('status','absent'),prediction=record.get('prediction') if ok else None,score=record.get('score') if ok else None,
            comparison=comparison,note='Historical comparison only; never a vote or primary model.' if historical else
                'Binary model cannot assess the primary Normal class.' if comparison=='not_comparable' else 'Separate model observation; never changes primary label or score.'))
        if historical:continue
        if comparison=='agrees':agree.append(model_id)
        elif comparison=='disagrees':disagree.append(model_id)
        elif comparison=='not_comparable':not_comparable.append(model_id)
        else:unavailable.append(model_id)
    indicator='disagreement' if disagree else 'agreement' if agree else 'not_comparable' if not_comparable else 'unavailable'
    return ResearchVerdict(status='ok' if available else 'unavailable',prediction=label,primary_model_id=primary_id,
        primary_model_score=score,requested_domain=source_domain,training_domain_relation=relation,
        experimental_fallback=source_domain in ('unknown','BUSI'),supported_primary_classes=classes,
        secondary_model_observations=observations,agreement=AgreementIndicators(indicator=indicator,agreeing_model_ids=agree,
            disagreeing_model_ids=disagree,not_comparable_model_ids=not_comparable,unavailable_model_ids=unavailable),
        applicability_limitations=limitations,error_code=None if available else 'primary_model_'+primary.get('status','absent'))


def decision_policy():
    return dict(method_id=METHOD_ID,method_version=METHOD_VERSION,routing={d:('bra' if d=='BUS-BRA' else 'uclm') for d in DOMAINS},
        experimental_fallback_domains=['unknown','BUSI'],historical_only_model='busi',weighted_ensemble=False,
        consensus_policy='Universal validated-consensus abstention remains separate.',research_warning=WARNING,schema=ResearchVerdict.model_json_schema())
