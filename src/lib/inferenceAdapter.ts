import type { AnalysisData, MultiModelAnalysis, PatientAnalysis, PatientRecord } from '../types/patient';
import { parseReportDetails } from '../types/patient';

export const ABSTENTION = 'No validated consensus available';

export function summarizeRecords(records: PatientRecord[], now = new Date()) {
  const latest=records.find(p=>p.analysis.recordKind==='research' && p.analysis.multiModel?.contract_version);
  return {
    scansToday: records.filter(p=>new Date(p.date).toDateString()===now.toDateString()).length,
    historical: records.filter(p=>p.analysis.recordKind==='historical').length,
    reviewed: records.filter(p=>p.analysis.reviewStatus==='reviewed').length,
    latestAvailable: latest?.analysis.multiModel ? Object.values(latest.analysis.multiModel.models).filter(m=>m?.status==='ok').length : null,
  };
}

export function predictionToColor(prediction: string): 'red' | 'green' | 'blue' {
  const p = prediction.toLowerCase();
  if (p.includes('malign')) return 'red';
  if (p.includes('benign')) return 'green';
  return 'blue';
}

function imageSrc(raw: unknown): string {
  const s = raw == null ? '' : String(raw).trim();
  if (!s || s.startsWith('data:') || s.startsWith('blob:')) return s;
  return `data:image/jpeg;base64,${s}`;
}

/** Fail closed on unknown versions or misleading consensus/score associations. */
export function readResearchResult(data: Record<string, unknown>): MultiModelAnalysis {
  if (!['1.0.0','1.1.0','1.2.0'].includes(String(data.contract_version))) throw new Error('Unsupported research inference contract');
  const result = data as unknown as MultiModelAnalysis;
  if (data.contract_version === '1.1.0' || data.contract_version === '1.2.0') {
    const s=result.segmentation; const im=data.image as {width:number;height:number}|undefined;
    if (!s || s.task!=='segmentation' || s.model_id!=='uclm_v3_binary' || !['ok','missing','not_run','error'].includes(s.status)
      || s.requested_domain!==result.source_domain || s.image_width!==im?.width || s.image_height!==im?.height
      || s.threshold!==0.5 || s.source_domain_verification!=='caller_declared_unverified') throw new Error('Invalid segmentation record');
    if(s.status==='ok') {
      if(s.requested_domain!=='UCLM' || s.checkpoint_verification!=='verified'
        || s.checkpoint_sha256!=='7acf7f5331e92389537ac85aac0c91ed848ece7455a4b688d49ce4da4d6ea957'
        || !s.predicted_mask?.startsWith('data:image/png;base64,') || !s.overlay?.startsWith('data:image/png;base64,')
        || !s.foreground_probability_npy_base64 || typeof s.empty_mask!=='boolean') throw new Error('Invalid segmentation outputs');
    } else if(s.predicted_mask!==null || s.overlay!==null || s.foreground_probability_npy_base64!==null || s.empty_mask!==null) throw new Error('Unavailable segmentation has outputs');
    if(s.requested_domain!=='UCLM' && s.status!=='not_run') throw new Error('Ineligible segmentation');
  } else if(result.segmentation) throw new Error('Segmentation requires contract 1.1.0');
  if(data.contract_version==='1.2.0') {
    const v=result.research_verdict;const id=result.source_domain==='BUS-BRA'?'bra':'uclm';
    const primary=result.models?.[id==='bra'?'BUS-BRA':'UCLM'];
    const relation=result.source_domain==='unknown'?'unknown':result.source_domain===(id==='bra'?'BUS-BRA':'UCLM')?'native_research':'cross_dataset';
    if(!v || v.method_id!=='domain_aware_research_prediction_v1' || v.method_version!=='1.0.0'
      || v.primary_model_id!==id || v.requested_domain!==result.source_domain || v.score_semantics!=='uncalibrated_softmax'
      || v.research_only!==true || v.source_domain_verification!=='caller_declared_unverified'
      || v.experimental_fallback!==['unknown','BUSI'].includes(String(result.source_domain)) || v.training_domain_relation!==relation
      || !Array.isArray(v.applicability_limitations) || v.applicability_limitations.length===0 || !v.agreement
      || !['agreement','disagreement','not_comparable','unavailable'].includes(v.agreement.indicator)
      || !Array.isArray(v.secondary_model_observations)) throw new Error('Invalid research verdict');
    if(v.status==='ok') {
      if(primary?.status!=='ok' || !['Normal','Benign','Malignant'].includes(String(v.prediction)) || v.prediction!==primary.prediction
        || v.primary_model_score!==primary.score || typeof v.primary_model_score!=='number' || !Number.isFinite(v.primary_model_score)
        || (id==='bra'&&v.prediction==='Normal')) throw new Error('Invalid research verdict primary label/score');
    } else if(v.status!=='unavailable' || v.prediction!==null || v.primary_model_score!==null || primary?.status==='ok') throw new Error('Invalid unavailable research verdict');
  } else if(result.research_verdict) throw new Error('Research verdict requires contract 1.2.0');
  const c = result.consensus;
  if (!c || c.prediction !== ABSTENTION || c.status !== 'abstained' || c.confidence !== null || c.score !== null) {
    throw new Error('Invalid consensus or score association');
  }
  for (const name of ['BUSI', 'UCLM', 'BUS-BRA', 'BUS-BRA BI-RADS'] as const) {
    const model = result.models?.[name];
    if (!model || !model.status || !['ok', 'missing', 'not_run', 'error'].includes(model.status)
      || model.eligibility?.consensus_eligible !== false) throw new Error('Invalid model record');
    if (model.status === 'ok') {
      const labels = model.ordered_class_labels;
      const index = model.predicted_class_index;
      if (!labels || index == null || labels[index] !== model.prediction || model.prediction == null
        || typeof model.score !== 'number' || !Number.isFinite(model.score) || model.score < 0 || model.score > 1
        || model.score !== model.probabilities[model.prediction] || model.confidence !== model.score) {
        throw new Error('Invalid model label/score association');
      }
    } else if (model.prediction !== null || model.score !== null || model.confidence !== null) {
      throw new Error('Unavailable model has a prediction or score');
    }
  }
  return result;
}

/** Display only persisted verdicts. Never recompute a new finding for older rows. */
export function displayResearchFinding(patient: PatientRecord):string {
  const v=patient.analysis.recordKind==='research'?patient.analysis.multiModel?.research_verdict:undefined;
  return v ? v.status==='ok'?v.prediction!:'Research finding unavailable' : patient.analysis.recordKind==='historical'?patient.analysis.prediction:'No stored research verdict';
}

export function displayRecordId(id:string):string {
  if(id.length<=18)return id;
  const value=id.replace(/^RESEARCH-/,'');return `LUM-${value.slice(0,8).toUpperCase()}-${value.slice(-4).toUpperCase()}`;
}

export function mapPredictResponseToAnalysis(data: Record<string, unknown>): AnalysisData {
  const versioned = data.contract_version != null;
  const multiModel = versioned ? readResearchResult(data)
    : data.models && data.consensus ? ({ models: data.models, consensus: data.consensus } as MultiModelAnalysis) : undefined;
  // Legacy multi-model votes are unvalidated, and their top score belonged to
  // BUSI. Keep individual records without promoting that vote or borrowing it.
  const prediction = multiModel ? ABSTENTION : typeof data.prediction === 'string' ? data.prediction : ABSTENTION;
  const score = !multiModel && prediction !== ABSTENTION && typeof data.confidence === 'number' && Number.isFinite(data.confidence)
    && data.confidence >= 0 && data.confidence <= 1 ? data.confidence : null;
  return {
    prediction, predictionColor: predictionToColor(prediction),
    confidence: score == null ? 'Unavailable' : `${Math.round(score * 100)}% (uncalibrated)`,
    tumorGrade: 'N/A', nodeStatus: 'Unknown', lesionSize: 'N/A', subtype: 'N/A',
    originalImage: '', heatmapImage: versioned && multiModel?.attribution?.status === 'ok' ? imageSrc(data.heatmap_base64) : '',
    reportDetails: versioned ? parseReportDetails(data.report_details) : parseReportDetails(null), multiModel,
    recordKind: versioned ? 'research' : 'historical', reviewStatus: 'unreviewed',
  };
}

export function analysisDataToPatientAnalysis(data: AnalysisData): PatientAnalysis {
  return {
    prediction: data.prediction, predictionColor: data.predictionColor, confidence: data.confidence,
    tumorGrade: data.tumorGrade, nodeStatus: data.nodeStatus, lesionSize: data.lesionSize,
    subtype: data.subtype, notes: data.notes ?? '', reportDetails: data.reportDetails, multiModel: data.multiModel,
    recordKind: data.recordKind, reviewStatus: data.reviewStatus, humanReview: data.humanReview, reviewHistory: data.reviewHistory, storageWarnings: data.storageWarnings,
  };
}

export interface ApiPatientRow {
  id: string; name: string; age: string; date: string; prediction: string; confidence: number;
  size_cm: number; stage: string; report_details: unknown; original_image: string; heatmap_image: string;
  multi_model_result?: MultiModelAnalysis;
  storage_warnings?: string[];
}

export function mapApiPatientToPatientRecord(row: ApiPatientRow): PatientRecord {
  const rawMulti = row.multi_model_result;
  // Older database rows may carry {} rather than a multi-model result.
  let multiModel = rawMulti?.contract_version || (rawMulti?.models && rawMulti?.consensus) ? rawMulti : undefined;
  let versioned=false;
  const warnings=[...(row.storage_warnings??[])];
  if(multiModel?.contract_version!=null){
    try { readResearchResult(multiModel as unknown as Record<string,unknown>);versioned=true; }
    catch { warnings.push('Unsupported or invalid stored inference contract; original record retained as historical.');multiModel=undefined; }
  }
  const review = rawMulti?.human_review?.status === 'recorded' ? rawMulti.human_review : undefined;
  const prediction = versioned ? ABSTENTION : row.prediction;
  const confidence = versioned || prediction === ABSTENTION ? 'Unavailable' : multiModel?.models
    ? 'Unavailable (legacy consensus score)' : 'Unavailable (historical)';
  return {
    id: row.id, name: row.name, age: row.age, date: row.date,
    originalImage: imageSrc(row.original_image),
    heatmapImage: !versioned || multiModel?.attribution?.status === 'ok' ? imageSrc(row.heatmap_image) : '',
    analysis: {
      prediction, predictionColor: predictionToColor(prediction), confidence,
      tumorGrade: 'N/A', nodeStatus: 'Unknown', lesionSize: row.size_cm > 0 ? `${row.size_cm} cm` : 'N/A',
      subtype: row.stage || 'N/A', notes: review?.notes ?? '', reportDetails: parseReportDetails(row.report_details), multiModel,
      recordKind: versioned ? 'research' : 'historical', reviewStatus: review ? 'reviewed' : 'unreviewed',
      humanReview: review, reviewHistory: Array.isArray(rawMulti?.review_history) ? rawMulti.review_history : undefined,
      storageWarnings: warnings,
    },
  };
}
