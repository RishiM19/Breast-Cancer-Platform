/** Multi-section structured report from the backend (Python keys preserved). */
export interface ClinicalReportDetails {
  Indication: string;
  Composition: string;
  Findings: string;
  Impression: string;
  BI_RADS: string;
}

export type ModelProbabilities = Record<string, number | null>;

export interface ModelPrediction {
  prediction: string | null;
  confidence: number | null;
  probabilities: ModelProbabilities;
  task: string;
  note?: string;
  model_id?: 'busi' | 'uclm' | 'bra' | 'birads';
  status?: 'ok' | 'missing' | 'not_run' | 'error';
  score?: number | null;
  predicted_class_index?: number | null;
  ordered_class_labels?: string[];
  source_training_domain?: string;
  checkpoint?: { filename: string; version: string; sha256: string | null; expected_sha256: string; verification: string };
  eligibility?: { requested_domain: string; domain_relation: string; research_task_supported: boolean; consensus_eligible: false; reason: string };
  error_code?: string | null;
}

export interface ConsensusResult {
  prediction: string;
  agreement_count: number;
  model_count?: number;
  participating_models?: number;
  agreement_ratio?: number;
  status: string;
  note?: string;
  confidence?: number | null;
  score?: number | null;
  score_semantics?: string;
}

export interface MultiModelAnalysis {
  research_verdict?: ResearchVerdict;
  segmentation?: SegmentationResult;
  contract_version?: string;
  source_domain?: string;
  research_limitations?: string[];
  attribution?: { status: string; model_id: string; target_label: string | null; note: string };
  human_review?: HumanReview;
  review_history?: HumanReview[];
  consensus: ConsensusResult;
  models: {
    BUSI: ModelPrediction;
    UCLM: ModelPrediction;
    'BUS-BRA': ModelPrediction;
    'BUS-BRA BI-RADS'?: ModelPrediction;
  };
}

export interface ResearchVerdict {
  method_id: 'domain_aware_research_prediction_v1'; method_version: '1.0.0'; status: 'ok'|'unavailable';
  prediction: 'Normal'|'Benign'|'Malignant'|null; primary_model_id: 'uclm'|'bra'; primary_model_score: number|null;
  score_semantics: 'uncalibrated_softmax'; requested_domain: string; training_domain_relation: string;
  source_domain_verification: 'caller_declared_unverified'; experimental_fallback: boolean; supported_primary_classes: string[];
  secondary_model_observations: {model_id:string;role:string;status:string;prediction:string|null;score:number|null;comparison:string;note:string}[];
  agreement: {indicator:string;agreeing_model_ids:string[];disagreeing_model_ids:string[];not_comparable_model_ids:string[];unavailable_model_ids:string[];semantics:string};
  applicability_limitations:string[]; research_only:true; research_warning:string; error_code:string|null;
}

export interface SegmentationResult {
  model_id: 'uclm_v3_binary'; task: 'segmentation'; model_version: string;
  checkpoint_filename: string; expected_checkpoint_sha256: string; checkpoint_sha256: string | null;
  checkpoint_verification: string; source_training_domain: 'UCLM'; requested_domain: string;
  source_domain_verification: 'caller_declared_unverified'; domain_relation: string;
  status: 'ok' | 'missing' | 'not_run' | 'error'; image_width: number; image_height: number;
  original_coordinate_system: string; threshold: number; preprocessing: Record<string,unknown>;
  predicted_mask_encoding: string; overlay_encoding: string; probability_encoding: string;
  predicted_mask: string | null; overlay: string | null; foreground_probability_npy_base64: string | null;
  empty_mask: boolean | null; limitations: string[]; reason: string; error_code: string | null;
}

export function emptyReportDetails(): ClinicalReportDetails {
  return {
    Indication: '',
    Composition: '',
    Findings: '',
    Impression: '',
    BI_RADS: '',
  };
}

export function parseReportDetails(raw: unknown): ClinicalReportDetails {
  if (!raw || typeof raw !== 'object') return emptyReportDetails();
  const o = raw as Record<string, unknown>;
  return {
    Indication: String(o.Indication ?? ''),
    Composition: String(o.Composition ?? ''),
    Findings: String(o.Findings ?? ''),
    Impression: String(o.Impression ?? ''),
    BI_RADS: String(o.BI_RADS ?? ''),
  };
}

/** Stored AI + human follow-up for a single patient scan */
export interface PatientAnalysis {
  prediction: string;
  predictionColor: 'red' | 'green' | 'blue';
  confidence: string;
  tumorGrade: string;
  nodeStatus: string;
  lesionSize: string;
  subtype: string;
  notes: string;
  reportDetails: ClinicalReportDetails;
  multiModel?: MultiModelAnalysis;
  recordKind?: 'research' | 'historical';
  reviewStatus?: 'reviewed' | 'unreviewed';
  humanReview?: HumanReview;
  reviewHistory?: HumanReview[];
  storageWarnings?: string[];
}

export interface HumanReview {
  status: 'recorded'; kind?: string; prediction: string | null; notes: string;
  reviewer_id?: string; identity_verified?: false; created_at?: string; feedback_id?: string;
  declared_confidence?: number | null;
}

/** Global patient record used across dashboard, list, reports, and correction */
export interface PatientRecord {
  id: string;
  name: string;
  age: string;
  date: string;
  /** Object URL or data URL for the original scan */
  originalImage: string;
  /** `data:image/jpeg;base64,...` from backend, or empty when Normal / unavailable */
  heatmapImage: string;
  analysis: PatientAnalysis;
}

/** Same shape as analysis panel / results (notes optional for transient UI state) */
export type AnalysisData = Omit<PatientAnalysis, 'notes'> & {
  notes?: string;
  originalImage: string;
  heatmapImage: string;
};
