/** Multi-section structured report from the backend (Python keys preserved). */
export interface ClinicalReportDetails {
  Indication: string;
  Composition: string;
  Findings: string;
  Impression: string;
  BI_RADS: string;
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
