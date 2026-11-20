import { useEffect, useRef, useState } from 'react';
import { Activity, ArrowLeft, Printer } from 'lucide-react';
import type { ModelPrediction, PatientRecord } from '../types/patient';
import { displayRecordId } from '../lib/inferenceAdapter';
import { FeedbackModal } from './FeedbackModal';
import { ReviewSummary } from './ReviewSummary';
import { ExploratorySegmentation } from './ExploratorySegmentation';
import { ResearchVerdictSummary, BiradsSummary } from './ResearchVerdictSummary';
import { diagnosticReportStyles } from './diagnosticReportStyles';

interface Props { patient: PatientRecord; onBack: () => void; onReviewSaved?: () => void | Promise<void> }

// Category terminology only; no clinical recommendations are inferred.
// https://accreditationsupport.acr.org/support/solutions/articles/11000067043-reporting-breast-ultrasound
const biradsDescriptions: Record<string, string> = {
  '2': 'Benign findings', '3': 'Probably benign findings',
  '4': 'Suspicious abnormality', '5': 'Highly suggestive of malignancy',
};
const pathologyNames = ['BUSI', 'UCLM', 'BUS-BRA'] as const;

function formatDate(value?: string) {
  if (!value) return 'Not recorded';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'Date unavailable' : date.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', timeZoneName: 'short',
  });
}

function formatScore(value?: number | null) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
    ? `${(value * 100).toFixed(1)}%` : 'Unavailable';
}

function modelAvailable(model?: ModelPrediction) {
  return !!model && (model.status === 'ok' || model.status == null);
}

function SectionHeading({ number, title, detail }: { number: string; title: string; detail?: string }) {
  return <div className="lr-section-heading"><span aria-hidden="true">{number}</span><div><h2>{title}</h2>{detail && <p>{detail}</p>}</div></div>;
}

export function DiagnosticReport({ patient, onBack, onReviewSaved }: Props) {
  const [review, setReview] = useState(false);
  const [heatmap, setHeatmap] = useState(false);
  const reportRef = useRef<HTMLElement>(null);
  const a = patient.analysis;
  const versioned = a.recordKind === 'research';
  const result = a.multiModel;
  const verdict = versioned ? result?.research_verdict : undefined;
  const birads = result?.models['BUS-BRA BI-RADS'];
  const biradsLabel = birads?.status === 'ok' ? birads.prediction : null;
  const hasCam = !!patient.heatmapImage && result?.attribution?.status === 'ok';
  const reviewRecorded = a.reviewStatus === 'reviewed';
  const primaryFinding = verdict?.status === 'ok' ? verdict.prediction : null;
  const impression = !versioned
    ? 'The original stored finding is reproduced below. This historical record has not been reclassified.'
    : !primaryFinding
      ? 'No stored primary research finding is available for this record. Individual model results are listed separately.'
      : primaryFinding === 'Malignant'
        ? 'The selected model assigned a Malignant image class. This is a provisional research finding; cancer is not confirmed by this report.'
        : primaryFinding === 'Benign'
          ? 'The selected model assigned a Benign image class. This research output does not establish a clinical diagnosis or exclude malignancy.'
          : 'The selected model assigned a Normal image class. This research output does not establish a reliable negative finding or exclude malignancy.';

  // Include disclosures for native printing, including Ctrl+P, then restore the
  // reader's screen state. No state outside the mounted report is modified.
  useEffect(() => {
    let priorDetails: Array<[HTMLDetailsElement, boolean]> | null = null;
    const beforePrint = () => {
      if (priorDetails) return;
      priorDetails = Array.from(reportRef.current?.querySelectorAll('details') ?? []).map(detail => [detail, detail.open]);
      priorDetails.forEach(([detail]) => { detail.open = true; });
    };
    const afterPrint = () => {
      priorDetails?.forEach(([detail, open]) => { detail.open = open; });
      priorDetails = null;
    };
    window.addEventListener('beforeprint', beforePrint);
    window.addEventListener('afterprint', afterPrint);
    return () => {
      afterPrint();
      window.removeEventListener('beforeprint', beforePrint);
      window.removeEventListener('afterprint', afterPrint);
    };
  }, [patient.id]);

  return <div className="l-report lumina-report">
    <style>{diagnosticReportStyles}</style>
    <div className="lr-workspace">
      <nav className="research-report-actions lr-actions" aria-label="Report actions">
        <button className="l-button" onClick={onBack}><ArrowLeft size={16} aria-hidden="true"/>Back to Patient List</button>
        <div className="lr-action-group"><button className="l-button" onClick={() => setReview(true)}>Record human review</button><button className="l-button l-button-primary" onClick={() => window.print()}><Printer size={16} aria-hidden="true"/>Print report</button></div>
        <p className="lr-print-help">Use your browser’s print dialog to print or choose Save as PDF. PDF export is not verified for every browser.</p>
      </nav>

      <article className="lr-document" ref={reportRef} aria-label="Breast ultrasound report">
        <header className="lr-header">
          <div className="lr-letterhead"><div className="l-brand"><div className="l-logo"><Activity aria-hidden="true"/></div><div><strong>Lumina</strong><small>Breast AI Research</small></div></div><div className="lr-document-kind"><span>BREAST IMAGING</span><strong>{versioned ? 'Research Inference Report' : 'Historical Analysis Record'}</strong><small>{reviewRecorded ? 'Human review recorded · identity unverified' : 'Awaiting independent human review'}</small></div></div>
          <h1>Breast Ultrasound Report</h1>
          <p className="lr-subtitle">{versioned ? 'Patient summary, image findings and independent model assessments' : 'Preserved patient record and original stored findings'}</p>
          <dl className="lr-patient-grid">
            <div className="lr-patient-name"><dt>Patient name</dt><dd>{patient.name.trim() || 'Not recorded'}</dd></div>
            <div><dt>Age</dt><dd>{patient.age.trim() ? `${patient.age} years` : 'Not recorded'}</dd></div>
            <div><dt>Record number</dt><dd>{displayRecordId(patient.id)}</dd></div>
            <div><dt>Record created</dt><dd>{formatDate(patient.date)}</dd></div>
            <div><dt>Examination</dt><dd>Breast ultrasound image analysis</dd></div>
            <div><dt>Dataset origin</dt><dd>{result?.source_domain ?? 'Unknown'} <span className="lr-muted">(unverified)</span></dd></div>
            <div><dt>Laterality / location</dt><dd>Not recorded</dd></div>
            <div><dt>Clinical sign-off</dt><dd>Not established by this system</dd></div>
          </dl>
          <p className="lr-research-note"><strong>Research use only.</strong> No model or ensemble has established clinical deployment readiness. Human identities and assessments are unverified.</p>
        </header>

        {!versioned && <aside className="lr-historical" role="note">Historical record: stored labels and text are retained as historical information. Provenance, calibration and diagnostic validity may be unavailable. This record has not been upgraded to the research contract.</aside>}

        <section className="lr-section lr-assessment">
          <SectionHeading number="01" title="Assessment & impression" detail="The primary finding and BI-RADS category are separate research outputs."/>
          {versioned ? <div className="lr-assessment-grid"><ResearchVerdictSummary result={result}/><div className="lr-birads-column"><BiradsSummary result={result}/>
            <p className="lr-birads-meaning"><strong>{biradsLabel && biradsDescriptions[biradsLabel] ? biradsDescriptions[biradsLabel] : 'No category interpretation available'}</strong><span>{biradsLabel ? 'Standard category terminology; this model output is not a radiologist-assigned BI-RADS assessment.' : 'A BI-RADS category was not produced. It has not been inferred from a pathology label.'}</span></p>
            {birads?.status === 'ok' && <dl className="lr-category-scores" aria-label="BI-RADS class scores">{Object.entries(birads.probabilities).map(([label, value]) => <div key={label}><dt>Category {label}</dt><dd>{formatScore(value)}</dd></div>)}</dl>}
            <p className="lr-small">Supported categories: 2, 3, 4 and 5. The model does not distinguish 4A, 4B or 4C. Model scores are uncalibrated.</p>
          </div></div> : <div className="lr-assessment-grid"><div className="lr-stored-finding"><span>Original stored finding</span><strong>{a.prediction || 'Unavailable'}</strong><p>Stored score: {a.confidence || 'Unavailable (historical)'}</p></div>{birads && <div className="lr-birads-column"><BiradsSummary result={result}/><p className="lr-small">Historical BI-RADS output retained as stored; provenance and diagnostic validity may be unavailable.</p><dl className="lr-category-scores" aria-label="Historical BI-RADS class scores">{Object.entries(birads.probabilities ?? {}).map(([label, value]) => <div key={label}><dt>Category {label}</dt><dd>{formatScore(value)}</dd></div>)}</dl></div>}</div>}
          <div className="lr-impression"><h3>Report impression</h3><p>{impression}</p><p><strong>Specific cancer type / histology:</strong> Not determined by these image classifiers. No pathology-confirmed cancer type is supplied in this record.</p><p><strong>Clinical recommendations:</strong> No patient-specific management recommendation is generated from these research scores. Any recorded human assessment appears in section 06.</p></div>
        </section>

        <section className="lr-section">
          <SectionHeading number="02" title="Clinical context & examination details"/>
          <dl className="lr-facts">
            <div><dt>Clinical indication / history</dt><dd>{a.reportDetails.Indication || 'Not recorded'}</dd></div>
            <div><dt>Technique</dt><dd>Analysis of the stored ultrasound image. Acquisition technique and scan protocol are not recorded.</dd></div>
            <div><dt>Comparison studies</dt><dd>Not supplied in this record</dd></div>
            <div><dt>Tissue composition</dt><dd>{a.reportDetails.Composition || 'Not assessed'}</dd></div>
            <div><dt>Lesion size</dt><dd>{!versioned && a.lesionSize !== 'N/A' ? `${a.lesionSize} (historical stored value; unverified)` : 'Not measured by these classifiers'}</dd></div>
            <div><dt>Lesion shape / margins / orientation</dt><dd>Not assessed by these classifiers</dd></div>
            <div><dt>Tumour grade / cancer stage</dt><dd>Not assessed by these classifiers</dd></div>
            <div><dt>Lymph node involvement</dt><dd>Not assessed by these classifiers</dd></div>
            <div><dt>Receptor status / molecular subtype</dt><dd>Not assessed by these classifiers</dd></div>
            <div><dt>Biopsy / histopathology</dt><dd>No structured result supplied</dd></div>
          </dl>
        </section>

        <section className="lr-section">
          <SectionHeading number="03" title="Individual model findings" detail="Each label and score belongs to its own model. These results are not combined into a diagnosis."/>
          <div className="lr-table-wrap"><table className="lr-model-table"><caption>Individual pathology research predictions</caption><thead><tr><th scope="col">Model</th><th scope="col">Finding</th><th scope="col">Model score</th><th scope="col">Class scores</th></tr></thead><tbody>
            {pathologyNames.map(name => {
              const model = result?.models[name];
              const available = modelAvailable(model);
              return <tr key={name}><th scope="row">{name}<small>{model?.status ?? (model ? 'Historical' : 'Not recorded')}</small><small>{model?.eligibility?.domain_relation.replaceAll('_', ' ') ?? 'Domain unverified'}</small></th><td>{available ? model?.prediction ?? 'Unavailable' : 'Unavailable'}</td><td>{available ? formatScore(model?.score ?? model?.confidence) : 'Unavailable'}</td><td><dl className="lr-distribution">{['Normal', 'Benign', 'Malignant'].map(label => <div key={label}><dt>{label}</dt><dd>{!available ? 'Unavailable' : model?.probabilities?.[label] == null ? name === 'BUS-BRA' && label === 'Normal' ? 'Not supported' : 'Unavailable' : formatScore(model.probabilities[label])}</dd></div>)}</dl></td></tr>;
            })}
          </tbody></table></div>
          <div className="lr-table-notes"><p><strong>BUSI:</strong> Historical comparison only; excluded from automated consensus and primary selection.</p><p><strong>BUS-BRA pathology:</strong> Distinguishes Benign and Malignant; does not support Normal. BUS-BRA BI-RADS is a separate task, shown in section 01.</p><p><strong>Consensus:</strong> No validated consensus available. Consensus score: unavailable. Agreement and disagreement do not establish a clinical diagnosis.</p></div>
        </section>

        <section className="lr-section">
          <SectionHeading number="04" title="Image evidence" detail="Stored image, genuine class attribution and available exploratory segmentation."/>
          <div className="lr-evidence-grid">
            <figure className="lr-evidence lr-screen-image"><div className="lr-image-frame">{patient.originalImage ? <img src={heatmap && hasCam ? patient.heatmapImage : patient.originalImage} alt={heatmap && hasCam ? 'Genuine BUS-BRA pathology Grad-CAM attribution' : 'Original research ultrasound'}/> : <p>No stored image available.</p>}</div><figcaption>{heatmap && hasCam ? 'BUS-BRA pathology Grad-CAM · class attribution' : 'Original stored ultrasound image'}</figcaption>
              <button className="l-button research-report-actions" disabled={!hasCam} aria-pressed={heatmap && hasCam} onClick={() => setHeatmap(value => !value)}>{hasCam ? heatmap ? 'Show original image' : 'Show Grad-CAM attribution' : 'Grad-CAM unavailable'}</button>
            </figure>
            <figure className="lr-evidence lr-print-image"><div className="lr-image-frame">{patient.originalImage ? <img src={patient.originalImage} alt="Original ultrasound for printing"/> : <p>No stored image available.</p>}</div><figcaption>Original stored ultrasound image</figcaption></figure>
            {hasCam && <figure className="lr-evidence lr-print-image"><div className="lr-image-frame"><img src={patient.heatmapImage} alt="Stored Grad-CAM attribution for printing"/></div><figcaption>BUS-BRA pathology Grad-CAM · class attribution</figcaption></figure>}
            <div className="lr-evidence-note"><h3>Evidence interpretation</h3><dl><div><dt>Attribution status</dt><dd>{result?.attribution?.status ?? 'Not recorded'}</dd></div><div><dt>Attribution target</dt><dd>{result?.attribution?.target_label ?? 'Not available'}</dd></div><div><dt>Attribution model</dt><dd>{result?.attribution?.model_id ?? 'Not recorded'}</dd></div></dl><p>Grad-CAM shows image regions associated with a class output. It is not segmentation, a measurement of tumour extent or a validated clinical explanation.</p><p>{result?.attribution?.note}</p></div>
          </div>
          <ExploratorySegmentation result={result?.segmentation} originalImage={patient.originalImage}/>
        </section>

        <section className="lr-section">
          <SectionHeading number="05" title={versioned ? 'Stored findings & report notes' : 'Stored historical text'} detail="Original saved text is reproduced without replacing it with generated clinical findings."/>
          <dl className="lr-narrative">{Object.entries(a.reportDetails).map(([name, text]) => <div key={name}><dt>{name === 'BI_RADS' ? 'BI-RADS — stored note' : name}</dt><dd>{text || 'Not recorded'}</dd></div>)}</dl>
        </section>

        <section className="lr-section lr-review-section">
          <SectionHeading number="06" title="Independent human review" detail="Recorded assessments remain separate from the original AI results."/>
          <ReviewSummary analysis={a}/>
          <p className="lr-small">A recorded review does not verify the reviewer’s identity, confirm a diagnosis or constitute a signed clinical report.</p>
        </section>

        <section className="lr-section lr-provenance">
          <SectionHeading number="07" title="Record provenance & limitations"/>
          <dl className="lr-facts"><div><dt>Full record identifier</dt><dd className="lr-mono">{patient.id}</dd></div><div><dt>Inference contract</dt><dd>{result?.contract_version ?? 'Unavailable (historical)'}</dd></div><div><dt>Record status</dt><dd>{versioned ? 'Stored research output' : 'Preserved historical record'}</dd></div><div><dt>Decision method</dt><dd>{verdict ? `${verdict.method_id} / ${verdict.method_version}` : 'No stored research verdict; not reclassified'}</dd></div></dl>
          <details className="lr-checkpoints"><summary>Model checkpoints, applicability & limitations</summary><div className="lr-checkpoint-list">{Object.entries(result?.models ?? {}).map(([name, model]) => <div key={name}><h3>{name}</h3><p>{model?.checkpoint?.filename ?? 'Checkpoint unavailable'} · {model?.checkpoint?.version ?? 'Version unavailable'}</p><p className="lr-mono">SHA-256: {model?.checkpoint?.sha256 ?? 'Unavailable'}</p><p>Verification: {model?.checkpoint?.verification ?? 'Unavailable'}. Class order: {model?.ordered_class_labels?.join(', ') ?? 'Not recorded'}.</p><p>{model?.eligibility?.reason ?? model?.note ?? 'Applicability not recorded.'}</p>{model?.error_code && <p>Model error: {model.error_code.replaceAll('_', ' ')}</p>}</div>)}</div>{result?.research_limitations && <ul>{result.research_limitations.map((text, index) => <li key={index}>{text}</li>)}</ul>}</details>
          <p className="lr-small">All model scores are uncalibrated outputs, not probabilities of disease. This report does not establish cancer type, stage, grade or clinical management. Older records retain their original provenance and are not reclassified.</p>
          {verdict && <div className="lr-print-method"><h3>Primary selection method & observations</h3><p>{verdict.method_id} · {verdict.method_version}</p><ul>{verdict.secondary_model_observations.map(observation => <li key={observation.model_id}>{observation.model_id.toUpperCase()}: {observation.prediction ?? 'Unavailable'} · {observation.role.replaceAll('_', ' ')} · {observation.comparison.replaceAll('_', ' ')}. {observation.note}</li>)}</ul><ul>{verdict.applicability_limitations.map((text, index) => <li key={index}>{text}</li>)}</ul></div>}
        </section>

        <footer className="lr-footer"><span>Lumina · Breast AI Research</span><span>{displayRecordId(patient.id)} · End of report</span></footer>
      </article>
    </div>
    {review && <FeedbackModal caseId={patient.id} originalPrediction={a.prediction} originalConfidence={a.confidence} researchVerdict={result?.research_verdict} onClose={() => setReview(false)} onSuccess={onReviewSaved}/>}
  </div>;
}
