import { Info, GitCompareArrows } from 'lucide-react';
import type { MultiModelAnalysis } from '../types/patient';

export function ResearchVerdictSummary({result}:{result?:MultiModelAnalysis}) {
  const v=result?.research_verdict;
  return <section className="l-card l-verdict research-report-section" aria-label="Final AI Research Prediction">
    <div className="l-section-head"><span className="l-eyebrow" style={{margin:0}}>Final AI Research Prediction</span><span className="l-badge l-badge-blue">Research use only</span></div>
    <div className="l-verdict-class">{v?.status==='ok'?v.prediction:v?'Unavailable':'Not recorded'}</div>
    <p className="l-muted">{v?'Provisional model-derived finding':'This older record has no stored research verdict. It has not been reclassified.'}</p>
    {v && <>
      <div className="l-verdict-meta"><div><small>Selected primary model</small><strong>{v.primary_model_id==='uclm'?'UCLM pathology':'BUS-BRA pathology'}</strong><small>{v.training_domain_relation.replaceAll('_',' ')} · origin unverified</small></div><div><small>Primary model score</small><strong>{v.primary_model_score==null?'Unavailable':`${(v.primary_model_score*100).toFixed(1)}%`}</strong><small>Uncalibrated softmax · not clinical probability</small></div></div>
      {v.experimental_fallback && <p className="l-warning">Experimental cross-dataset fallback. Domain validity is unknown. A Normal output is not a reliable negative or verified Normal finding.</p>}
      {v.primary_model_id==='bra' && <p className="l-notice">The primary BUS-BRA model cannot classify Normal. It distinguishes Benign and Malignant only.</p>}
      {v.status==='unavailable' && <p className="l-warning">Required primary inference is unavailable. No other model label or score was substituted. {v.error_code?.replaceAll('_',' ')}</p>}
      <div className="l-disagreement" style={{marginTop:16}}><GitCompareArrows aria-hidden="true"/><p>Secondary-model comparison: <strong>{v.agreement.indicator.replaceAll('_',' ')}</strong>. Descriptive observation only; it does not adjust the result or its score.</p></div>
      {v.secondary_model_observations.some(o=>o.model_id==='busi'&&o.comparison==='disagrees')&&<p className="l-helper" style={{marginTop:8}}>Historical BUSI comparison disagrees. It is excluded from primary selection and the comparison indicator.</p>}
      <details className="l-limitations"><summary>Decision method, observations &amp; limitations</summary><p>{v.method_id} · version {v.method_version}</p><ul>{v.secondary_model_observations.map(o=><li key={o.model_id}>{o.model_id.toUpperCase()}: {o.prediction??'Unavailable'} · {o.role.replaceAll('_',' ')} · {o.comparison.replaceAll('_',' ')}. {o.note}</li>)}</ul><ul>{v.applicability_limitations.map(t=><li key={t}>{t}</li>)}</ul></details>
    </>}
    <div className="l-notice" style={{marginTop:18,display:'flex',gap:9}}><Info size={17} aria-hidden="true"/><p>{result?.consensus.prediction??'No validated consensus available'}. Consensus score: unavailable. This separate policy remains abstained; individual scores are not consensus confidence.</p></div>
  </section>;
}

export function BiradsSummary({result}:{result?:MultiModelAnalysis}) {
  const m=result?.models['BUS-BRA BI-RADS'];
  return <section className="l-card l-birads research-report-section" aria-label="Separate BI-RADS research output"><div><span className="l-eyebrow">Separate task</span><h2>BI-RADS research category</h2><p className="l-muted">BUS-BRA BI-RADS · {m?.status??'Not available'}</p><p className="l-helper">Category prediction only. Never a pathology vote or clinical assessment.</p></div><div><div className="l-birads-value">{m?.prediction??'—'}</div><p className="l-birads-label">{m?.score!=null?`${(m.score*100).toFixed(1)}% model score`:'Score unavailable'}</p><p className="l-birads-label">Uncalibrated</p></div></section>;
}
