import { useState, useRef } from 'react';
import { submitReview, commitReviewOnce } from '../lib/api';
import type { ResearchVerdict } from '../types/patient';
interface FeedbackModalProps {
  caseId: string; originalPrediction: string; originalConfidence: string;
  onClose: () => void; onSuccess?: () => void | Promise<void>;
  researchVerdict?: ResearchVerdict;
}
export function FeedbackModal({ caseId, originalPrediction, originalConfidence, onClose, onSuccess, researchVerdict }: FeedbackModalProps) {
  const [reviewer,setReviewer]=useState('');const [kind,setKind]=useState('reviewed');
  const [prediction,setPrediction]=useState('');const [notes,setNotes]=useState('');const [confidence,setConfidence]=useState('');
  const [busy,setBusy]=useState(false);const [error,setError]=useState('');const [saved,setSaved]=useState(false);
  const commit=useRef({saved:false});
  const save=async(e:React.FormEvent)=>{
    e.preventDefault();setError('');
    if(!reviewer.trim() || !notes.trim() || (kind==='independent_assessment' && !prediction)) {
      setError('Enter a reviewer ID and note, and select an assessment when required.');return;
    }
    setBusy(true);
    try {
      await commitReviewOnce(commit.current,
        ()=>submitReview(caseId,reviewer,notes,kind,kind==='reviewed'?null:prediction||null,confidence===''?null:Number(confidence)/100),
        async()=>{setSaved(true);await onSuccess?.();});onClose();
    }catch(err){setSaved(commit.current.saved);setError(`${commit.current.saved?'Review is saved; retry reload. ':''}${err instanceof Error?err.message:'Review could not be saved.'}`);}
    finally{setBusy(false);}
  };
  return <div className="fixed inset-0 z-50 bg-black/60 overflow-y-auto flex items-center justify-center p-5" role="dialog" aria-modal="true" aria-label="Record human review">
    <form onSubmit={save} className="l-card w-full max-w-xl space-y-4">
      <h2 className="text-xl font-semibold">Record human review</h2>
      {researchVerdict && <div className="l-notice"><p>Stored AI research finding: <strong>{researchVerdict.prediction??'Unavailable'}</strong>.</p><p>Primary model: {researchVerdict.primary_model_id.toUpperCase()}. Own model score: {researchVerdict.primary_model_score==null?'Unavailable':`${(researchVerdict.primary_model_score*100).toFixed(1)}%`} (uncalibrated).</p></div>}
      <p className="text-sm">Original AI result: {originalPrediction}. Consensus score: {originalConfidence}. Model outputs remain unchanged.</p>
      <p className="text-xs text-amber-700 dark:text-amber-300">Research annotations only. Sign-in does not establish physician identity or authorization.</p>
      <label className="l-field">Reviewer ID (unverified)<input aria-label="Reviewer ID (unverified)" required disabled={busy||saved} value={reviewer} onChange={e=>setReviewer(e.target.value)} className="l-input" /></label>
      <label className="l-field">Review action<select aria-label="Review action" value={kind} disabled={busy||saved} onChange={e=>setKind(e.target.value)} className="l-input">
        <option value="reviewed">Mark reviewed without an assessment</option><option value="independent_assessment">Record an independent assessment</option><option value="uncertain">Record uncertainty</option>
      </select></label>
      {kind!=='reviewed' && <label className="l-field">Human pathology assessment<select aria-label="Human pathology assessment" required={kind==='independent_assessment'} disabled={busy||saved} value={prediction} onChange={e=>setPrediction(e.target.value)} className="l-input"><option value="">Not assessed</option>{['Normal','Benign','Malignant'].map(p=><option key={p}>{p}</option>)}</select></label>}
      <label className="l-field">Reviewer note<textarea aria-label="Reviewer note" required maxLength={2000} rows={3} disabled={busy||saved} value={notes} onChange={e=>setNotes(e.target.value)} className="l-input" /></label>
      <label className="l-field">Optional self-reported confidence (%)<input aria-label="Optional self-reported confidence (%)" type="number" min="0" max="100" disabled={busy||saved} value={confidence} onChange={e=>setConfidence(e.target.value)} className="l-input" /></label>
      {error && <p role="alert" className="text-red-700">{error}</p>}
      <div className="flex gap-3"><button type="button" onClick={onClose} disabled={busy} className="l-button">Cancel</button><button type="submit" disabled={busy} className="l-button l-button-primary">{busy?'Saving review…':saved?'Retry reload':'Save review'}</button></div>
    </form>
  </div>;
}
