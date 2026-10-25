import { useState, useRef } from 'react';
import { commitReviewOnce } from '../lib/api';
interface CorrectionModeProps {
  currentPrediction: string; originalImage?: string;
  onSubmit: (assessment: string, notes: string, reviewerId: string) => void | Promise<void>;
  onCancel: () => void;
  onSaved?: () => void | Promise<void>;
}
export function CorrectionMode({ currentPrediction, originalImage, onSubmit, onCancel, onSaved }: CorrectionModeProps) {
  const [assessment,setAssessment]=useState('');const [notes,setNotes]=useState('');const [reviewer,setReviewer]=useState('');
  const [busy,setBusy]=useState(false);const [error,setError]=useState('');
  const [saved,setSaved]=useState(false);const commit=useRef({saved:false});
  const save=async(e:React.FormEvent)=>{
    e.preventDefault();setError('');
    if(!assessment || !notes.trim() || !reviewer.trim()){setError('Select an assessment and enter a reviewer ID and note.');return;}
    setBusy(true);try{await commitReviewOnce(commit.current,async()=>{await onSubmit(assessment,notes,reviewer);},async()=>{setSaved(true);await onSaved?.();});}
    catch(err){setSaved(commit.current.saved);setError(`${commit.current.saved?'Assessment is saved; retry reload. ':''}${err instanceof Error?err.message:'Assessment could not be saved.'}`);}finally{setBusy(false);}
  };
  return <div className="p-8 grid grid-cols-1 lg:grid-cols-2 gap-8">
    <div>{originalImage && <img src={originalImage} alt="Original scan for independent human review" className="max-h-[650px] w-full object-contain bg-black" />}<p className="mt-3">Frozen AI result: {currentPrediction}</p></div>
    <form onSubmit={save} className="space-y-5 rounded-xl bg-white dark:bg-gray-800 p-6">
      <h2 className="text-xl font-semibold">Independent human assessment</h2><p>This records your assessment separately. No AI label, score, heatmap or measurement is overwritten.</p>
      <label className="block">Reviewer ID (unverified)<input aria-label="Assessment reviewer ID" required disabled={busy||saved} value={reviewer} onChange={e=>setReviewer(e.target.value)} className="block border rounded p-2 w-full text-gray-900" /></label>
      <label className="block">Human pathology assessment<select aria-label="Independent pathology assessment" required disabled={busy||saved} value={assessment} onChange={e=>setAssessment(e.target.value)} className="block border rounded p-2 w-full text-gray-900"><option value="">Select assessment</option>{['Normal','Benign','Malignant'].map(p=><option key={p}>{p}</option>)}</select></label>
      <label className="block">Reviewer note<textarea aria-label="Assessment note" required disabled={busy||saved} maxLength={2000} value={notes} onChange={e=>setNotes(e.target.value)} rows={4} className="block border rounded p-2 w-full text-gray-900" /></label>
      <p className="text-sm text-amber-700">Reviewer identity and clinical authority are not verified by this prototype.</p>
      {error && <p role="alert" className="text-red-700">{error}</p>}
      <button type="submit" disabled={busy} className="bg-blue-600 text-white px-5 py-3 rounded">{busy?'Saving assessment…':saved?'Retry reload':'Save independent assessment'}</button>
      <button type="button" disabled={busy} onClick={onCancel} className="border px-5 py-3 rounded ml-3">Cancel</button>
    </form>
  </div>;
}
