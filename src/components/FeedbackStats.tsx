import { useState, useEffect } from 'react';
import { API_BASE, responseError } from '../lib/api';
export function FeedbackStats() {
  const [stats,setStats]=useState<any>(null);const [error,setError]=useState('');const [loading,setLoading]=useState(true);
  const refresh=async()=>{try{const r=await fetch(`${API_BASE}/feedback/stats`);if(!r.ok)throw new Error(await responseError(r));setStats(await r.json());setError('');}catch(err){setError(err instanceof Error?err.message:'Statistics unavailable');}finally{setLoading(false);}};
  useEffect(()=>{void refresh();},[]);
  if(loading)return <p role="status">Loading research workflow statistics…</p>;
  if(error)return <div role="alert"><p>{error}</p><button onClick={refresh}>Retry statistics</button></div>;
  if(!stats)return <p>No workflow statistics available.</p>;
  const c=stats.counts;
  return <section className="space-y-5"><div className="flex justify-between"><h2 className="text-xl font-semibold">Research workflow statistics</h2><button onClick={refresh} className="border rounded px-4 py-2">Refresh statistics</button></div>
    <p>{stats.note}</p><div className="grid grid-cols-2 lg:grid-cols-4 gap-4">{[['Saved records',c.total_records],['Versioned research',c.versioned_records],['Historical records',c.historical_records],['Reviewed records',c.reviewed_records],['Unreviewed records',c.unreviewed_records],['Abstentions',c.abstentions],['Feedback submissions',stats.total_feedback],['Independent assessments',c.independent_assessments]].map(([label,value])=><div key={String(label)} className="rounded-xl border p-4"><p>{label}</p><p className="text-2xl">{value}</p></div>)}</div>
    <p>Diagnostic accuracy: Unavailable. Average model confidence: Unavailable. Workflow reviews are not diagnostic ground truth.</p>
    <h3 className="font-semibold">Unavailable model observations</h3><p>{Object.entries(c.unavailable_model_observations).map(([k,v])=>`${k}: ${v}`).join('; ')||'None recorded'}</p>
    <h3 className="font-semibold">Recent feedback (up to 50 entries)</h3>{stats.recent_feedback.length?stats.recent_feedback.map((f:any)=><article key={f.id} className="border rounded p-4 space-y-1"><p>{f.case_id} · {f.record_kind} · {f.feedback_type.replaceAll('_',' ')}</p><p>Original AI: {f.original_prediction}</p><p>Human assessment: {f.human_assessment??'Not assessed'}</p><p>Reviewer: {f.reviewer_id} (identity unverified)</p><p>{f.reasoning}</p></article>):<p>No feedback recorded.</p>}
  </section>;
}
