import type { PatientAnalysis } from '../types/patient';

export function ReviewSummary({ analysis }: { analysis: PatientAnalysis }) {
  const review = analysis.humanReview;
  return <section className="l-card l-review research-report-section break-inside-avoid">
    <h3 className="font-semibold text-gray-900 dark:text-white">Human review: {analysis.reviewStatus === 'reviewed' ? 'Recorded' : 'Not recorded'}</h3>
    <p className="text-sm text-gray-500 dark:text-gray-400">{analysis.recordKind === 'research' ? 'Versioned research analysis' : 'Historical record; original data retained'}. Review does not change the original AI outputs.</p>
    {analysis.storageWarnings?.map((warning,i)=><p key={i} role="note" className="text-sm text-amber-700">{warning}</p>)}
    {review && <div className="mt-3 space-y-2 text-sm text-gray-700 dark:text-gray-300">
      <p>Independent human assessment: {review.prediction ?? 'Not assessed'}</p>
      <p>Reviewer: {review.reviewer_id ?? 'Not recorded'} · Identity unverified</p>
      <p>Review type: {review.kind?.replaceAll('_', ' ') ?? 'Historical review'}</p>
      <p className="whitespace-pre-wrap">Reviewer note: {review.notes}</p>
      <p>Recorded: {review.created_at ? new Date(review.created_at).toLocaleString() : 'Unknown'}</p>
      <p>Saved review entries: {analysis.reviewHistory?.length ?? 'Unknown (history not recorded)'}</p>
      {analysis.reviewHistory && <details className="l-limitations"><summary>View review history</summary>{analysis.reviewHistory.map((h,i)=><div className="l-history-entry" key={h.feedback_id??i}><p>{h.created_at?new Date(h.created_at).toLocaleString():'Date unavailable'} · {h.reviewer_id??'Reviewer not recorded'} (unverified)</p><p>{h.kind?.replaceAll('_',' ')??'Review'}: {h.prediction??'Not assessed'}</p><p>{h.notes}</p></div>)}</details>}
    </div>}
  </section>;
}
