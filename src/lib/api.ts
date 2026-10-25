export const API_BASE = import.meta.env?.VITE_API_BASE_URL ?? 'http://localhost:8000';

export async function commitReviewOnce(state: {saved:boolean}, submit:()=>Promise<unknown>, reload:()=>Promise<unknown>) {
  if(!state.saved){await submit();state.saved=true;}
  await reload();
}

export async function responseError(response: Response): Promise<string> {
  try {
    const data = await response.json();
    return typeof data.detail === 'string' ? data.detail : `Request rejected (${response.status}); check the required fields.`;
  } catch { return `Request failed (${response.status})`; }
}

export async function submitReview(caseId: string, reviewer: string, notes: string, kind = 'reviewed', prediction: string | null = null, confidence: number | null = null) {
  const response = await fetch(`${API_BASE}/feedback/submit`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ case_id: caseId, doctor_id: reviewer.trim(), reasoning: notes.trim(), feedback_type: kind,
      corrected_prediction: prediction, corrected_confidence: confidence }),
  });
  if (!response.ok) throw new Error(await responseError(response));
  return response.json();
}
