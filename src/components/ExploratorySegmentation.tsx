import { useState } from 'react';
import type { SegmentationResult } from '../types/patient';

export function ExploratorySegmentation({result,originalImage}:{result?:SegmentationResult;originalImage:string}) {
  const [overlay,setOverlay]=useState(true);
  const available=result?.status==='ok';
  return <section className="l-card research-report-section space-y-4" aria-label="Exploratory UCLM segmentation">
    <h2 className="text-xl font-semibold">Exploratory Lesion Segmentation — UCLM V3</h2>
    <p className="text-sm">Segmentation predicts a foreground mask. Grad-CAM separately shows class attribution. Both are research outputs without clinical validation.</p>
    {!available ? <p role="note">{result?`Segmentation ${result.status}: ${result.reason}${result.error_code?` (${result.error_code})`:''}`:'No segmentation available in this historical or contract 1.0 record.'}</p> : <>
      <p>Declared source: UCLM (unverified). {result.empty_mask?'Empty AI-predicted foreground mask; this does not establish a normal image.':'AI-predicted foreground is shown in green.'}</p>
      <div className="l-segmentation-grid grid grid-cols-1 md:grid-cols-3 gap-3">
        <figure><img src={originalImage} alt="Segmentation original ultrasound" className="w-full bg-black"/><figcaption>Original ultrasound</figcaption></figure>
        <figure><img src={result.predicted_mask!} alt="AI-predicted binary lesion mask" className="w-full bg-black"/><figcaption>AI-predicted binary foreground</figcaption></figure>
        <figure><div className="relative w-full bg-black" style={{aspectRatio:`${result.image_width} / ${result.image_height}`}}>
          <img src={originalImage} alt="Ultrasound beneath segmentation overlay" className="absolute inset-0 w-full h-full object-contain"/>
          {overlay && <img src={result.overlay!} alt="Transparent AI segmentation overlay" className="absolute inset-0 w-full h-full object-contain pointer-events-none"/>}
        </div><figcaption>Original coordinates with optional overlay</figcaption></figure>
      </div>
      <button type="button" role="switch" aria-label="Segmentation overlay" aria-checked={overlay} onClick={()=>setOverlay(v=>!v)} className="l-button research-report-actions">Segmentation overlay {overlay?'on':'off'}</button>
      <p className="text-xs">Original dimensions: {result.image_width} × {result.image_height}. Threshold: {result.threshold}; exact ties select background. Resize: stretch to 256 × 256, binary decision, nearest-neighbor restoration. Fine detail may be lost.</p>
    </>}
    {result && <details><summary>Segmentation provenance and limitations</summary><p>Model: {result.model_id} · {result.model_version}</p><p>Checkpoint: {result.checkpoint_filename} ({result.checkpoint_verification})</p><p className="break-all">SHA-256: {result.checkpoint_sha256??'Not verified'}</p><p className="break-all">Pinned SHA-256: {result.expected_checkpoint_sha256}</p><p>{result.original_coordinate_system}</p><ul>{result.limitations.map(t=><li key={t}>{t}</li>)}</ul></details>}
  </section>;
}
