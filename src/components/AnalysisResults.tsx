import { useState } from 'react';
import { ArrowLeft, FileText } from 'lucide-react';
import type { AnalysisData } from '../types/patient';
import { ResearchModelResults } from './ResearchModelResults';
import { ReviewSummary } from './ReviewSummary';
import { ExploratorySegmentation } from './ExploratorySegmentation';
import { ResearchVerdictSummary, BiradsSummary } from './ResearchVerdictSummary';
interface Props {analysisData:AnalysisData;onVerifyAndAccept:()=>void;onCorrection:()=>void;onViewReport?:()=>void}
export function AnalysisResults({analysisData:a,onVerifyAndAccept,onCorrection,onViewReport}:Props){
 const [heatmap,setHeatmap]=useState(false);const has=!!a.heatmapImage;
 return <div className="l-page"><header className="l-page-heading"><div><span className="l-eyebrow">Analysis workspace</span><h1>Ultrasound Research Analysis</h1><p className="l-muted">Provisional findings with independent image evidence and model provenance.</p></div><span className="l-badge l-badge-blue">Research inference results</span></header>
 <div className="l-split"><div className="l-stack"><section className="l-card"><div className="l-section-head"><h2>Ultrasound Viewer</h2><span className="l-badge">{heatmap?'Class attribution':'Original image'}</span></div><div className="l-image">{a.originalImage?<img src={heatmap&&has?a.heatmapImage:a.originalImage} alt={heatmap&&has?'BUS-BRA pathology Grad-CAM attribution':'Original scan'}/>:<p>No image</p>}</div><div className="l-actions" style={{marginTop:16}}><button className="l-button" role="switch" aria-checked={heatmap&&has} aria-label={has?heatmap?'Switch to original scan':'Switch to model heatmap':'Model heatmap unavailable'} disabled={!has} onClick={()=>setHeatmap(v=>!v)}>{has?heatmap?'Show original scan':'Show Grad-CAM attribution':'Heatmap unavailable'}</button></div><p className="l-image-caption">Genuine BRA Grad-CAM = class attribution. It is not segmentation or a clinically validated explanation.</p></section><BiradsSummary result={a.multiModel}/></div><ResearchVerdictSummary result={a.multiModel}/></div>
 <section><div className="l-section-head"><div><span className="l-eyebrow">Model comparison</span><h2>Four independent research outputs</h2></div></div>{a.multiModel&&<ResearchModelResults result={a.multiModel}/>}</section>
 <ExploratorySegmentation result={a.multiModel?.segmentation} originalImage={a.originalImage}/>
 <div className="l-split"><ReviewSummary analysis={{...a,notes:a.notes??''}}/><section className="l-card"><span className="l-eyebrow">Next step</span><h2>Keep human judgment independent</h2><p className="l-muted">Review metadata never replaces the original model labels, scores, attribution or segmentation.</p><div className="l-actions" style={{marginTop:20}}><button className="l-button l-button-primary" onClick={onVerifyAndAccept}>Record Review</button><button className="l-button" onClick={onCorrection}>Record Independent Assessment</button>{onViewReport&&<button className="l-button" onClick={onViewReport}><FileText size={17}/>View Report</button>}</div></section></div></div>;
}
