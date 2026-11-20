import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Layout } from './components/Layout';
import { Activity } from 'lucide-react';
import { motion } from 'motion/react';
import { Dashboard } from './components/Dashboard';
import { AnalysisResults } from './components/AnalysisResults';
import { CorrectionMode } from './components/CorrectionMode';
import { PatientList } from './components/PatientList';
import { Settings } from './components/Settings';
import { ModelStats } from './components/ModelStats';
import { DiagnosticReport } from './components/DiagnosticReport';
import { FeedbackModal } from './components/FeedbackModal';
import { API_BASE, responseError, submitReview } from './lib/api';
import type { AnalysisData, PatientRecord } from './types/patient';
import { mapPredictResponseToAnalysis, analysisDataToPatientAnalysis, mapApiPatientToPatientRecord, predictionToColor, displayResearchFinding } from './lib/inferenceAdapter';
import type { ApiPatientRow } from './lib/inferenceAdapter';
import { createLatestRequestGate } from './lib/latestRequest';

function readInitialThemeIsDark(): boolean {
  if (typeof window === 'undefined') return false;
  const stored = localStorage.getItem('theme');
  if (stored === 'dark') return true;
  if (stored === 'light') return false;
  return false;
}



function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = () => reject(new Error('Failed to read file'));
    r.readAsDataURL(file);
  });
}

function parseLesionSizeCm(lesion: string): number {
  const m = lesion.match(/(\d+(?:\.\d+)?)/);
  if (!m) return 0;
  return parseFloat(m[1]);
}

async function persistPatientToApi(record: PatientRecord, confidenceNum: number | null, sizeCm: number, stage: string) {
  const response = await fetch(`${API_BASE}/patients`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id: record.id,
      name: record.name,
      age: record.age,
      date: record.date,
      prediction: record.analysis.prediction,
      // Existing database column is NOT NULL. Zero is only a storage placeholder;
      // the versioned consensus metadata retains null, and adapters never display zero as a score.
      confidence: confidenceNum ?? 0,
      size_cm: sizeCm,
      stage,
      report_details: record.analysis.reportDetails,
      original_image: record.originalImage,
      heatmap_image: record.heatmapImage || '',
      multi_model_result: record.analysis.multiModel || {},
    }),
  });
  if (!response.ok) throw new Error(`Could not save research result (${response.status})`);
}

type Screen =
  | 'login'
  | 'dashboard'
  | 'analysis'
  | 'correction'
  | 'patient-list'
  | 'settings'
  | 'model-stats'
  | 'diagnostic-report';

const VALID_SCREENS: Screen[] = [
  'login',
  'dashboard',
  'analysis',
  'correction',
  'patient-list',
  'settings',
  'model-stats',
  'diagnostic-report',
];

function isValidScreen(value: string): value is Screen {
  return VALID_SCREENS.includes(value as Screen);
}

export interface CaseRecord {
  id: string;
  time: string;
  prediction: string;
  predictionColor: 'red' | 'green' | 'blue';
  confidence: string;
  status: string;
  statusColor: 'orange' | 'grey' | 'blue';
  originalImage: string;
}

function patientToCaseRow(p: PatientRecord): CaseRecord {
  const time = (() => {
    try {
      const d = new Date(p.date);
      return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    } catch {
      return p.date;
    }
  })();
  const humanVerified = p.analysis.reviewStatus==='reviewed';
  return {
    id: p.id,
    time,
    prediction: displayResearchFinding(p),
    predictionColor: p.analysis.predictionColor,
    confidence: p.analysis.confidence,
    status: humanVerified ? 'Review recorded (unverified identity)' : p.analysis.recordKind==='research' ? 'Research - unreviewed' : 'Historical - unreviewed',
    statusColor: humanVerified ? 'blue' : 'grey',
    originalImage: p.originalImage,
  };
}

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    return sessionStorage.getItem('isAuthenticated') === 'true';
  });

  const [currentScreen, setCurrentScreen] = useState<Screen>(() => {
    const auth = sessionStorage.getItem('isAuthenticated') === 'true';
    if (!auth) return 'login';
    const saved = sessionStorage.getItem('currentScreen');
    if (saved && isValidScreen(saved)) {
      const s = saved as Screen;
      if (s === 'analysis' || s === 'correction') return 'dashboard';
      if (s === 'diagnostic-report') return 'patient-list';
      return s;
    }
    return 'dashboard';
  });

  const [patientDatabase, setPatientDatabase] = useState<PatientRecord[]>([]);

  const [activePatientId, setActivePatientId] = useState<string | null>(null);
  const activePatientIdRef=useRef<string|null>(null);
  const refreshGate=useRef(createLatestRequestGate());
  const uploadGate=useRef(createLatestRequestGate());
  const [selectedReportPatientId, setSelectedReportPatientId] = useState<string | null>(null);

  const [currentAnalysis, setCurrentAnalysis] = useState<AnalysisData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [workflowError,setWorkflowError]=useState('');
  const [patientsLoading,setPatientsLoading]=useState(true);
  const [reviewOpen,setReviewOpen]=useState(false);

  const [userName, setUserName] = useState(() => {
    const saved = sessionStorage.getItem('userName');
    return saved || 'Research user (unverified)';
  });

  const [isDarkMode, setIsDarkMode] = useState(readInitialThemeIsDark);

  const toggleTheme = useCallback(() => {
    setIsDarkMode((d) => !d);
  }, []);

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDarkMode]);

  useEffect(() => {
    sessionStorage.setItem('currentScreen', currentScreen);
  }, [currentScreen]);

  useEffect(() => {
    sessionStorage.setItem('userName', userName);
  }, [userName]);

  const refreshPatients = useCallback(async () => {
    const request=refreshGate.current.begin();
    const requestedActiveId=activePatientIdRef.current;
    setPatientsLoading(true);
    try {
      const res=await fetch(`${API_BASE}/patients`);
      if(!res.ok)throw new Error(await responseError(res));
      const body=await res.json() as {patients:ApiPatientRow[]};
      const mapped=body.patients.map(mapApiPatientToPatientRecord);
      if(!request.isCurrent())return mapped;
      setPatientDatabase(mapped);setWorkflowError('');
      const active=requestedActiveId===activePatientIdRef.current
        ?mapped.find(p=>p.id===requestedActiveId):undefined;
      if(active)setCurrentAnalysis({...active.analysis,originalImage:active.originalImage,heatmapImage:active.heatmapImage});
      return mapped;
    }catch(err){if(request.isCurrent())setWorkflowError(err instanceof Error?err.message:'Saved records could not be loaded.');throw err;}
    finally{if(request.isCurrent())setPatientsLoading(false);}
  },[activePatientId]);
  useEffect(()=>{void refreshPatients().catch(()=>{});},[refreshPatients]);

  const recentCases = useMemo((): CaseRecord[] => {
    return patientDatabase.slice(0, 4).map(patientToCaseRow);
  }, [patientDatabase]);

  const selectedReportPatient = useMemo(() => {
    if (!selectedReportPatientId) return null;
    return patientDatabase.find((p) => p.id === selectedReportPatientId) ?? null;
  }, [patientDatabase, selectedReportPatientId]);

  const activePatient = useMemo(() => {
    if (!activePatientId) return null;
    return patientDatabase.find((p) => p.id === activePatientId) ?? null;
  }, [patientDatabase, activePatientId]);

  useEffect(() => {
    if (currentScreen === 'diagnostic-report' && !selectedReportPatient && !patientsLoading) {
      setCurrentScreen('patient-list');
    }
  }, [currentScreen, selectedReportPatient, patientsLoading]);

  const handleLogin = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const hospitalId = formData.get('hospital-id');
    const password = formData.get('password');

    if (hospitalId === 'demo@lumina.ai' && password === 'demo123') {
      sessionStorage.setItem('isAuthenticated', 'true');
      setIsAuthenticated(true);
      setCurrentScreen('dashboard');
    } else {
      alert('Invalid credentials. Use:\nHospital ID: demo@lumina.ai\nPassword: demo123');
    }
  };

  const handleUpload = useCallback(async (file: File, patientName: string, patientAge: string, sourceDomain = 'unknown') => {
    const request=uploadGate.current.begin();
    setIsAnalyzing(true);setWorkflowError('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('source_domain', sourceDomain);
      formData.append('contract_version', '1.2.0');

      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(await responseError(res));
      }

      const data = (await res.json()) as Record<string, unknown>;
      const originalDataUrl = await fileToDataUrl(file);
      if(!request.isCurrent())return;
      const analysisData: AnalysisData = {
        ...mapPredictResponseToAnalysis(data),
        originalImage: originalDataUrl,
      };
      const patientAnalysis = analysisDataToPatientAnalysis(analysisData);

      const id = `RESEARCH-${crypto.randomUUID()}`;

      const record: PatientRecord = {
        id,
        name: patientName.trim(),
        age: patientAge.trim(),
        date: new Date().toISOString(),
        originalImage: originalDataUrl,
        heatmapImage: analysisData.heatmapImage,
        analysis: patientAnalysis,
      };

      const confNum = null; // Abstention has no consensus score. Never borrow an individual score.

      await persistPatientToApi(record, confNum, 0, 'N/A');
      if(!request.isCurrent())return;

      // An older /patients reload must never replace this newly saved scan.
      refreshGate.current.invalidate();
      setPatientsLoading(false);
      activePatientIdRef.current=id;

      setPatientDatabase((prev) => {
        const filtered = prev.filter((p) => p.id !== id);
        return [record, ...filtered];
      });
      setActivePatientId(id);
      setCurrentAnalysis(analysisData);
      setCurrentScreen('analysis');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Analysis request failed.';
      if(request.isCurrent())setWorkflowError(`Could not analyze or save the scan. ${message}`);
    } finally {
      if(request.isCurrent())setIsAnalyzing(false);
    }
  }, []);

  const handleVerifyAndAccept = () => {
    if(currentAnalysis && activePatientId)setReviewOpen(true);
  };
  const handleCorrection = () => setCurrentScreen('correction');
  const handleCorrectionSubmit = async (assessment:string,notes:string,reviewerId:string) => {
    if(!activePatientId)throw new Error('No active research record.');
    await submitReview(activePatientId,reviewerId,notes,'independent_assessment',assessment);
  };

  const handleCorrectionCancel = () => {
    setCurrentScreen('analysis');
  };

  const handleNavigate = (screen: string) => {
    if (isValidScreen(screen)) {
      setCurrentScreen(screen);
    }
  };

  const handleViewReport = (patient: PatientRecord) => {
    setSelectedReportPatientId(patient.id);
    setCurrentScreen('diagnostic-report');
  };

  const handleDeletePatient = useCallback(async (id: string) => {
    try {
      const res = await fetch(`${API_BASE}/patients/${encodeURIComponent(id)}`, {
        method: 'DELETE',
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Delete failed (${res.status})`);
      }
      setPatientDatabase((prev) => prev.filter((p) => p.id !== id));
      if(activePatientIdRef.current===id) {
        activePatientIdRef.current=null;
        refreshGate.current.invalidate();
        setPatientsLoading(false);
        setCurrentAnalysis(null);
      }
      setActivePatientId((prev) => (prev === id ? null : prev));
      setSelectedReportPatientId((prev) => (prev === id ? null : prev));
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Delete request failed.';
      alert(`Could not delete patient record. ${msg}`);
    }
  }, []);

  const handleBackToPatientList = () => {
    setSelectedReportPatientId(null);
    setCurrentScreen('patient-list');
  };

  const handleUpdateUserName = (newName: string) => {
    setUserName(newName);
  };

  if (currentScreen === 'diagnostic-report' && selectedReportPatient) {
    return (
      <DiagnosticReport patient={selectedReportPatient} onBack={handleBackToPatientList} onReviewSaved={async()=>{await refreshPatients();}} />
    );
  }

  if (currentScreen === 'diagnostic-report' && !selectedReportPatient && !patientsLoading) {
    return null;
  }

  if (!isAuthenticated) {
    return <div className="l-report" style={{display:'grid',placeItems:'center'}}><section className="l-card" style={{width:'100%',maxWidth:470,padding:36}}><div className="l-brand" style={{marginBottom:28}}><div className="l-logo"><Activity/></div><div><strong>Lumina</strong><small>Breast AI Research</small></div></div><span className="l-eyebrow">Research workspace</span><h1 style={{fontSize:28}}>Explore. Inspect. Review.</h1><p className="l-muted" style={{margin:'12px 0 24px'}}>Frozen models and transparent research findings, with independent human review.</p><form className="l-stack" onSubmit={handleLogin}><label className="l-field" htmlFor="hospital-id">Hospital ID<input className="l-input" id="hospital-id" name="hospital-id" autoComplete="username" placeholder="demo@lumina.ai" required/></label><label className="l-field" htmlFor="password">Password<input className="l-input" type="password" id="password" name="password" autoComplete="current-password" placeholder="demo123" required/></label><button className="l-button l-button-primary" type="submit">Open Demo Workspace</button></form><p className="l-helper" style={{marginTop:20}}>Local research prototype. Demo sign-in does not verify identity. No clinical deployment readiness is established.</p></section></div>;
  }

  return (
    <Layout
      currentScreen={currentScreen}
      onNavigate={handleNavigate}
      userName={userName}
      isDimmed={currentScreen === 'correction'}
      isDarkMode={isDarkMode}
      toggleTheme={toggleTheme}
    >
      {workflowError && <div role="alert" className="m-4 border border-red-400 bg-red-50 text-red-900 rounded p-4">{workflowError} <button onClick={()=>void refreshPatients().catch(()=>{})} className="underline">Retry loading records</button></div>}
      {patientsLoading && <p role="status" className="px-8 py-2">Loading saved records…</p>}
      {reviewOpen && activePatientId && currentAnalysis && <FeedbackModal caseId={activePatientId} originalPrediction={currentAnalysis.prediction} originalConfidence={currentAnalysis.confidence} researchVerdict={currentAnalysis.multiModel?.research_verdict} onClose={()=>setReviewOpen(false)} onSuccess={async()=>{await refreshPatients();}} />}
      {currentScreen === 'dashboard' && (
        <Dashboard
          onUpload={handleUpload}
          recentCases={recentCases}
          patientDatabase={patientDatabase}
          userName={userName}
          isAnalyzing={isAnalyzing}
        />
      )}
      {currentScreen === 'patient-list' && (
        <PatientList
          patientDatabase={patientDatabase}
          onViewReport={handleViewReport}
          onDeletePatient={handleDeletePatient}
        />
      )}
      {currentScreen === 'model-stats' && <ModelStats />}
      {currentScreen === 'settings' && (
        <Settings onNavigate={handleNavigate} userName={userName} onUpdateUserName={handleUpdateUserName} />
      )}
      {currentScreen === 'analysis' && currentAnalysis && (
        <AnalysisResults
          analysisData={currentAnalysis}
          onVerifyAndAccept={handleVerifyAndAccept}
          onCorrection={handleCorrection}
          onViewReport={activePatient?()=>handleViewReport(activePatient):undefined}
        />
      )}
      {currentScreen === 'correction' && (
        <CorrectionMode
          currentPrediction={currentAnalysis?.multiModel?.research_verdict?.prediction ?? currentAnalysis?.prediction ?? 'No validated consensus available'}
          originalImage={activePatient?.originalImage}
          onSubmit={handleCorrectionSubmit}
          onSaved={async()=>{await refreshPatients();setCurrentScreen('patient-list');}}
          onCancel={handleCorrectionCancel}
        />
      )}
    </Layout>
  );
}
